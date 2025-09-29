# src/centralized_baseline.py

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Import functions from your data_preprocessing module
from data_preprocessing import engineer_features, split_data_by_client, create_and_fit_preprocessor, apply_preprocessing

def main():
    print("Running Centralized Baseline Model Training")

    # Load and Engineer Features
    df_raw = pd.read_csv("FedCycleData071012 (2).csv", skipinitialspace=True)
    df_featured = engineer_features(df_raw)

    # Split Data by Client ID (non-IID)
    train_df, val_df, test_df, _, _, _ = split_data_by_client(df_featured)
    print(f"Data split into Train: {train_df.shape[0]}, Val: {val_df.shape[0]}, Test: {test_df.shape[0]} samples.")

    # Data Clipping 
    print("Clipping data based on training set quantiles...")
    clip_cols = ['next_cycle_length', 'L1_LengthofCycle', 'L2_LengthofCycle', 'L3_LengthofCycle', 'roll_mean_3_cycles', 'roll_std_3_cycles']
    for c in clip_cols:
        if c in train_df.columns:
            lo, hi = train_df[c].quantile([0.01, 0.99])
            for d in (train_df, val_df, test_df):
                # Ensure the column exists in the dataframe before clipping
                if c in d.columns:
                    d[c] = d[c].clip(lo, hi)

    # Create and Fit Preprocessor on TRAINING data only
    preprocessor, numeric_features, categorical_features, text_cols_raw = create_and_fit_preprocessor(train_df)
    
    # Apply preprocessing to all splits
    X_train, y_train = apply_preprocessing(train_df, preprocessor, text_cols_raw)
    X_val, y_val = apply_preprocessing(val_df, preprocessor, text_cols_raw)
    X_test, y_test = apply_preprocessing(test_df, preprocessor, text_cols_raw)
    print(f"Preprocessed feature shapes: X_train: {X_train.shape}, X_test: {X_test.shape}")

    results = []

    # Random Forest 
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10, max_depth=15, max_features=0.7, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    results.append({"Model": "Random Forest", "RMSE": rmse_rf, "MAE": mae_rf})
    print(f"  RF -> RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}")

    # XGBoost 
    print("\nTraining XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "eta": 0.02, "max_depth": 4, "seed": 42}
    bst = xgb.train(params, dtrain, num_boost_round=5000, evals=[(dval, "eval")], early_stopping_rounds=200, verbose_eval=False)
    y_pred_xgb = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    results.append({"Model": "XGBoost", "RMSE": rmse_xgb, "MAE": mae_xgb})
    print(f"  XGBoost -> RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}")

    # MLP (Centralized) 
    def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        # Check if text transformer exists and has features
        if 'text' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['text'].named_steps['tfidf'], 'get_feature_names_out'):
            tfidf_feature_names = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
        else:
            tfidf_feature_names = []
        
        return numeric_features + list(ohe_feature_names) + list(tfidf_feature_names)

    all_feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features)

    # Use the already trained RF model for feature importances
    importances = pd.DataFrame({
        'feature': all_feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    N_TOP_FEATURES = 40
    top_features = importances.head(N_TOP_FEATURES)['feature'].tolist()
    print(f"\nSelected top {N_TOP_FEATURES} features for MLP.")

    # Get the indices of these top features
    name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
    top_idx = [name_to_idx[f] for f in top_features]

    # Create new, sliced datasets for the MLP
    X_train_sel = X_train[:, top_idx]
    X_val_sel = X_val[:, top_idx]
    X_test_sel = X_test[:, top_idx]
        
    print(f"MLP feature shape after selection: {X_train_sel.shape}")

    # Train the MLP on the selected features
    # Train the MLP on the selected features
    print("\nTraining Centralized MLP on selected features...")
    input_dim = X_train_sel.shape[1] # This will be 40

    # Define the model architecture
    mlp = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1) # Output layer for regression
    ])

    # Compile the model
    mlp.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define an early stopping callback to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True
    )

    # Fit the model
    history = mlp.fit(
        X_train_sel, 
        y_train, 
        validation_data=(X_val_sel, y_val), 
        epochs=300, 
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0  # Set to 1 if you want to see training progress
    )

    print("MLP training complete.")
    y_pred_mlp = mlp.predict(X_test_sel).ravel()

    rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
    mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
    results.append({"Model": "MLP (Centralized)", "RMSE": rmse_mlp, "MAE": mae_mlp})
    print(f"  MLP -> RMSE: {rmse_mlp:.4f}, MAE: {mae_mlp:.4f}")
    
    # Results Summary 
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n--- Centralized Model Comparison ---")
    print(results_df)

if __name__ == "__main__":
    main()
