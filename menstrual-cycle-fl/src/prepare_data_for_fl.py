# prepare_data_for_fl.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# IMPORTANT: Ensure your new data_preprocessing.py is in the 'src' folder
# and accessible. Let's assume you've named it data_preprocessing_final.py to avoid confusion.
from data_preprocessing import (
    engineer_features,
    split_data_by_client,
    create_and_fit_preprocessor,
    apply_preprocessing,
    simulate_federated_data_split
)

def main():
    print("--- Starting Data Preparation for Federated Learning ---")

    # Load and Engineer Features
    df_raw = pd.read_csv("FedCycleData071012 (2).csv", skipinitialspace=True)
    df_featured = engineer_features(df_raw)

    # Split Data by Client ID 
    train_df, val_df, test_df, _, _, _ = split_data_by_client(df_featured, test_size=0.3, val_size=0.5)

    # Data Clipping
    clip_cols = ['next_cycle_length', 'L1_LengthofCycle', 'L2_LengthofCycle', 'L3_LengthofCycle', 'roll_mean_3_cycles', 'roll_std_3_cycles']
    for c in clip_cols:
        if c in train_df.columns:
            lo, hi = train_df[c].quantile([0.01, 0.99])
            for d in (train_df, val_df, test_df):
                if c in d.columns:
                    d[c] = d[c].clip(lo, hi)

    # Create and Fit the Global Preprocessor
    # This is a critical centralized step.
    preprocessor, numeric, categorical, text_raw = create_and_fit_preprocessor(train_df)

    # Apply Preprocessing to get Full Datasets 
    X_train, y_train = apply_preprocessing(train_df, preprocessor, text_raw)
    X_val, y_val = apply_preprocessing(val_df, preprocessor, text_raw)
    X_test, y_test = apply_preprocessing(test_df, preprocessor, text_raw)

    # Perform Centralized Feature Selection
    # We train a temporary RF on the full training set to find the best features.
    print("Training temporary RF for feature selection...")
    rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)

    # Get all feature names from the fitted preprocessor
    ohe_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical)
    tfidf_names = preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
    all_feature_names = numeric + list(ohe_names) + list(tfidf_names)

    importances = pd.Series(rf_selector.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    
    N_TOP_FEATURES = 40
    top_features = importances.head(N_TOP_FEATURES).index.tolist()
    print(f"Selected top {N_TOP_FEATURES} features.")

    name_to_idx = {name: i for i, name in enumerate(all_feature_names)}
    top_idx = np.array([name_to_idx[f] for f in top_features])

    # Save Artifacts and Data 
    # Create a directory to store these files
    os.makedirs('fl_artifacts', exist_ok=True)

    # Save the fitted preprocessor and top feature indices
    joblib.dump(preprocessor, 'fl_artifacts/preprocessor.joblib')
    np.save('fl_artifacts/top_40_feature_indices.npy', top_idx)

    # Save the global test set (already processed and sliced) for server-side evaluation
    X_test_sel = X_test[:, top_idx]
    np.savez('fl_artifacts/global_test_set.npz', X_test=X_test_sel, y_test=y_test)
    
    # Simulate and Save Data for Each Client 
    NUM_CLIENTS = 3 # Or however many you want to simulate
    client_dfs = simulate_federated_data_split(train_df, num_clients=NUM_CLIENTS)

    for i, client_df in enumerate(client_dfs):
        # Apply the SAME fitted preprocessor and feature selection to each client's data
        X_client, y_client = apply_preprocessing(client_df, preprocessor, text_raw)
        X_client_sel = X_client[:, top_idx]
        
        # Save this client's specific data
        client_filepath = f'fl_artifacts/client_{i}_data.npz'
        np.savez(client_filepath, X_train=X_client_sel, y_train=y_client)
        print(f"Saved data for client {i} at {client_filepath} with shape {X_client_sel.shape}")

    print("Data Preparation Complete")
    print("Artifacts are saved in the 'fl_artifacts' directory.")

if __name__ == "__main__":
    main()