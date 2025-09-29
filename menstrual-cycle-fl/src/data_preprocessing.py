# src/data_preprocessing.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# Helper function 

def is_yes_like(x):
    if pd.isna(x): return np.nan
    return 1.0 if str(x).strip().lower() in {'y', 'yes', 'true', 't', '1'} else 0.0

def squeeze_array(x):
    return x.ravel()

# Core pre-processing logic 

def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Applies all cleaning and feature engineering steps."""
    df = df_raw.copy()
    
    # Correct data types
    df['CycleNumber'] = pd.to_numeric(df['CycleNumber'], errors='coerce')

    # Coalesce paired columns
    pair_fallbacks = [('Age', 'AgeM'), ('Maristatus', 'MaristatusM'), ('Religion', 'ReligionM'), 
                      ('Ethnicity', 'EthnicityM'), ('Schoolyears', 'SchoolyearsM'), ('Medvits', 'MedvitsM'),
                      ('Medvitexplain', 'MedvitexplainM'), ('Livingkids', 'LivingkidsM'), ('Nextpreg', 'NextpregM'),
                      ('Spousesame', 'SpousesameM')]
    for primary, fallback in pair_fallbacks:
        if primary in df.columns and fallback in df.columns:
            df[primary] = df[primary].fillna(df[fallback])
            df = df.drop(columns=fallback, errors='ignore')

    # Sort for time-series features
    df = df.sort_values(['ClientID', 'CycleNumber']).reset_index(drop=True)

    # Define Target Variable and drop rows where it's missing
    df['LengthofCycle'] = pd.to_numeric(df['LengthofCycle'], errors='coerce')
    df['next_cycle_length'] = df.groupby('ClientID')['LengthofCycle'].shift(-1)
    df = df.dropna(subset=['next_cycle_length']).copy()
    
    # Time-Series Features (Lags, Rolling, Expanding, EWMA)
    for i in range(1, 4):
        df[f'L{i}_LengthofCycle'] = df.groupby('ClientID')['LengthofCycle'].shift(i)
    lag_cols = [f'L{i}_LengthofCycle' for i in range(1, 4)]
    df['roll_mean_3_cycles'] = df[lag_cols].mean(axis=1)
    df['roll_std_3_cycles'] = df[lag_cols].std(axis=1)
    df['ewma_3_cycles'] = df.groupby('ClientID')['LengthofCycle'].transform(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
    df['user_mean_cycle_length_so_far'] = df.groupby('ClientID')['LengthofCycle'].transform(lambda x: x.shift(1).expanding().mean())
    df['user_std_cycle_length_so_far'] = df.groupby('ClientID')['LengthofCycle'].transform(lambda x: x.shift(1).expanding().std()).fillna(0)
    df['user_cycle_count'] = df.groupby('ClientID').cumcount()

    # Text-Based Feature Engineering
    if 'Gynosurgeries' in df.columns:
        surg_text = df['Gynosurgeries'].astype(str).str.lower()
        df['had_c_section'] = surg_text.str.contains('c-section|c section|cesarean', regex=True).astype(float)
        df['had_d_and_c'] = surg_text.str.contains('d&c|d & c|d and c', regex=True).astype(float)
        df['had_laparoscopy'] = surg_text.str.contains('laparoscopy|laproscopy', regex=True).astype(float)

    # Binary Flags
    for col in ['CycleWithPeakorNot', 'IntercourseInFertileWindow', 'UnusualBleeding', 'Breastfeeding']:
        if col in df.columns:
            df[f'{col}_flag'] = df[col].apply(is_yes_like)

    # Correct data type for numerically-coded categoricals
    cat_as_num_cols = ['Group', 'ReproductiveCategory', 'Maristatus', 'Religion', 'Ethnicity', 'Schoolyears', 'Reprocate', 'Method', 'Prevmethod', 'Whychart']
    for col in cat_as_num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').astype(str).replace('<NA>', np.nan)
            
    # Drop high-missingness and redundant columns
    WHITELIST = {'Age', 'BMI', 'Abortions', 'Miscarriages', 'Numberpreg', 'Livingkids', 'Gynosurgeries', 'Urosurgeries', 'Medvits'}
    missing_frac = df.isna().mean()
    cols_to_drop_missing = missing_frac[(missing_frac > 0.9) & (~missing_frac.index.isin(WHITELIST))].index
    cols_to_drop_other = ['CycleWithPeakorNot', 'IntercourseInFertileWindow', 'UnusualBleeding', 'Breastfeeding', 'LengthofCycle', 'CycleNumber']
    all_cols_to_drop = list(cols_to_drop_missing) + [c for c in cols_to_drop_other if c in df.columns]
    df = df.drop(columns=all_cols_to_drop, errors='ignore')

    return df

def create_and_fit_preprocessor(df_train: pd.DataFrame):
    """Creates the ColumnTransformer and fits it ONLY on the training data."""
    
    target = 'next_cycle_length'
    features_to_exclude = [target, 'ClientID']
    all_features = [col for col in df_train.columns if col not in features_to_exclude]
    
    numeric_features = [col for col in all_features if pd.api.types.is_numeric_dtype(df_train[col])]
    categorical_features = [col for col in all_features if pd.api.types.is_object_dtype(df_train[col]) or pd.api.types.is_categorical_dtype(df_train[col])]
    text_features_raw = [c for c in ['Gynosurgeries', 'Urosurgeries', 'Medvitexplain'] if c in all_features]
    
    # Create combined text feature
    if text_features_raw:
        df_train['combined_text'] = df_train[text_features_raw].fillna('').agg(' '.join, axis=1)
    else:
        df_train['combined_text'] = ''
        
    text_features = ['combined_text']
    categorical_features = [c for c in categorical_features if c not in text_features_raw]

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    text_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='')), ('squeeze', FunctionTransformer(squeeze_array)), ('tfidf', TfidfVectorizer(max_features=50, stop_words='english'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_features),
        ],
        remainder='drop'
    )
    
    # Fit the preprocessor
    preprocessor.fit(df_train)

    return preprocessor, numeric_features, categorical_features, text_features_raw

def apply_preprocessing(df: pd.DataFrame, preprocessor, text_features_raw: list):
    """Applies a fitted preprocessor to a dataframe."""
    df_proc = df.copy()
    
    # Create combined text feature
    if text_features_raw:
        df_proc['combined_text'] = df_proc[text_features_raw].fillna('').agg(' '.join, axis=1)
    else:
        df_proc['combined_text'] = ''

    features_df = df_proc.drop(columns=['next_cycle_length', 'ClientID'], errors='ignore')
    
    X = preprocessor.transform(features_df)
    y = df_proc['next_cycle_length'].values
    
    return X.astype(np.float32), y.astype(np.float32)

def split_data_by_client(df: pd.DataFrame, test_size=0.3, val_size=0.5, random_state=42):
    """Performs a user-level (non-IID) split."""
    client_ids = df['ClientID'].dropna().unique()
    train_clients, temp_clients = train_test_split(client_ids, test_size=test_size, random_state=random_state)
    val_clients, test_clients = train_test_split(temp_clients, test_size=val_size, random_state=random_state)
    
    train_df = df[df['ClientID'].isin(train_clients)].copy()
    val_df = df[df['ClientID'].isin(val_clients)].copy()
    test_df = df[df['ClientID'].isin(test_clients)].copy()
    
    return train_df, val_df, test_df, train_clients, val_clients, test_clients

def simulate_federated_data_split(train_df: pd.DataFrame, num_clients: int):
    """Simulates a non-IID federated data split from the training dataframe."""
    client_data = []
    user_ids = train_df['ClientID'].unique()
    np.random.shuffle(user_ids)
    
    user_splits = np.array_split(user_ids, num_clients)
    
    for i in range(num_clients):
        client_user_ids = user_splits[i]
        client_df = train_df[train_df['ClientID'].isin(client_user_ids)]
        client_data.append(client_df)
        
    return client_data