import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

def load_and_preprocess_data(csv_path, target_column='LengthofCycle', test_size=0.2, random_state=42):
    """
    Load and preprocess the menstrual cycle data.
    
    Args:
        csv_path (str): Path to the CSV file
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """

    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Target column '{target_column}' statistics:")
        print(df[target_column].describe())

        # Handle missing values
        df = handle_missing_values(df)

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical variables
        X = encode_categorical_features(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to numpy arrays
        X_train_scaled = np.array(X_train_scaled, dtype=np.float32)
        X_test_scaled = np.array(X_test_scaled, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)
    
    except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise

def handle_missing_values(df):
    """ Handle missing values in the dataset."""

    # Check for missing values
    missing_counts = df.isnull().sum()

    if missing_counts.sum() > 0:
        print(f"Missing values found: {missing_counts[missing_counts > 0]}")
        
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def encode_categorical_features(X):
    """Encode categorical features using label encoding."""
    
    X_encoded = X.copy()
    
    # Identify categorical columns
    categorical_cols = X_encoded.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"Encoding categorical features: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    return X_encoded

def simulate_federated_data_split(X_train, y_train, num_clients=3):
    """
    Simulate federated data distribution across multiple clients.
    Each client gets a subset of the training data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        num_clients: Number of federated clients
    
    Returns:
        list: List of tuples (X_client, y_client) for each client
    """
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Split data among clients
    client_data = []
    data_per_client = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        if i == num_clients - 1:  # Last client gets remaining data
            end_idx = len(X_train)
        else:
            end_idx = (i + 1) * data_per_client
        
        X_client = X_shuffled[start_idx:end_idx]
        y_client = y_shuffled[start_idx:end_idx]
        
        client_data.append((X_client, y_client))
        print(f"Client {i}: {len(X_client)} samples")
    
    return client_data