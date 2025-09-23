#!/usr/bin/env python3
"""
Test script to verify your federated learning setup works.
Run this before starting the actual federated learning process.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append('src')

from src.model import create_mlp_model
from src.data_preprocessing import load_and_preprocess_data, simulate_federated_data_split

def test_data_loading(csv_path):
    """Test data loading and preprocessing."""
    print("=" * 50)
    print("TESTING DATA LOADING")
    print("=" * 50)
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check if target column exists
        if 'LengthofCycle' in df.columns:
            print(f"✓ Target column 'LengthofCycle' found")
            print(f"  Target statistics: {df['LengthofCycle'].describe()}")
        else:
            print(f"✗ Target column 'LengthofCycle' not found!")
            print(f"  Available columns: {list(df.columns)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return False

def test_preprocessing(csv_path):
    """Test data preprocessing."""
    print("\n" + "=" * 50)
    print("TESTING DATA PREPROCESSING")
    print("=" * 50)
    
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(csv_path)
        
        print(f"✓ Data preprocessing successful")
        print(f"  Training features shape: {X_train.shape}")
        print(f"  Test features shape: {X_test.shape}")
        print(f"  Training targets shape: {y_train.shape}")
        print(f"  Test targets shape: {y_test.shape}")
        print(f"  Number of features: {len(feature_names)}")
        
        return X_train, X_test, y_train, y_test, scaler, feature_names
        
    except Exception as e:
        print(f"✗ Error in preprocessing: {str(e)}")
        return None

def test_model_creation(input_dim):
    """Test model creation."""
    print("\n" + "=" * 50)
    print("TESTING MODEL CREATION")
    print("=" * 50)
    
    try:
        model = create_mlp_model(input_dim)
        print(f"✓ Model created successfully")
        print(f"  Input dimension: {input_dim}")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"✗ Error creating model: {str(e)}")
        return None

def test_federated_split(X_train, y_train):
    """Test federated data splitting."""
    print("\n" + "=" * 50)
    print("TESTING FEDERATED DATA SPLIT")
    print("=" * 50)
    
    try:
        client_data = simulate_federated_data_split(X_train, y_train, num_clients=3)
        
        print(f"✓ Federated split successful")
        print(f"  Number of clients: {len(client_data)}")
        
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"  Client {i}: {len(X_client)} samples")
        
        return client_data
        
    except Exception as e:
        print(f"✗ Error in federated split: {str(e)}")
        return None

def test_model_training(model, X_sample, y_sample):
    """Test model training with sample data."""
    print("\n" + "=" * 50)
    print("TESTING MODEL TRAINING")
    print("=" * 50)
    
    try:
        # Take a small sample for quick testing
        sample_size = min(100, len(X_sample))
        X_sample = X_sample[:sample_size]
        y_sample = y_sample[:sample_size]
        
        # Train for 1 epoch
        history = model.fit(X_sample, y_sample, epochs=1, verbose=1)
        
        print(f"✓ Model training successful")
        print(f"  Training loss: {history.history['loss'][0]:.4f}")
        
        # Test prediction
        predictions = model.predict(X_sample[:5])
        print(f"  Sample predictions: {predictions.flatten()}")
        print(f"  Sample actuals: {y_sample[:5]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model training: {str(e)}")
        return False

def main():
    """Run all tests."""
    # Get CSV path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python test_setup.py <path_to_your_csv_file>")
        print("Example: python test_setup.py data/menstrual_data.csv")
        return
    
    csv_path = sys.argv[1]
    
    print("FEDERATED LEARNING SETUP TEST")
    print("=" * 50)
    print(f"Testing with data file: {csv_path}")
    
    # Test data loading
    if not test_data_loading(csv_path):
        print("\n❌ SETUP TEST FAILED - Data loading issue")
        return
    
    # Test preprocessing
    result = test_preprocessing(csv_path)
    if result is None:
        print("\n❌ SETUP TEST FAILED - Preprocessing issue")
        return
    
    X_train, X_test, y_train, y_test, scaler, feature_names = result
    
    # Test model creation
    model = test_model_creation(X_train.shape[1])
    if model is None:
        print("\n❌ SETUP TEST FAILED - Model creation issue")
        return
    
    # Test federated split
    client_data = test_federated_split(X_train, y_train)
    if client_data is None:
        print("\n❌ SETUP TEST FAILED - Federated split issue")
        return
    
    # Test model training
    if not test_model_training(model, X_train, y_train):
        print("\n❌ SETUP TEST FAILED - Model training issue")
        return
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("Your federated learning setup is ready!")
    print("\nNext steps:")
    print("1. Start the server: python src/server.py --data-path your_data.csv")
    print("2. Start clients: python src/client.py --client-id 0 --data-path your_data.csv")
    print("3. Start more clients with different IDs (1, 2, etc.)")

if __name__ == "__main__":
    main()