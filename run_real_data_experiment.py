# run_real_data_experiment.py
import pandas as pd
import numpy as np
from asla_framework import *

def run_experiment_with_real_data():
    """Run ASLA experiment using real PJM data"""
    
    print("="*60)
    print("ASLA with Real PJM Energy Data")
    print("="*60)
    
    # Load real data
    try:
        df = pd.read_csv('data/PJM_Load_hourly.csv')
        print(f"Loaded file with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")  # Show first 5 columns
    except FileNotFoundError:
        print("Error: Please check your file path. Looking for 'data/PJM_Load_hourly.csv'")
        return
    
    # Identify client columns (all except datetime)
    datetime_col = df.columns[0] if 'date' in df.columns[0].lower() else df.columns[0]
    client_cols = [c for c in df.columns if c != datetime_col][:10]  # Use 10 clients
    
    print(f"\nUsing {len(client_cols)} clients")
    
    # Extract and preprocess data
    client_data = df[client_cols].values.T  # Shape: (n_clients, n_samples)
    
    # Test heterogeneity (as in paper)
    print("\n" + "-"*40)
    test_data_heterogeneity(client_data.T)  # Transpose for correct shape
    
    # Prepare clients
    clients = []
    for i in range(len(client_cols)):
        client_load = client_data[i]  # Get this client's load data
        
        # Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_client_data(
            client_load, lookback=24, test_size=0.3
        )
        
        client = FederatedClient(
            client_id=i,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            model_fn=create_model_for_data1,
            scaler=scaler
        )
        clients.append(client)
    
    # Test different aggregation layers
    results = {}
    for layer_idx, layer_name in [(0, 'Layer 1'), (1, 'Layer 2'), (2, 'Layer 3')]:
        print(f"\n--- Testing {layer_name} aggregation ---")
        
        server = ASLAFederatedServer(
            model_fn=create_model_for_data1,
            num_clients=len(clients),
            aggregation_layer=layer_idx,
            quantization_bits=32,
            stopping_threshold=0.3
        )
        
        # Reset clients
        fresh_clients = []
        for c in clients:
            fresh_clients.append(FederatedClient(
                c.client_id, c.X_train, c.y_train,
                c.X_test, c.y_test, create_model_for_data1, c.scaler
            ))
        
        history = server.train(fresh_clients, num_rounds=100, local_epochs=1)
        results[layer_name] = history
        print(f"  Final MAPE: {history['mape'][-1]:.2f}%")
        print(f"  Rounds: {len(history['mape'])}")
    
    # Test quantization
    print("\n" + "="*60)
    print("Testing Quantization Effects on Real Data")
    print("="*60)
    
    for bits in [32, 16, 8]:
        print(f"\n--- {bits}-bit System ---")
        server = ASLAFederatedServer(
            model_fn=create_model_for_data1,
            num_clients=len(clients),
            aggregation_layer=0,  # First layer
            quantization_bits=bits,
            int_bits=8 if bits==32 else (5 if bits==16 else 2)
        )
        
        fresh_clients = []
        for c in clients:
            fresh_clients.append(FederatedClient(
                c.client_id, c.X_train, c.y_train,
                c.X_test, c.y_test, create_model_for_data1, c.scaler
            ))
        
        history = server.train(fresh_clients, num_rounds=50, local_epochs=1)
        print(f"  Final MAPE: {history['mape'][-1]:.2f}%")
    
    return results

if __name__ == "__main__":
    results = run_experiment_with_real_data()