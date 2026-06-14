"""
ASLA Framework with Real PJM Data
Handles multiple CSV files: AEP_HOURLY, COMED_hourly, pjm_hourly_est, etc.
"""

import pandas as pd
import numpy as np
import os
from asla_framework import *  # Import previous framework code

def load_pjm_multiple_files(data_folder='.', file_patterns=None):
    """
    Load PJM data from multiple CSV files
    
    Expected files:
    - AEP_HOURLY.csv
    - COMED_hourly.csv  
    - pjm_hourly_est.csv
    - DAYTON_hourly.csv (if exists)
    - etc.
    """
    if file_patterns is None:
        # Common PJM provider files
        file_patterns = [
            'AEP_HOURLY', 'COMED_hourly', 'pjm_hourly_est',
            'DAYTON_hourly', 'DOM_hourly', 'DUQ_hourly',
            'EKPC_hourly', 'FE_hourly', 'NI_hourly', 'PJM_Load_hourly'
        ]
    
    client_data = {}
    
    for pattern in file_patterns:
        # Try different file extensions
        for ext in ['.csv', '.CSV', '.txt']:
            filename = f"{pattern}{ext}"
            filepath = os.path.join(data_folder, filename)
            
            if os.path.exists(filepath):
                print(f"Loading: {filename}")
                df = pd.read_csv(filepath)
                
                # Check if first column is datetime
                first_col = df.columns[0]
                if 'date' in first_col.lower() or 'time' in first_col.lower() or 'datetime' in first_col.lower():
                    df[first_col] = pd.to_datetime(df[first_col])
                    # Extract the numeric column (usually second column or named 'load')
                    numeric_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    # If numeric column is still datetime, find the actual numeric column
                    for col in df.columns:
                        if 'load' in col.lower() or 'value' in col.lower() or 'mw' in col.lower():
                            numeric_col = col
                            break
                    
                    client_data[pattern] = df[numeric_col].values
                else:
                    # Assume all columns are numeric, use first column as load
                    client_data[pattern] = df.iloc[:, 0].values
                
                break
    
    if not client_data:
        # Fallback: try to load any CSV file in the folder
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        for csv_file in csv_files[:10]:  # Load up to 10 files
            print(f"Loading: {csv_file}")
            df = pd.read_csv(os.path.join(data_folder, csv_file))
            client_name = csv_file.replace('.csv', '').replace('_hourly', '')
            
            # Find numeric column
            numeric_col = None
            for col in df.columns:
                if 'load' in col.lower() or 'value' in col.lower() or 'mw' in col.lower():
                    numeric_col = col
                    break
            
            if numeric_col is None and len(df.columns) > 1:
                numeric_col = df.columns[1]  # Take second column
            
            if numeric_col:
                client_data[client_name] = df[numeric_col].values
            else:
                client_data[client_name] = df.iloc[:, 0].values
    
    # Convert to numpy array (all clients must have same length)
    # Find minimum length to align all clients
    min_length = min(len(data) for data in client_data.values())
    
    # Create aligned data matrix
    client_names = list(client_data.keys())
    aligned_data = np.column_stack([client_data[name][:min_length] for name in client_names])
    
    print(f"\n✅ Loaded {len(client_names)} clients")
    print(f"   Clients: {', '.join(client_names)}")
    print(f"   Each client has {min_length} hourly readings")
    print(f"   Data shape: {aligned_data.shape}")
    
    return aligned_data, client_names

def inspect_file_structure(data_folder='.'):
    """Inspect all CSV files in the folder to understand their structure"""
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    print("="*60)
    print("FILE STRUCTURE ANALYSIS")
    print("="*60)
    
    for csv_file in csv_files[:5]:  # Show first 5 files
        filepath = os.path.join(data_folder, csv_file)
        df = pd.read_csv(filepath, nrows=5)
        print(f"\n📁 {csv_file}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   First row:\n{df.iloc[0].to_dict()}")
        print("-"*40)

def load_and_preprocess_pjm(data_folder='.', lookback=24, test_size=0.3, max_clients=10):
    """
    Complete pipeline: load, inspect, and preprocess PJM data
    
    Args:
        data_folder: Folder containing the CSV files
        lookback: Hours to look back for features
        test_size: Proportion for testing
        max_clients: Maximum number of clients to use
    """
    
    # First inspect the files
    inspect_file_structure(data_folder)
    
    # Load the data
    client_data, client_names = load_pjm_multiple_files(data_folder)
    
    # Limit number of clients if needed
    if client_data.shape[1] > max_clients:
        client_data = client_data[:, :max_clients]
        client_names = client_names[:max_clients]
    
    # Test for heterogeneity (as in paper)
    print("\n" + "="*60)
    print("HETEROGENEITY ANALYSIS ON REAL PJM DATA")
    print("="*60)
    test_data_heterogeneity(client_data)
    
    # Prepare clients for federated learning
    clients = []
    
    for i, client_name in enumerate(client_names):
        print(f"\nPreprocessing client {i+1}: {client_name}")
        
        # Get this client's load data
        load_data = client_data[:, i]
        
        # Remove NaN values if any
        load_data = load_data[~np.isnan(load_data)]
        
        # Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_client_data(
            load_data, lookback=lookback, test_size=test_size
        )
        
        client = FederatedClient(
            client_id=i,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            model_fn=create_model_for_data1,  # Use ANN for PJM data
            scaler=scaler
        )
        clients.append(client)
        
        print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return clients, client_names, client_data

def run_pjm_experiment(data_folder='.'):
    """
    Run full ASLA experiment on PJM data
    """
    print("="*60)
    print("ASLA FRAMEWORK ON REAL PJM SMART GRID DATA")
    print("="*60)
    
    # Load data
    clients, client_names, client_data = load_and_preprocess_pjm(
        data_folder=data_folder,
        lookback=24,
        test_size=0.3,
        max_clients=10
    )
    
    # Visualize client data to show heterogeneity
    print("\n" + "="*60)
    print("VISUALIZING CLIENT DATA HETEROGENEITY")
    print("="*60)
    
    # Plot client data
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: First 500 hours for first 5 clients
    ax1 = axes[0, 0]
    for i in range(min(5, len(client_names))):
        ax1.plot(client_data[:500, i], label=client_names[i], alpha=0.7)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Load (MW)')
    ax1.set_title('First 500 Hours - Different Clients')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution boxplot
    ax2 = axes[0, 1]
    ax2.boxplot([client_data[:, i] for i in range(len(client_names))], 
                labels=client_names, rot=45)
    ax2.set_ylabel('Load (MW)')
    ax2.set_title('Load Distribution by Client')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weekly pattern for first client
    ax3 = axes[1, 0]
    first_client = client_data[:168, 0]  # First week
    ax3.plot(first_client)
    ax3.set_xlabel('Hour of Week')
    ax3.set_ylabel('Load (MW)')
    ax3.set_title(f'{client_names[0]} - Weekly Pattern')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation heatmap between clients
    ax4 = axes[1, 1]
    corr_matrix = np.corrcoef(client_data.T)
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(client_names)))
    ax4.set_yticks(range(len(client_names)))
    ax4.set_xticklabels(client_names, rotation=45, ha='right')
    ax4.set_yticklabels(client_names)
    ax4.set_title('Correlation Between Clients')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()
    
    # Run experiments with different aggregation layers
    print("\n" + "="*60)
    print("EXPERIMENT 1: Comparing Different Aggregation Layers")
    print("="*60)
    
    results = {}
    for layer_idx, layer_name in [(0, 'Layer 1 (Input)'), (1, 'Layer 2 (Hidden)'), (2, 'Layer 3 (Output)')]:
        print(f"\n--- Aggregating {layer_name} ---")
        
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
        
        history = server.train(fresh_clients, num_rounds=100, local_epochs=1, batch_size=300)
        results[layer_name] = history
        
        print(f"   ✅ Final MAPE: {history['mape'][-1]:.2f}%")
        print(f"   📊 Rounds completed: {len(history['mape'])}")
    
    # Compare layers
    compare_aggregation_layers(results)
    
    # Quantization experiment (as in paper Section 5.3)
    print("\n" + "="*60)
    print("EXPERIMENT 2: Quantization Effects (8-bit vs 16-bit vs 32-bit)")
    print("="*60)
    
    quantization_results = {}
    for bits, int_bits, name in [(32, 8, '32-bit Float'), (16, 5, '16-bit Fixed'), (8, 2, '8-bit Fixed')]:
        print(f"\n--- {name} System ---")
        
        server = ASLAFederatedServer(
            model_fn=create_model_for_data1,
            num_clients=len(clients),
            aggregation_layer=0,  # First layer
            quantization_bits=bits,
            int_bits=int_bits,
            stopping_threshold=0.3
        )
        
        fresh_clients = []
        for c in clients:
            fresh_clients.append(FederatedClient(
                c.client_id, c.X_train, c.y_train,
                c.X_test, c.y_test, create_model_for_data1, c.scaler
            ))
        
        history = server.train(fresh_clients, num_rounds=50, local_epochs=1)
        quantization_results[name] = history
        
        print(f"   ✅ Final MAPE: {history['mape'][-1]:.2f}%")
        print(f"   💾 Rounds: {len(history['mape'])}")
    
    # Plot quantization comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in quantization_results.items():
        ax.plot(history['mape'], label=name, linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Effect of Quantization on Model Performance (Real PJM Data)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate communication savings (as in paper Section 6.4)
    print("\n" + "="*60)
    print("COMMUNICATION COST ANALYSIS")
    print("="*60)
    
    # Layer sizes for 3-layer ANN
    # Input to Hidden1: 5×100 = 500
    # Hidden1 to Hidden2: 100×50 = 5000  
    # Hidden2 to Output: 50×1 = 50
    layer_sizes = [500, 5000, 50]
    
    for bits, name in [(32, '32-bit'), (16, '16-bit'), (8, '8-bit')]:
        cost = calculate_communication_cost(None, 100, layer_sizes, bits)
        print(f"\n{name} System:")
        print(f"   Communication cost: {cost['KB']:.2f} KB per client")
        if bits < 32:
            saving = (1 - bits/32) * 100
            print(f"   Memory saving: {saving:.0f}%")
    
    # Compare with paper's claims
    print("\n" + "="*60)
    print("COMPARISON WITH PAPER'S CLAIMS")
    print("="*60)
    print("Paper's results:")
    print("  • 8-bit quantization: 0.01% loss degradation for Data 1")
    print("  • Communication reduction: 829.2x for Data 1")
    print("  • Memory reduction: 75%")
    print("\nObserved results (will vary based on data):")
    print(f"  • Quantization MAPE change: {quantization_results['32-bit Float']['mape'][-1]:.2f}% → {quantization_results['8-bit Fixed']['mape'][-1]:.2f}%")
    
    return results, quantization_results

# Run the experiment
if __name__ == "__main__":
    # Set the path to your downloaded files
    data_folder = "."  # Current directory, or specify path like "./data/pjm"
    
    # Run the full experiment
    results, quantization_results = run_pjm_experiment(data_folder)