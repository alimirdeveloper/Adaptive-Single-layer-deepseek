"""
ASLA Framework for Energy-efficient and Privacy-preserving Load Forecasting
Complete implementation with real PJM data support - FULLY FIXED
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# PART 1: DATA LOADING FOR PJM FILES
# ============================================================================

def inspect_file_structure(data_folder='.'):
    """Inspect all CSV files in the folder to understand their structure"""
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    print("="*60)
    print("FILE STRUCTURE ANALYSIS")
    print("="*60)
    
    for csv_file in csv_files[:5]:
        filepath = os.path.join(data_folder, csv_file)
        df = pd.read_csv(filepath, nrows=5)
        print(f"\n📁 {csv_file}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   First row:\n{df.iloc[0].to_dict()}")
        print("-"*40)

def load_pjm_multiple_files(data_folder='.', file_patterns=None):
    """
    Load PJM data from multiple CSV files
    """
    if file_patterns is None:
        file_patterns = [
            'AEP_hourly', 'COMED_hourly', 'pjm_hourly_est',
            'DAYTON_hourly', 'DOM_hourly', 'DUQ_hourly',
            'EKPC_hourly', 'FE_hourly', 'NI_hourly', 'PJM_Load_hourly'
        ]
    
    client_data = {}
    
    # First try specific patterns
    for pattern in file_patterns:
        for ext in ['.csv', '.CSV']:
            filename = f"{pattern}{ext}"
            filepath = os.path.join(data_folder, filename)
            
            if os.path.exists(filepath):
                print(f"Loading: {filename}")
                df = pd.read_csv(filepath)
                
                # Find numeric column
                numeric_col = None
                for col in df.columns:
                    if 'MW' in col or 'load' in col.lower() or 'value' in col.lower():
                        numeric_col = col
                        break
                
                if numeric_col is None and len(df.columns) > 1:
                    numeric_col = df.columns[1]
                else:
                    numeric_col = df.columns[0]
                
                values = pd.to_numeric(df[numeric_col], errors='coerce').dropna().values
                
                if len(values) > 0:
                    client_data[pattern] = values
                    print(f"   Loaded {len(values)} readings from column '{numeric_col}'")
                break
    
    # Try loading any CSV file if no data found
    if not client_data:
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        print(f"\nScanning {len(csv_files)} CSV files...")
        
        for csv_file in csv_files:
            filepath = os.path.join(data_folder, csv_file)
            df = pd.read_csv(filepath)
            client_name = csv_file.replace('.csv', '').replace('_hourly', '')
            
            for col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(values) > 1000:  # Only keep if substantial data
                    client_data[client_name] = values.values
                    print(f"  ✓ {csv_file} -> column '{col}': {len(values)} readings")
                    break
    
    if not client_data:
        raise ValueError("No valid numeric data found in any CSV file")
    
    # Align all clients to same length
    min_length = min(len(data) for data in client_data.values())
    print(f"\nAligning to minimum length: {min_length}")
    
    client_names = list(client_data.keys())
    aligned_data_list = [client_data[name][:min_length] for name in client_names]
    aligned_data = np.column_stack(aligned_data_list)
    
    print(f"\n✅ Loaded {len(client_names)} clients")
    print(f"   Data shape: {aligned_data.shape}")
    
    return aligned_data, client_names

# ============================================================================
# PART 2: FEATURE PREPROCESSING
# ============================================================================

def prepare_features(data, lookback=24):
    """Prepare features for load forecasting"""
    features = []
    targets = []
    
    for i in range(lookback, len(data) - 1):
        prev_hour = data[i-1]
        prev_day = data[i-24] if i >= 24 else data[0]
        prev_week = data[i-168] if i >= 168 else data[0]
        avg_24h = np.mean(data[i-24:i]) if i >= 24 else np.mean(data[:i])
        avg_week = np.mean(data[i-168:i]) if i >= 168 else np.mean(data[:i])
        
        features.append([prev_hour, prev_day, prev_week, avg_24h, avg_week])
        targets.append(data[i])
    
    return np.array(features), np.array(targets)

def preprocess_client_data(data, lookback=24, test_size=0.3):
    """Preprocess data for a single client"""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    data = data[~np.isinf(data)]
    
    if len(data) < lookback + 10:
        return None, None, None, None, None
    
    X, y = prepare_features(data, lookback)
    
    if len(X) < 10:
        return None, None, None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# ============================================================================
# PART 3: HETEROGENEITY ANALYSIS
# ============================================================================

def test_data_heterogeneity(client_data, client_names):
    """Analyze data heterogeneity across clients"""
    print("\n" + "="*60)
    print("DATA HETEROGENEITY ANALYSIS")
    print("="*60)
    
    print("\nClient Load Statistics:")
    print("-" * 70)
    print(f"{'Client':<15} {'Mean (MW)':<12} {'Std (MW)':<12} {'Min (MW)':<10} {'Max (MW)':<10}")
    print("-" * 70)
    
    std_values = []
    mean_values = []
    
    for i in range(min(len(client_names), client_data.shape[1])):
        data = client_data[:, i]
        data = data[~np.isnan(data)]
        
        if len(data) > 0:
            mean_val = np.mean(data)
            std_val = np.std(data)
            std_values.append(std_val)
            mean_values.append(mean_val)
            
            print(f"{client_names[i]:<15} {mean_val:>10.2f}  {std_val:>10.2f}  {np.min(data):>10.0f}  {np.max(data):>10.0f}")
    
    if len(std_values) >= 2:
        print("-" * 70)
        std_ratio = max(std_values) / min(std_values) if min(std_values) > 0 else float('inf')
        mean_ratio = max(mean_values) / min(mean_values) if min(mean_values) > 0 else float('inf')
        
        print(f"\n📊 Heterogeneity Metrics:")
        print(f"   Standard Deviation Ratio (max/min): {std_ratio:.2f}")
        print(f"   Mean Ratio (max/min): {mean_ratio:.2f}")
        
        if std_ratio > 3:
            print("\n   ✓ Significant heterogeneity detected!")
            print("   → ASLA framework is well-suited for this data")

# ============================================================================
# PART 4: MODEL DEFINITIONS
# ============================================================================

def create_model_for_data1(input_dim=5):
    """3-layer ANN"""
    model = Sequential([
        Dense(100, activation='relu', input_shape=(input_dim,), name='layer1'),
        Dense(50, activation='relu', name='layer2'),
        Dense(1, activation='linear', name='layer3')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mape'])
    return model

def get_layer_weights(model, layer_idx):
    return model.layers[layer_idx].get_weights()

def set_layer_weights(model, layer_idx, weights):
    model.layers[layer_idx].set_weights(weights)

# ============================================================================
# PART 5: FEDERATED LEARNING WITH ASLA
# ============================================================================

class FederatedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test, model_fn, scaler=None, name=""):
        self.client_id = client_id
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model_fn()
        self.scaler = scaler
        self.stopped = False
        self.last_update = None
        self.consecutive_no_improvement = 0
        self.best_loss = float('inf')
    
    def train_local(self, epochs=1, batch_size=300):
        if self.stopped or self.X_train is None or len(self.X_train) == 0:
            return None
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, 
                                batch_size=min(batch_size, len(self.X_train)), verbose=0)
        return history.history['loss'][0]
    
    def evaluate(self):
        if self.X_test is None or len(self.X_test) == 0:
            return [float('inf'), float('inf'), float('inf')]
        return self.model.evaluate(self.X_test, self.y_test, verbose=0)
    
    def get_layer_weights(self, layer_idx):
        return get_layer_weights(self.model, layer_idx)
    
    def set_layer_weights(self, layer_idx, weights):
        set_layer_weights(self.model, layer_idx, weights)
    
    def check_stopping_criteria(self, current_loss, patience=5, min_delta=0.001):
        if current_loss is None:
            return False
        if current_loss < self.best_loss - min_delta:
            self.best_loss = current_loss
            self.consecutive_no_improvement = 0
            return False
        else:
            self.consecutive_no_improvement += 1
            if self.consecutive_no_improvement >= patience:
                self.stopped = True
                return True
        return False


class ASLAFederatedServer:
    def __init__(self, model_fn, num_clients, aggregation_layer=0, stopping_threshold=0.3):
        self.global_model = model_fn()
        self.num_clients = num_clients
        self.aggregation_layer = aggregation_layer
        self.stopping_threshold = stopping_threshold
        self.round = 0
        self.history = {'loss': [], 'mape': []}
    
    def aggregate_layer(self, client_weights_list):
        if not client_weights_list:
            return None
        avg_weights = []
        for i in range(len(client_weights_list[0])):
            layer_weights = [w[i] for w in client_weights_list]
            avg_weights.append(np.mean(layer_weights, axis=0))
        return avg_weights
    
    def federated_round(self, clients, local_epochs=1, batch_size=300):
        self.round += 1
        global_layer_weights = get_layer_weights(self.global_model, self.aggregation_layer)
        
        client_updates = []
        for client in clients:
            if client.stopped:
                if client.last_update is not None:
                    client_updates.append(client.last_update)
                continue
            
            client.set_layer_weights(self.aggregation_layer, global_layer_weights)
            loss = client.train_local(epochs=local_epochs, batch_size=batch_size)
            
            if loss is not None:
                layer_weights = client.get_layer_weights(self.aggregation_layer)
                client_updates.append(layer_weights)
                client.last_update = layer_weights
                client.check_stopping_criteria(loss)
        
        if client_updates:
            aggregated_weights = self.aggregate_layer(client_updates)
            set_layer_weights(self.global_model, self.aggregation_layer, aggregated_weights)
        
        stopped_clients = sum(1 for c in clients if c.stopped)
        if stopped_clients >= self.stopping_threshold * len(clients):
            return False
        
        self.evaluate_global(clients)
        return True
    
    def evaluate_global(self, clients):
        losses, mape_values = [], []
        global_layer = get_layer_weights(self.global_model, self.aggregation_layer)
        
        for client in clients:
            client.set_layer_weights(self.aggregation_layer, global_layer)
            eval_results = client.evaluate()
            if len(eval_results) >= 3 and not np.isinf(eval_results[2]):
                losses.append(eval_results[0])
                mape_values.append(eval_results[2])
        
        if losses:
            self.history['loss'].append(np.mean(losses))
            self.history['mape'].append(np.mean(mape_values))
            print(f"  Round {self.round}: MAPE = {np.mean(mape_values):.2f}%")
    
    def train(self, clients, num_rounds=30, local_epochs=1, batch_size=300):
        for _ in range(num_rounds):
            if not self.federated_round(clients, local_epochs, batch_size):
                break
        return self.history

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_client_data(client_data, client_names):
    """Plot client data to visualize heterogeneity"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: First 500 hours
    ax1 = axes[0, 0]
    for i in range(min(5, len(client_names))):
        ax1.plot(client_data[:500, i], label=client_names[i], alpha=0.7)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Load (MW)')
    ax1.set_title('First 500 Hours - Different Clients')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution boxplot
    ax2 = axes[0, 1]
    data_for_box = [client_data[:, i][:2000] for i in range(min(8, len(client_names)))]
    ax2.boxplot(data_for_box, tick_labels=client_names[:len(data_for_box)])
    ax2.set_ylabel('Load (MW)')
    ax2.set_title('Load Distribution by Client')
    ax2.set_xticklabels(client_names[:len(data_for_box)], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weekly pattern
    ax3 = axes[1, 0]
    ax3.plot(client_data[:168, 0])
    ax3.set_xlabel('Hour of Week')
    ax3.set_ylabel('Load (MW)')
    ax3.set_title(f'{client_names[0]} - Weekly Pattern')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram
    ax4 = axes[1, 1]
    for i in range(min(4, len(client_names))):
        ax4.hist(client_data[:, i][:5000], bins=50, alpha=0.5, label=client_names[i])
    ax4.set_xlabel('Load (MW)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Load Distribution Histogram')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_aggregation_layers(results_dict):
    """Compare results from different aggregation layers"""
    layers = list(results_dict.keys())
    mape_values = [results_dict[l]['mape'][-1] if results_dict[l]['mape'] else 100 for l in layers]
    rounds = [len(results_dict[l]['mape']) if results_dict[l]['mape'] else 0 for l in layers]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(layers, mape_values, color='skyblue')
    axes[0].set_xlabel('Aggregation Layer')
    axes[0].set_ylabel('Final MAPE (%)')
    axes[0].set_title('Performance by Layer')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(layers, rounds, color='lightcoral')
    axes[1].set_xlabel('Aggregation Layer')
    axes[1].set_ylabel('Communication Rounds')
    axes[1].set_title('Convergence Speed')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('ASLA Framework: Layer-wise Comparison')
    plt.tight_layout()
    plt.show()

# ============================================================================
# PART 7: MAIN EXPERIMENT
# ============================================================================

def run_pjm_experiment(data_folder='.'):
    print("="*60)
    print("ASLA FRAMEWORK ON REAL PJM SMART GRID DATA")
    print("="*60)
    
    inspect_file_structure(data_folder)
    
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    try:
        client_data, client_names = load_pjm_multiple_files(data_folder)
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
    test_data_heterogeneity(client_data, client_names)
    
    print("\n" + "="*60)
    print("VISUALIZING DATA")
    print("="*60)
    plot_client_data(client_data, client_names)
    
    print("\n" + "="*60)
    print("PREPROCESSING CLIENTS")
    print("="*60)
    
    clients = []
    for i, name in enumerate(client_names):
        X_train, X_test, y_train, y_test, scaler = preprocess_client_data(client_data[:, i])
        if X_train is not None and len(X_train) > 100:
            clients.append(FederatedClient(i, X_train, y_train, X_test, y_test, 
                                          create_model_for_data1, scaler, name))
            print(f"  ✓ {name}: {len(X_train)} samples")
    
    if len(clients) < 2:
        print("Error: Not enough valid clients!")
        return None, None
    
    print(f"\n✅ Using {len(clients)} clients")
    
    print("\n" + "="*60)
    print("EXPERIMENT: Comparing Aggregation Layers")
    print("="*60)
    
    results = {}
    for layer_idx, layer_name in [(0, 'Layer 1'), (1, 'Layer 2'), (2, 'Layer 3')]:
        print(f"\n--- Aggregating {layer_name} ---")
        server = ASLAFederatedServer(create_model_for_data1, len(clients), layer_idx)
        
        fresh_clients = [FederatedClient(c.client_id, c.X_train, c.y_train,
                       c.X_test, c.y_test, create_model_for_data1, c.scaler, c.name) for c in clients]
        
        history = server.train(fresh_clients, num_rounds=20, local_epochs=1)
        results[layer_name] = history
        if history['mape']:
            print(f"   Final MAPE: {history['mape'][-1]:.2f}%")
    
    compare_aggregation_layers(results)
    
    print("\n" + "="*60)
    print("EXPECTED RESULTS (from paper):")
    print("="*60)
    print("• Communication cost reduction: 829.2x for Data 1")
    print("• Memory reduction: 75% (32-bit → 8-bit)")
    
    return results, clients

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results, clients = run_pjm_experiment(".")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)