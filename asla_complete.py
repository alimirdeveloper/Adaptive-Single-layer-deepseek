"""
ASLA Framework for Energy-efficient and Privacy-preserving Load Forecasting
Complete implementation with: Layer Comparison + Quantization + FedAvg Comparison
FIXED VERSION
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# PART 1: DATA LOADING FUNCTIONS
# ============================================================================

def inspect_file_structure(data_folder='.'):
    """Inspect all CSV files in the folder"""
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print("="*60)
    print("FILE STRUCTURE ANALYSIS")
    print("="*60)
    for csv_file in csv_files[:5]:
        filepath = os.path.join(data_folder, csv_file)
        df = pd.read_csv(filepath, nrows=5)
        print(f"\n📁 {csv_file}")
        print(f"   Columns: {df.columns.tolist()}")
        print("-"*40)

def load_pjm_multiple_files(data_folder='.'):
    """Load PJM data from multiple CSV files"""
    client_data = {}
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    print(f"\nScanning {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        filepath = os.path.join(data_folder, csv_file)
        df = pd.read_csv(filepath)
        client_name = csv_file.replace('.csv', '').replace('_hourly', '')
        
        for col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 1000:
                client_data[client_name] = values.values
                print(f"  ✓ {csv_file} -> {len(values)} readings")
                break
    
    if not client_data:
        raise ValueError("No valid numeric data found")
    
    # Align all clients to same length
    min_length = min(len(data) for data in client_data.values())
    client_names = list(client_data.keys())
    aligned_data = np.column_stack([client_data[name][:min_length] for name in client_names])
    
    print(f"\n✅ Loaded {len(client_names)} clients, Data shape: {aligned_data.shape}")
    return aligned_data, client_names

# ============================================================================
# PART 2: FEATURE PREPROCESSING
# ============================================================================

def prepare_features(data, lookback=24):
    """Prepare features for load forecasting"""
    features, targets = [], []
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
    print("-"*70)
    print(f"{'Client':<15} {'Mean (MW)':<12} {'Std (MW)':<12} {'Min (MW)':<10} {'Max (MW)':<10}")
    print("-"*70)
    
    std_values = []
    for i in range(min(len(client_names), client_data.shape[1])):
        data = client_data[:, i][~np.isnan(client_data[:, i])]
        if len(data) > 0:
            std_values.append(np.std(data))
            print(f"{client_names[i]:<15} {np.mean(data):>10.2f}  {np.std(data):>10.2f}  {np.min(data):>10.0f}  {np.max(data):>10.0f}")
    
    if len(std_values) >= 2:
        std_ratio = max(std_values) / min(std_values)
        print("-"*70)
        print(f"\n📊 Heterogeneity: Std Ratio = {std_ratio:.2f}")
        if std_ratio > 3:
            print("   ✓ Significant heterogeneity detected - ASLA framework well-suited!")

# ============================================================================
# PART 4: MODEL DEFINITIONS
# ============================================================================

def create_model(input_dim=5):
    """3-layer ANN as in paper"""
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

def calculate_model_size(model, bits=32):
    """Calculate model size in KB"""
    total_weights = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            total_weights += weights[0].size
            if len(weights) > 1:
                total_weights += weights[1].size
    return total_weights * (bits // 8) / 1024

# ============================================================================
# PART 5: QUANTIZATION
# ============================================================================

def quantize_weights(weights, bits=32):
    """Quantize weights to fixed-point"""
    if bits == 32:
        return weights
    
    if bits == 16:
        int_bits = 5
    else:  # 8-bit
        int_bits = 2
    
    frac_bits = bits - int_bits - 1
    scale = 2 ** frac_bits
    max_val = (2 ** int_bits) - (1 / scale)
    min_val = -(2 ** int_bits)
    
    def quantize(arr):
        return np.round(np.clip(arr, min_val, max_val) * scale) / scale
    
    if isinstance(weights, list):
        return [quantize(w) for w in weights]
    return quantize(weights)

# ============================================================================
# PART 6: FEDERATED CLIENT
# ============================================================================

class FederatedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test, name="", quant_bits=32):
        self.client_id = client_id
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = create_model()
        self.quant_bits = quant_bits
        self.stopped = False
        self.last_update = None
        self.best_loss = float('inf')
        self.no_improve_count = 0
    
    def train_local(self, epochs=1, batch_size=300):
        if self.stopped or self.X_train is None:
            return None
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, 
                                batch_size=min(batch_size, len(self.X_train)), verbose=0)
        return history.history['loss'][0]
    
    def evaluate(self):
        if self.X_test is None:
            return [float('inf')] * 3
        return self.model.evaluate(self.X_test, self.y_test, verbose=0)
    
    def get_layer_weights(self, layer_idx):
        w = get_layer_weights(self.model, layer_idx)
        return quantize_weights(w, self.quant_bits) if self.quant_bits < 32 else w
    
    def set_layer_weights(self, layer_idx, weights):
        set_layer_weights(self.model, layer_idx, weights)
    
    def check_stopping(self, loss, patience=5):
        if loss is None:
            return False
        if loss < self.best_loss - 0.001:
            self.best_loss = loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= patience:
                self.stopped = True
        return self.stopped

# ============================================================================
# PART 7: ASLA SERVER (Single Layer Aggregation)
# ============================================================================

class ASLAServer:
    def __init__(self, num_clients, agg_layer=1, quant_bits=32):
        self.global_model = create_model()
        self.num_clients = num_clients
        self.agg_layer = agg_layer
        self.quant_bits = quant_bits
        self.comm_round = 0  # Changed from 'round' to 'comm_round'
        self.history = {'mape': []}
    
    def aggregate(self, client_weights):
        if not client_weights:
            return None
        avg_weights = []
        for i in range(len(client_weights[0])):
            avg_weights.append(np.mean([w[i] for w in client_weights], axis=0))
        return avg_weights
    
    def do_round(self, clients, local_epochs=1, batch_size=300):  # Renamed method
        self.comm_round += 1
        global_weights = get_layer_weights(self.global_model, self.agg_layer)
        
        updates = []
        for client in clients:
            if client.stopped:
                if client.last_update:
                    updates.append(client.last_update)
                continue
            
            client.set_layer_weights(self.agg_layer, global_weights)
            loss = client.train_local(local_epochs, batch_size)
            
            if loss is not None:
                updates.append(client.get_layer_weights(self.agg_layer))
                client.last_update = updates[-1]
                client.check_stopping(loss)
        
        if updates:
            agg_weights = self.aggregate(updates)
            set_layer_weights(self.global_model, self.agg_layer, agg_weights)
        
        # Evaluate
        mape_values = []
        for client in clients:
            client.set_layer_weights(self.agg_layer, 
                get_layer_weights(self.global_model, self.agg_layer))
            res = client.evaluate()
            if len(res) >= 3 and not np.isinf(res[2]):
                mape_values.append(res[2])
        
        if mape_values:
            avg_mape = np.mean(mape_values)
            self.history['mape'].append(avg_mape)
            print(f"  Round {self.comm_round}: MAPE = {avg_mape:.2f}%")
        
        stopped = sum(1 for c in clients if c.stopped)
        return stopped < 0.3 * len(clients)
    
    def train(self, clients, num_rounds=20):
        for _ in range(num_rounds):
            if not self.do_round(clients):
                break
        return self.history

# ============================================================================
# PART 8: FEDAVG SERVER (All Layers Aggregation)
# ============================================================================

class FedAvgServer:
    def __init__(self, num_clients, quant_bits=32):
        self.global_model = create_model()
        self.num_clients = num_clients
        self.quant_bits = quant_bits
        self.comm_round = 0  # Changed from 'round' to 'comm_round'
        self.history = {'mape': []}
    
    def aggregate_all(self, client_models):
        if not client_models:
            return None
        num_layers = len(client_models[0])
        aggregated = []
        for layer_idx in range(num_layers):
            layer_weights = [model[layer_idx] for model in client_models]
            avg_layer = []
            for i in range(len(layer_weights[0])):
                avg_layer.append(np.mean([w[i] for w in layer_weights], axis=0))
            aggregated.append(avg_layer)
        return aggregated
    
    def do_round(self, clients, local_epochs=1, batch_size=300):  # Renamed method
        self.comm_round += 1
        
        # Save global weights
        global_weights = [get_layer_weights(self.global_model, i) 
                         for i in range(len(self.global_model.layers))]
        
        updates = []
        for client in clients:
            if client.stopped:
                continue
            
            # Set all global weights
            for i, w in enumerate(global_weights):
                client.set_layer_weights(i, w)
            
            loss = client.train_local(local_epochs, batch_size)
            
            if loss is not None:
                # Get all layer weights
                all_weights = [client.get_layer_weights(i) 
                             for i in range(len(client.model.layers))]
                updates.append(all_weights)
                client.check_stopping(loss)
        
        if updates:
            agg_weights = self.aggregate_all(updates)
            for i, w in enumerate(agg_weights):
                set_layer_weights(self.global_model, i, w)
        
        # Evaluate
        mape_values = []
        for client in clients:
            for i, w in enumerate(global_weights):
                client.set_layer_weights(i, w)
            res = client.evaluate()
            if len(res) >= 3 and not np.isinf(res[2]):
                mape_values.append(res[2])
        
        if mape_values:
            avg_mape = np.mean(mape_values)
            self.history['mape'].append(avg_mape)
            print(f"  Round {self.comm_round}: MAPE = {avg_mape:.2f}%")
        
        stopped = sum(1 for c in clients if c.stopped)
        return stopped < 0.3 * len(clients)
    
    def train(self, clients, num_rounds=20):
        for _ in range(num_rounds):
            if not self.do_round(clients):
                break
        return self.history

# ============================================================================
# PART 9: VISUALIZATION
# ============================================================================

def plot_comparison(fedavg_history, asla_history):
    """Plot ASLA vs FedAvg comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy over rounds
    ax1 = axes[0]
    ax1.plot(range(1, len(fedavg_history['mape'])+1), fedavg_history['mape'], 
             'o-', label='FedAvg (All Layers)', color='blue', linewidth=2)
    ax1.plot(range(1, len(asla_history['mape'])+1), asla_history['mape'], 
             's-', label='ASLA (Single Layer)', color='red', linewidth=2)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('MAPE (%)')
    ax1.set_title('ASLA vs FedAvg: Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Final comparison
    ax2 = axes[1]
    methods = ['FedAvg', 'ASLA']
    final_mape = [fedavg_history['mape'][-1], asla_history['mape'][-1]]
    bars = ax2.bar(methods, final_mape, color=['blue', 'red'], edgecolor='black')
    ax2.set_ylabel('Final MAPE (%)')
    ax2.set_title('Final Accuracy Comparison')
    for bar, val in zip(bars, final_mape):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, 
                f'{val:.2f}%', ha='center', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('asla_vs_fedavg.png', dpi=150)
    plt.show()

def plot_quantization(quant_results):
    """Plot quantization comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'32-bit': 'blue', '16-bit': 'green', '8-bit': 'red'}
    markers = {'32-bit': 'o', '16-bit': 's', '8-bit': '^'}
    
    for name, history in quant_results.items():
        rounds = range(1, len(history['mape'])+1)
        ax.plot(rounds, history['mape'], marker=markers[name], label=name, 
                color=colors[name], linewidth=2, markersize=4)
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Quantization Comparison (32-bit vs 16-bit vs 8-bit)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quantization_comparison.png', dpi=150)
    plt.show()

# ============================================================================
# PART 10: MAIN EXPERIMENT
# ============================================================================

def run_experiment(data_folder='.'):
    """Run complete experiment"""
    print("="*60)
    print("ASLA FRAMEWORK - COMPLETE EXPERIMENT")
    print("Layer Comparison + Quantization + FedAvg Comparison")
    print("="*60)
    
    # Load data
    inspect_file_structure(data_folder)
    client_data, client_names = load_pjm_multiple_files(data_folder)
    test_data_heterogeneity(client_data, client_names)
    
    # Preprocess clients
    print("\n" + "="*60)
    print("PREPROCESSING CLIENTS")
    print("="*60)
    
    clients = []
    for i, name in enumerate(client_names[:10]):  # Use first 10 clients
        X_train, X_test, y_train, y_test, _ = preprocess_client_data(client_data[:, i])
        if X_train is not None and len(X_train) > 100:
            clients.append(FederatedClient(i, X_train, y_train, X_test, y_test, name))
            print(f"  ✓ {name}: {len(X_train)} samples")
    
    print(f"\n✅ Using {len(clients)} clients")
    
    # Experiment 1: Find best aggregation layer
    print("\n" + "="*60)
    print("EXPERIMENT 1: Finding Best Aggregation Layer")
    print("="*60)
    
    layer_results = {}
    for layer_idx, layer_name in [(0, 'Layer 1'), (1, 'Layer 2'), (2, 'Layer 3')]:
        print(f"\n--- {layer_name} ---")
        server = ASLAServer(len(clients), agg_layer=layer_idx)
        fresh = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                for c in clients]
        history = server.train(fresh, num_rounds=20)
        layer_results[layer_name] = history['mape'][-1] if history['mape'] else 100
        print(f"   Final MAPE: {layer_results[layer_name]:.2f}%")
    
    best_layer = min(layer_results, key=layer_results.get)
    print(f"\n🏆 Best layer: {best_layer} with MAPE = {layer_results[best_layer]:.2f}%")
    
    # Experiment 2: Quantization
    print("\n" + "="*60)
    print("EXPERIMENT 2: Quantization Effects")
    print("="*60)
    
    quant_results = {}
    for bits, name in [(32, '32-bit'), (16, '16-bit'), (8, '8-bit')]:
        print(f"\n--- {name} System ---")
        server = ASLAServer(len(clients), agg_layer=1, quant_bits=bits)
        fresh = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name, bits) 
                for c in clients]
        history = server.train(fresh, num_rounds=20)
        quant_results[name] = history
        print(f"   Final MAPE: {history['mape'][-1]:.2f}%")
    
    plot_quantization(quant_results)
    
    # Experiment 3: ASLA vs FedAvg
    print("\n" + "="*60)
    print("EXPERIMENT 3: ASLA vs FedAvg Comparison")
    print("="*60)
    
    # FedAvg
    print("\n--- FedAvg (All Layers) ---")
    fedavg_server = FedAvgServer(len(clients))
    fedavg_clients = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                     for c in clients]
    fedavg_history = fedavg_server.train(fedavg_clients, num_rounds=20)
    
    # ASLA (best layer)
    print("\n--- ASLA (Single Layer - Layer 2) ---")
    asla_server = ASLAServer(len(clients), agg_layer=1)
    asla_clients = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                   for c in clients]
    asla_history = asla_server.train(asla_clients, num_rounds=20)
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    model = create_model()
    model_size = calculate_model_size(model, 32)
    layer_size = (5000 + 50) * 4 / 1024  # Layer 2: 5000 weights + 50 biases
    
    print(f"\n📊 Communication per round (32-bit):")
    print(f"   FedAvg (full model): {model_size:.2f} KB")
    print(f"   ASLA (single layer): {layer_size:.2f} KB")
    print(f"   🚀 ASLA saves: {model_size/layer_size:.1f}x per round!")
    
    print(f"\n📊 Total Communication (20 rounds, {len(clients)} clients):")
    fedavg_total = model_size * 20 * len(clients) * 2
    asla_total = layer_size * 20 * len(clients) * 2
    print(f"   FedAvg: {fedavg_total:.2f} KB")
    print(f"   ASLA:   {asla_total:.2f} KB")
    print(f"   🚀 ASLA total saving: {fedavg_total/asla_total:.1f}x")
    
    print(f"\n📊 Accuracy (Final MAPE):")
    print(f"   FedAvg: {fedavg_history['mape'][-1]:.2f}%")
    print(f"   ASLA:   {asla_history['mape'][-1]:.2f}%")
    
    if asla_history['mape'][-1] < fedavg_history['mape'][-1]:
        improvement = fedavg_history['mape'][-1] - asla_history['mape'][-1]
        print(f"   🎯 ASLA is {improvement:.2f}% MORE accurate!")
    else:
        diff = asla_history['mape'][-1] - fedavg_history['mape'][-1]
        print(f"   📊 FedAvg is {diff:.2f}% more accurate")
    
    # Plot comparison
    plot_comparison(fedavg_history, asla_history)
    
    # Paper comparison
    print("\n" + "="*60)
    print("PAPER COMPARISON (Section 7)")
    print("="*60)
    print("\nPaper's claims:")
    print("  • FedKD: 19x communication improvement")
    print("  • FedProto: 161.25x improvement")
    print("  • ASLA (paper): 829.2x improvement for Data 1")
    print(f"\nYour results with PJM data:")
    print(f"  • ASLA communication saving: {fedavg_total/asla_total:.1f}x")
    print(f"  • Memory saving (32-bit→8-bit): 75%")
    
    return layer_results, quant_results, fedavg_history, asla_history


if __name__ == "__main__":
    results = run_experiment(".")
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("Visualizations saved: asla_vs_fedavg.png, quantization_comparison.png")
    print("="*60)