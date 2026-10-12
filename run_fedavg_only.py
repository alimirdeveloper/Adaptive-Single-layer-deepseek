"""
اجرای فقط FedAvg - با Early Stopping غیرفعال
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

np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# توابع کمکی
# ============================================================================

def load_pjm_multiple_files(data_folder='.'):
    client_data = {}
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        filepath = os.path.join(data_folder, csv_file)
        df = pd.read_csv(filepath)
        client_name = csv_file.replace('.csv', '').replace('_hourly', '')
        
        for col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 1000:
                client_data[client_name] = values.values
                break
    
    if not client_data:
        raise ValueError("No valid numeric data found")
    
    target_samples = 13896
    min_length = min(len(data) for data in client_data.values())
    
    if min_length > target_samples:
        print(f"  ⚠️ کاهش نمونه‌ها از {min_length} به {target_samples} (مطابق مقاله)")
        min_length = target_samples
    
    client_names = list(client_data.keys())
    aligned_data = np.column_stack([client_data[name][:min_length] for name in client_names])
    
    print(f"✅ Loaded {len(client_names)} clients, Data shape: {aligned_data.shape}")
    return aligned_data, client_names

def prepare_features(data, lookback=24):
    features, targets = [], []
    for i in range(lookback, len(data)):
        prev_hour = data[i-1]
        prev_day = data[i-24] if i >= 24 else data[0]
        prev_week = data[i-168] if i >= 168 else data[0]
        avg_24h = np.mean(data[i-24:i]) if i >= 24 else np.mean(data[:i])
        avg_week = np.mean(data[i-168:i]) if i >= 168 else np.mean(data[:i])
        features.append([prev_hour, prev_day, prev_week, avg_24h, avg_week])
        targets.append(data[i])
    return np.array(features), np.array(targets)

def preprocess_client_data(data, lookback=24, test_size=0.3):
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

def create_model(input_dim=5):
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
# کلاس FederatedClient با Early Stopping غیرفعال
# ============================================================================

class FederatedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test, name=""):
        self.client_id = client_id
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = create_model()
        self.stopped = False  # همیشه False می‌ماند
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
        return get_layer_weights(self.model, layer_idx)
    
    def set_layer_weights(self, layer_idx, weights):
        set_layer_weights(self.model, layer_idx, weights)
    
    def check_stopping(self, loss):
        # ✅ Early Stopping کاملاً غیرفعال
        return False  # هیچ‌وقت متوقف نشو

# ============================================================================
# کلاس FedAvgServer با Early Stopping غیرفعال
# ============================================================================

class FedAvgServer:
    def __init__(self, num_clients):
        self.global_model = create_model()
        self.num_clients = num_clients
        self.comm_round = 0
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
    
    def do_round(self, clients, local_epochs=1, batch_size=300):
        self.comm_round += 1
        
        global_weights = [get_layer_weights(self.global_model, i) 
                         for i in range(len(self.global_model.layers))]
        
        updates = []
        for client in clients:
            for i, w in enumerate(global_weights):
                client.set_layer_weights(i, w)
            
            loss = client.train_local(local_epochs, batch_size)
            
            if loss is not None:
                all_weights = [client.get_layer_weights(i) 
                             for i in range(len(client.model.layers))]
                updates.append(all_weights)
                client.check_stopping(loss)  # همیشه False برمی‌گرداند
        
        if updates:
            agg_weights = self.aggregate_all(updates)
            for i, w in enumerate(agg_weights):
                set_layer_weights(self.global_model, i, w)
        
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
        
        # ✅ Early Stopping در سرور هم غیرفعال
        return True
    
    def train(self, clients, num_rounds=100):
        print(f"\n🚀 Starting FedAvg Training for {num_rounds} rounds...")
        print("❌ Early Stopping: غیرفعال")
        print("-" * 50)
        for _ in range(num_rounds):
            if not self.do_round(clients):
                break
        print("-" * 50)
        if self.history['mape']:
            print(f"✅ FedAvg completed! Final MAPE: {self.history['mape'][-1]:.2f}%")
        return self.history

# ============================================================================
# تابع اصلی
# ============================================================================

def run_fedavg_only(data_folder='.'):
    print("="*60)
    print("🚀 FEDAVG ONLY - 100 ROUNDS")
    print("❌ Early Stopping: کاملاً غیرفعال")
    print("="*60)
    
    client_data, client_names = load_pjm_multiple_files(data_folder)
    
    print("\n" + "="*60)
    print("PREPROCESSING CLIENTS")
    print("="*60)
    
    clients = []
    for i, name in enumerate(client_names[:10]):
        X_train, X_test, y_train, y_test, _ = preprocess_client_data(client_data[:, i])
        if X_train is not None and len(X_train) > 100:
            clients.append(FederatedClient(i, X_train, y_train, X_test, y_test, name))
            print(f"  ✓ {name}: {len(X_train)} samples")
    
    print(f"\n✅ Using {len(clients)} clients")
    
    fedavg_server = FedAvgServer(len(clients))
    fedavg_clients = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                     for c in clients]
    history = fedavg_server.train(fedavg_clients, num_rounds=100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history['mape'])+1), history['mape'], 'b-', linewidth=2)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('MAPE (%)', fontsize=12)
    plt.title('FedAvg Performance (100 Rounds - No Early Stopping)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fedavg_no_early_stop.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"  Final MAPE: {history['mape'][-1]:.2f}%")
    print(f"  Rounds completed: {len(history['mape'])}")
    print(f"  Best MAPE: {min(history['mape']):.2f}%")
    print(f"  Best Round: {np.argmin(history['mape']) + 1}")
    
    return history

if __name__ == "__main__":
    history = run_fedavg_only(".")
    print("\n✅ Visualizations saved: fedavg_no_early_stop.png")