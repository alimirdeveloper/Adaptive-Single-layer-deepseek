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
import argparse
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# Keras 3's seedless initializers draw from Python's `random`, not from the
# TF global RNG, so `--seed` must seed it too or the initial weights are not
# reproducible across processes (measured in asla_privacy_attack.py).
random.seed(42)
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
        print(f"\n  {csv_file}")
        print(f"   Columns: {df.columns.tolist()}")
        print("-"*40)

# Genuine utility-level series. PJME / PJMW / PJM_Load are PJM *aggregates*
# (east region, west region, total), and pjm_hourly_est.csv is a wide table
# whose first numeric column is a second copy of AEP - none of them is a
# client. Only the files named below are candidates.
UTILITY_SERIES = ('AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI')


def _load_one_series(path):
    """Read one PJM CSV -> (timestamp-indexed Series, value column name).

    Duplicate timestamps are collapsed by their mean. They are the DST
    fall-back hour, which occurs twice in local time (e.g. 2014-11-02 02:00),
    so averaging keeps both readings instead of silently dropping one.
    Unparsable timestamps are dropped.
    """
    df = pd.read_csv(path)
    dt_col = None
    for col in df.columns:
        if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.5:
            continue  # numeric column -> a load/value column, not a timestamp
        if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.9:
            dt_col = col
            break
    if dt_col is None:
        return None, None
    val_col = None
    for col in df.columns:
        if col == dt_col:
            continue
        if pd.to_numeric(df[col], errors='coerce').notna().sum() > 1000:
            val_col = col
            break
    if val_col is None:
        return None, None
    ts = pd.to_datetime(df[dt_col], errors='coerce')
    s = pd.Series(pd.to_numeric(df[val_col], errors='coerce').values, index=ts)
    s = s[s.index.notna()]
    s = s.groupby(level=0).mean().sort_index()
    return s, val_col


def load_pjm_multiple_files(data_folder='.', return_meta=False):
    """Load the genuine utility series on one shared hourly grid.

    Replaces the old "first numeric column of every *.csv, then take the
    oldest N rows of each" loader, which (a) admitted PJM aggregates and a
    duplicate copy of AEP as clients and (b) aligned clients by row position
    even though their series cover different calendar periods.

    Selection: only UTILITY_SERIES files are read. A series that cannot
    overlap the others is excluded and reported (NI ends 2011-01-01, before
    EKPC begins 2013-06-01, so it cannot share an interval with the rest).

    Alignment: the shared interval is [max(first), min(last)] over the
    surviving series, sampled hourly. Duplicate timestamps are averaged
    (DST fall-back); the few missing hours (DST spring-forward) are filled by
    time interpolation and counted in the returned metadata. No other
    imputation, resampling or smoothing is applied.
    """
    series, value_cols = {}, {}
    print(f"\nScanning {len(UTILITY_SERIES)} utility series "
          f"(PJM aggregates and pjm_hourly_est.csv are excluded)...")
    for name in UTILITY_SERIES:
        path = os.path.join(data_folder, name + '_hourly.csv')
        if not os.path.exists(path):
            print(f"  {name:<8} not found - skipped")
            continue
        s, col = _load_one_series(path)
        if s is None or len(s) < 1000:
            print(f"  {name:<8} no usable load column - skipped")
            continue
        series[name] = s
        value_cols[name] = col
        print(f"  {name:<8} {len(s):>7} readings  {s.index[0]} .. {s.index[-1]}")

    if not series:
        raise ValueError("No valid numeric data found")

    # Drop any series that cannot overlap the others. The binding constraint
    # is the series that ends first, so that is the one removed.
    excluded = []
    while len(series) > 2:
        lo = max(s.index[0] for s in series.values())
        hi = min(s.index[-1] for s in series.values())
        if hi > lo:
            break
        victim = min(series, key=lambda n: series[n].index[-1])
        print(f"  EXCLUDED {victim}: ends {series[victim].index[-1]}, before the "
              f"others begin - no shared interval")
        excluded.append(victim)
        del series[victim]

    lo = max(s.index[0] for s in series.values())
    hi = min(s.index[-1] for s in series.values())
    if hi <= lo:
        raise ValueError("No common time interval across the utility series")

    grid = pd.date_range(start=lo, end=hi, freq='h')
    client_names = sorted(series)
    columns, filled = [], {}
    for name in client_names:
        s = series[name].reindex(grid)
        n_missing = int(s.isna().sum())
        if n_missing:
            s = s.interpolate(method='time', limit_direction='both')
        filled[name] = n_missing
        columns.append(s.values)
    aligned_data = np.column_stack(columns)

    print(f"\n  shared interval  : {lo} .. {hi}")
    print(f"  sampling         : hourly ({len(grid)} hours)")
    print(f"  clients          : {len(client_names)} -> {', '.join(client_names)}")
    print(f"  missing hours    : "
          + ", ".join(f"{n}={filled[n]}" for n in client_names))
    print(f"  data shape       : {aligned_data.shape}")

    if not return_meta:
        return aligned_data, client_names
    meta = {
        'interval_start': str(lo),
        'interval_end': str(hi),
        'hours': len(grid),
        'clients': client_names,
        'missing_filled': filled,
        'excluded': excluded,
        'value_columns': value_cols,
    }
    return aligned_data, client_names, meta

# ============================================================================
# PART 2: FEATURE PREPROCESSING
# ============================================================================

def prepare_features(data, lookback=24):
    """Prepare features for load forecasting"""
    features, targets = [], []
    for i in range(max(lookback, 168), len(data)):
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
    return total_weights * (bits / 8) / 1024

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
# PART 5B: CLIENT-SIDE DIFFERENTIAL PRIVACY (opt-in, per-round)
# ============================================================================

def apply_dp_none(weights, clip, sigma):
    """No-DP path: return the layer weights unchanged.

    `clip` and `sigma` are accepted only so that every DP method shares the
    same signature; they are deliberately ignored here.
    """
    return weights

def apply_dp_gaussian(weights, clip, sigma, diag=None):
    """Per-round local Gaussian mechanism on ONE layer's parameters.

    flatten (kernel + bias -> one vector) -> global L2 clip to `clip` ->
    add zero-mean Gaussian noise (std = sigma * clip) -> reshape back to the
    original array shapes.

    Disabled (weights returned unchanged) unless BOTH clip > 0.0 and
    sigma > 0.0, which keeps the previous behaviour exactly.
    Per-round LOCAL mechanism only - no end-to-end privacy accounting.

    `diag` is the OPTIONAL diagnostics sink (opt-in via --dp-diagnostics).
    When a dict is passed it is filled IN PLACE with the scalar norms of this
    very update (raw/clipped/noise/noisy L2 and the clip factor), measured on
    the same arrays that are returned. The noise vector is drawn once and only
    once and is never regenerated for the sake of measurement, so the DP update
    is bit-identical whether or not `diag` is supplied. With the default
    `diag=None` not a single extra operation is performed.
    """
    if clip <= 0.0 or sigma <= 0.0:
        return weights
    shapes = [t.shape for t in weights]
    flat = np.concatenate([t.reshape(-1) for t in weights])
    norm = float(np.linalg.norm(flat))
    clip_factor = 1.0
    if norm > clip:
        clip_factor = clip / norm
        flat = flat * clip_factor
    if diag is not None:
        # Measured on the real vector, never assumed: clipped_delta_l2 is the
        # actual norm of the array that is about to be noised.
        diag['raw_delta_l2'] = norm
        diag['clipped_delta_l2'] = float(np.linalg.norm(flat))
        diag['clip_factor'] = float(clip_factor)
    # The one and only noise draw. Keeping it in a named variable is what lets
    # the diagnostics below measure the EXACT noise that is added, instead of
    # generating a second noise vector.
    noise = np.random.normal(0.0, sigma * clip, size=flat.shape)
    flat = flat + noise
    if diag is not None:
        noise_l2 = float(np.linalg.norm(noise))
        diag['noise_l2'] = noise_l2
        diag['noisy_delta_l2'] = float(np.linalg.norm(flat))
        # Undefined (None -> empty CSV cell) when the raw delta is exactly zero.
        diag['noise_to_raw_delta_ratio'] = (noise_l2 / norm) if norm > 0.0 else None
    parts, pos = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        parts.append(flat[pos:pos + n].reshape(shape))
        pos += n
    return parts

def apply_dp_clip_only(weights, clip, sigma=None):
    """ABLATION MODE: global L2 clipping of one layer's delta, WITHOUT noise.

    flatten (kernel + bias -> one vector) -> if the combined L2 norm exceeds
    `clip`, scale the vector by clip / norm -> reshape back to the original
    array shapes.

    No noise is added and NO random number is drawn, so the NumPy RNG stream is
    left exactly where it was: a clip_only run is reproducible from the seeds
    alone and cannot perturb the stream the 'gaussian' arm relies on.
    `clip <= 0` returns the input unchanged.

    `sigma` is accepted and deliberately ignored, only so that every method in
    DP_METHODS shares one call signature (same convention as apply_dp_none).

    WARNING: this is a control arm for the ablation, NOT a privacy mechanism.
    Clipping bounds the norm of an update but hides nothing: the clipped delta
    is still transmitted, and scaling preserves its direction exactly. It
    provides NO differential-privacy guarantee.
    """
    if clip <= 0.0:
        return weights
    shapes = [t.shape for t in weights]
    flat = np.concatenate([t.reshape(-1) for t in weights])
    norm = float(np.linalg.norm(flat))
    if norm > clip:
        clip_factor = clip / norm
        flat = flat * clip_factor
    parts, pos = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        parts.append(flat[pos:pos + n].reshape(shape))
        pos += n
    return parts

# Registry of the supported client-side DP methods (default: 'none').
DP_METHODS = {
    'none': apply_dp_none,
    'gaussian': apply_dp_gaussian,
    'clip_only': apply_dp_clip_only,
}

def get_dp_method(name):
    """Select a client-side DP function by name (unknown/None -> 'none')."""
    return DP_METHODS.get(name, apply_dp_none)

# ----------------------------------------------------------------------------
# PART 5C: OPTIONAL GAUSSIAN-DP DIAGNOSTICS (opt-in, default OFF)
# ----------------------------------------------------------------------------
# Scalars recorded for EVERY client update, in CSV column order. Only norms and
# identifiers are stored - never weights, deltas, gradients or any other
# parameter value.
DP_DIAGNOSTICS_HEADER = ('round', 'client_id', 'layer_index', 'raw_delta_l2',
                         'clipped_delta_l2', 'noise_l2', 'noisy_delta_l2',
                         'clip_factor', 'noise_to_raw_delta_ratio')

def dp_diagnostics_active(dp_method, dp_clip, dp_sigma):
    """True only when there is an active client-side DP method to observe.

    Mirrors the rule used by the console banner and the report: 'gaussian' is in
    effect only with BOTH clip and sigma positive. 'clip_only' is in effect with
    a positive clip bound alone, because it adds no noise by design (its noise
    columns are recorded as zero). Everything else has nothing to observe.
    """
    if dp_method == 'clip_only':
        return dp_clip > 0.0
    return dp_method == 'gaussian' and dp_clip > 0.0 and dp_sigma > 0.0

def unique_output_path(path):
    """Return `path`, or a `_1`/`_2`/... sibling if `path` already exists.

    Diagnostics must never overwrite an existing file silently, so an existing
    target is never reused; the caller reports the name that was chosen.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"

def format_dp_diagnostics_row(row):
    """Render one diagnostics dict as a CSV line (scalars only).

    Every field is an int, a float or None, so no quoting is ever required and
    the writer stays dependency-free (the stdlib `csv` module is not imported).
    Floats use repr() so the written value round-trips exactly.
    """
    cells = []
    for key in DP_DIAGNOSTICS_HEADER:
        value = row.get(key)
        if value is None:
            cells.append('')
        elif isinstance(value, float):
            cells.append(repr(value))
        else:
            cells.append(str(value))
    return ','.join(cells)

# ============================================================================
# PART 6: FEDERATED CLIENT
# ============================================================================

class FederatedClient:
    def __init__(self, client_id, X_train, y_train, X_test, y_test, name="", quant_bits=32,
                 dp_clip=0.0, dp_sigma=0.0, dp_method='none',
                 dp_diagnostics=False):
        self.client_id = client_id
        self.name = name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = create_model()
        self.quant_bits = quant_bits
        # Client-side DP (opt-in). The method is chosen by name; the default
        # 'none' leaves the transmitted weights unchanged:
        #   dp_method = 'none' | 'gaussian' | 'clip_only'
        #               (see DP_METHODS / get_dp_method)
        #   dp_clip   = global L2 clipping bound C on the DELTA of the selected
        #               layer (delta = local_weights - round reference G)
        #   dp_sigma  = noise multiplier; added noise std = dp_sigma * C
        self.dp_method = dp_method
        self.dp_clip = dp_clip
        self.dp_sigma = dp_sigma
        # Round reference G for the 'gaussian' method, keyed by layer index and
        # filled by set_layer_weights with an INDEPENDENT copy. The 'none'
        # method never reads it and never fills it.
        self.received_layer_weights = {}
        # Optional DP diagnostics (opt-in, default OFF). When disabled nothing
        # is collected and nothing is measured: dp_diag_rows stays empty and
        # apply_dp_gaussian is called without a diagnostics sink.
        self.dp_diagnostics = dp_diagnostics
        self.dp_diag_rows = []
        # Communication round of the update currently being produced, tagged by
        # ASLAServer.do_round (None when diagnostics are not in use).
        self.dp_diag_round = None
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
        # Client-side DP (opt-in) on the selected layer, i.e. exactly the
        # parameters that are sent to the server.
        #   'none' (default) - returns `w` unchanged (apply_dp_none is an
        #       identity), so this path is identical to the non-DP baseline.
        #   'gaussian' - ACTIVE only when both dp_clip and dp_sigma are positive
        #       (the same rule as the console banner and the report). It noises
        #       the DELTA against the round reference G (the layer the server
        #       sent at the START of this round), then adds the noised delta back
        #       to G. The returned value therefore stays ABSOLUTE and the
        #       server's mean-of-weights aggregation stays mathematically
        #       correct: mean_i(G + d~_i) = G + mean_i(d~_i).
        #       When it is NOT active (dp_clip <= 0 or dp_sigma <= 0) the
        #       ORIGINAL local weights go straight to the quantization path
        #       below: no delta is computed and no G + delta is reconstructed, so
        #       the result is bit-identical to the 'none' path, not merely close.
        #   'clip_only' - ABLATION arm, active whenever dp_clip is positive
        #       (dp_sigma is ignored). Same delta definition and same global L2
        #       clipping as 'gaussian', but NO noise is added and no random
        #       number is consumed, so the RNG stream is the one a 'none' run
        #       would have. Clipping alone is NOT a privacy guarantee: the
        #       clipped delta is still sent and keeps its direction.
        #       When it is NOT active (dp_clip <= 0) the ORIGINAL local weights
        #       go straight to the quantization path below, bit-identical to the
        #       'none' path.
        #   Optional diagnostics (--dp-diagnostics, default OFF) only MEASURE
        #       that same delta and that same noise; they never change the
        #       update, the clipping or the number of random draws.
        if self.dp_method == 'gaussian':
            if self.dp_clip > 0.0 and self.dp_sigma > 0.0:
                ref = self.received_layer_weights.get(layer_idx)
                if ref is None:
                    raise RuntimeError(
                        f"client-side DP method 'gaussian' requires the round "
                        f"reference weights for layer {layer_idx}, but none were "
                        f"received (set_layer_weights was never called for this "
                        f"layer). Refusing to silently noise the absolute weights "
                        f"instead.")
                delta = [a - b for a, b in zip(w, ref)]
                if self.dp_diagnostics:
                    # Diagnostics ON: collect scalars measured from the SAME
                    # delta and the SAME noise the update uses. The sink dict is
                    # filled in place by apply_dp_gaussian.
                    diag = {}
                    delta = apply_dp_gaussian(delta, self.dp_clip,
                                              self.dp_sigma, diag=diag)
                    diag['round'] = self.dp_diag_round
                    diag['client_id'] = self.client_id
                    diag['layer_index'] = layer_idx
                    self.dp_diag_rows.append(diag)
                else:
                    # Diagnostics OFF (default): the original call, unchanged.
                    delta = apply_dp_gaussian(delta, self.dp_clip, self.dp_sigma)
                w = [a + b for a, b in zip(ref, delta)]
        elif self.dp_method == 'clip_only':
            # ABLATION ARM: the same delta definition and the same global L2
            # clipping as the 'gaussian' method, but NO noise is added and no
            # random number is consumed, so the NumPy RNG stream stays exactly
            # the one a 'none' run would have. Clipping alone is NOT a privacy
            # guarantee: the clipped delta is still transmitted, and scaling
            # preserves its direction exactly.
            if self.dp_clip > 0.0:
                ref = self.received_layer_weights.get(layer_idx)
                if ref is None:
                    raise RuntimeError(
                        f"client-side DP method 'clip_only' requires the round "
                        f"reference weights for layer {layer_idx}, but none were "
                        f"received (set_layer_weights was never called for this "
                        f"layer). Refusing to silently clip the absolute weights "
                        f"instead.")
                delta = [a - b for a, b in zip(w, ref)]
                if self.dp_diagnostics:
                    # Measured on the real delta, before clipping. No random
                    # number is drawn anywhere in this branch.
                    raw_l2 = float(np.linalg.norm(
                        np.concatenate([d.reshape(-1) for d in delta])))
                delta = apply_dp_clip_only(delta, self.dp_clip)
                if self.dp_diagnostics:
                    clipped_l2 = float(np.linalg.norm(
                        np.concatenate([d.reshape(-1) for d in delta])))
                    self.dp_diag_rows.append({
                        'round': self.dp_diag_round,
                        'client_id': self.client_id,
                        'layer_index': layer_idx,
                        'raw_delta_l2': raw_l2,
                        'clipped_delta_l2': clipped_l2,
                        # This arm adds no noise at all.
                        'noise_l2': 0.0,
                        'noisy_delta_l2': clipped_l2,
                        # Same definition as the Gaussian arm: the multiplier
                        # actually applied to the original delta.
                        'clip_factor': (self.dp_clip / raw_l2
                                        if raw_l2 > self.dp_clip else 1.0),
                        # 0 / raw is 0 when raw > 0; undefined (empty cell) at 0.
                        'noise_to_raw_delta_ratio': (0.0 if raw_l2 > 0.0
                                                     else None),
                    })
                w = [a + b for a, b in zip(ref, delta)]
        else:
            w = get_dp_method(self.dp_method)(w, self.dp_clip, self.dp_sigma)
        return quantize_weights(w, self.quant_bits) if self.quant_bits < 32 else w
    
    def set_layer_weights(self, layer_idx, weights):
        set_layer_weights(self.model, layer_idx, weights)
        # Keep the round reference G for the 'gaussian' / 'clip_only' methods as
        # an INDEPENDENT copy, so that later in-place updates of the arrays
        # cannot corrupt the reference. The 'none' method never reads it and pays
        # nothing for it.
        if self.dp_method in ('gaussian', 'clip_only'):
            self.received_layer_weights[layer_idx] = [np.array(w, copy=True)
                                                      for w in weights]
    
    def check_stopping(self, loss, patience=5, min_delta=1e-6):
        if loss is None:
            return False
        if loss < self.best_loss - min_delta:
            self.best_loss = loss
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= patience:
                self.stopped = True
        return self.stopped


class ASLAServer:
    """
    کلاس سرور برای چارچوب ASLA (تجمیع تطبیقی تک لایه)
    Adaptive Single-Layer Aggregation Server
    این کلاس پیاده‌کننده روش اصلی مقاله است
    """
    
    def __init__(self, num_clients, agg_layer=1, quant_bits=32):
        """
        سازنده کلاس - مقداردهی اولیه سرور ASLA
        
        پارامترها:
            num_clients: تعداد کلاینت‌ها (دستگاه‌های IoT)
            agg_layer: شماره لایه‌ای که قرار است تجمیع شود (پیش‌فرض: 1 = لایه دوم)
            quant_bits: تعداد بیت‌های کوانتیزاسیون (8, 16, 32)
        """
        self.global_model = create_model()  # ایجاد مدل سراسری (شامل ۳ لایه)
        self.num_clients = num_clients      # تعداد کلاینت‌ها
        self.agg_layer = agg_layer          # لایه انتخاب شده برای تجمیع
        self.quant_bits = quant_bits        # بیت‌های کوانتیزاسیون
        self.comm_round = 0                 # شمارنده دورهای ارتباطی
        # Stage 1B: record the execution facts needed for a trustworthy baseline.
        # 'participants' = fresh clients that actually trained in that round.
        self.history = {'mape': [], 'mae': [], 'participants': []}
    
    def aggregate(self, client_weights):
        """
        تابع تجمیع ASLA - قلب تپنده روش پیشنهادی
        اینجا فقط یک لایه (نه همه لایه‌ها) تجمیع می‌شود
        
        فرمول: G_t^selected = (1/K) * Σ W_{t,i}^{selected}
        
        ورودی: لیست وزن‌های دریافتی از کلاینت‌ها (فقط یک لایه)
        خروجی: وزن‌های تجمیع شده برای آن لایه
        """
        if not client_weights:
            return None
        
        avg_weights = []
        # برای هر لایه (اما فقط یک لایه ارسال شده است!)
        for i in range(len(client_weights[0])):
            # جمع تمام وزن‌های کلاینت‌ها برای این لایه
            # محاسبه میانگین: (1/K) * Σ W_i
            avg_weights.append(np.mean([w[i] for w in client_weights], axis=0))
        
        return avg_weights  # بازگشت وزن تجمیع شده برای یک لایه
    
    def do_round(self, clients, local_epochs=1, batch_size=300):
        """
        اجرای یک دور کامل از یادگیری فدرال با روش ASLA
        
        مراحل:
        1. ارسال لایه انتخاب‌شده از سرور به کلاینت‌ها
        2. آموزش محلی روی هر کلاینت
        3. دریافت فقط همان لایه از کلاینت‌ها
        4. تجمیع فقط همان لایه (با تابع aggregate)
        5. ارزیابی مدل سراسری
        """
        self.comm_round += 1  # افزایش شمارنده دور
        
        # مرحله ۱: دریافت لایه انتخاب شده از مدل سراسری
        global_weights = get_layer_weights(self.global_model, self.agg_layer)
        
        updates = []  # لیست برای ذخیره به‌روزرسانی‌های کلاینت‌ها
        
        # مرحله ۲: ارسال به تمام کلاینت‌ها و دریافت به‌روزرسانی
        for client in clients:
            # اگر کلاینت متوقف شده باشد، آخرین به‌روزرسانی را استفاده کن
            # B2 fix: stopped clients are excluded from aggregation entirely.
            # (The comment on the line above is now STALE - it described the old
            # behaviour of re-injecting the cached update.)
            # Re-injecting the cached `last_update` (an absolute weight vector
            # from an older round) biased np.mean toward a stale model and
            # silently changed the effective participation count.
            if client.stopped:
                continue
            
            # ارسال لایه انتخاب شده به کلاینت (فقط یک لایه!)
            client.set_layer_weights(self.agg_layer, global_weights)
            # DP diagnostics (inert unless enabled): tag this client's update
            # with the current communication round. Assigning an attribute has
            # no numerical effect on training or on the RNG sequence.
            client.dp_diag_round = self.comm_round
            
            # مرحله ۳: آموزش محلی روی داده‌های کلاینت
            loss = client.train_local(local_epochs, batch_size)
            
            # اگر آموزش موفق بود، وزن لایه را دریافت کن
            if loss is not None:
                # دریافت فقط همان لایه از کلاینت (نه همه لایه‌ها)
                updates.append(client.get_layer_weights(self.agg_layer))
                client.last_update = [
                                            w.copy()
                                            for w in updates[-1]
                                    ]  # ذخیره آخرین به‌روزرسانی
                client.check_stopping(loss)  # بررسی شرط توقف
        
        # مرحله ۴: تجمیع فقط یک لایه (فرمول اصلی ASLA!)
        # Stage 1B guard: if no client produced a fresh update there is nothing
        # to aggregate. Stop explicitly rather than crashing or silently logging
        # the unchanged global model's MAPE as if it were this round's result.
        if not updates:
            print(f"  Round {self.comm_round}: no fresh client update - stopping")
            return False
        
        agg_weights = self.aggregate(updates)  # ← قلب ASLA اینجاست
        set_layer_weights(self.global_model, self.agg_layer, agg_weights)
        n_participants = len(updates)  # Stage 1B: fresh clients contributing this round
        
        # مرحله ۵: ارزیابی مدل سراسری روی همه کلاینت‌ها
        mape_values = []
        mae_values = []  # Stage 1B: MAE collected from the same evaluate() call
        for client in clients:
            # تنظیم لایه تجمیع شده روی کلاینت برای ارزیابی
            client.set_layer_weights(self.agg_layer, 
                get_layer_weights(self.global_model, self.agg_layer))
            res = client.evaluate()  # محاسبه خطا روی داده‌های تست
            if len(res) >= 3 and not np.isinf(res[2]):
                mape_values.append(res[2])  # ذخیره MAPE
            if len(res) >= 2 and not np.isinf(res[1]):
                mae_values.append(res[1])   # Stage 1B: ذخیره MAE
        
        # محاسبه میانگین MAPE/MAE و ذخیره در تاریخچه (Stage 1B: + participants)
        if mape_values:
            avg_mape = np.mean(mape_values)
            self.history['mape'].append(avg_mape)
            self.history['mae'].append(np.mean(mae_values) if mae_values else float('nan'))
            self.history['participants'].append(n_participants)
            print(f"  Round {self.comm_round}: MAPE = {avg_mape:.2f}% "
                  f"({n_participants} clients)")
        
        # بررسی شرط توقف: اگر بیش از ۳۰٪ کلاینت‌ها متوقف شده‌اند، پایان
        stopped = sum(1 for c in clients if c.stopped)
        return stopped < 0.3 * len(clients)
    
    def train(self, clients, num_rounds=20):
        """
        حلقه اصلی آموزش - اجرای چندین دور ارتباطی
        
        ورودی: لیست کلاینت‌ها، تعداد دورها
        خروجی: تاریخچه خطا (MAPE) در هر دور
        """
        for _ in range(num_rounds):
            # هر دور یک مرحله do_round را اجرا کن
            if not self.do_round(clients):
                break  # اگر شرط توقف برقرار شد، متوقف کن
        return self.history  # بازگشت تاریخچه خطاها


class FedAvgServer:
    """
    کلاس سرور برای روش سنتی FedAvg با Early Stopping
    """
    
    # def __init__(self, num_clients, quant_bits=32, patience=10):
    def __init__(self, num_clients, quant_bits=32, patience=999):
        self.global_model = create_model()
        self.num_clients = num_clients
        self.quant_bits = quant_bits
        self.comm_round = 0
        self.history = {'mape': []}
        # ✅ Early Stopping parameters
        self.patience = patience
        self.best_mape = float('inf')
        self.no_improve_count = 0
        self.early_stopped = False
    
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
            if client.stopped:
                continue
            
            for i, w in enumerate(global_weights):
                client.set_layer_weights(i, w)
            
            loss = client.train_local(local_epochs, batch_size)
            
            if loss is not None:
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
            # B1 fix: evaluate the CURRENT aggregated global model instead of
            # the pre-aggregation snapshot captured in `global_weights`, which
            # made every logged FedAvg MAPE one round stale.
            for i in range(len(self.global_model.layers)):
                client.set_layer_weights(i, get_layer_weights(self.global_model, i))
            res = client.evaluate()
            if len(res) >= 3 and not np.isinf(res[2]):
                mape_values.append(res[2])
        
        if mape_values:
            avg_mape = np.mean(mape_values)
            self.history['mape'].append(avg_mape)
            print(f"  Round {self.comm_round}: MAPE = {avg_mape:.2f}%")
            
            # ✅ Early Stopping برای FedAvg
            if avg_mape < self.best_mape - 0.01:
                self.best_mape = avg_mape
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience:
                    print(f"  ⏹ Early Stopping at round {self.comm_round} (no improvement for {self.patience} rounds)")
                    self.early_stopped = True
                    return False
        
        stopped = sum(1 for c in clients if c.stopped)
        return stopped < 0.3 * len(clients)
    
    def train(self, clients, num_rounds=100):
        for _ in range(num_rounds):
            if not self.do_round(clients):
                if self.early_stopped:
                    print(f"  ✅ FedAvg converged early at round {self.comm_round}")
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
    """Run complete experiment - مطابق با مقاله اصلی (۱۰۰ دور)"""
    print("="*60)
    print("ASLA FRAMEWORK - مطابق با مقاله اصلی")
    print("Data: 10 clients, ~13896 samples each, 100 rounds")
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
    for i, name in enumerate(client_names[:10]):  # ✅ فقط ۱۰ کلاینت اول (مطابق مقاله)
        X_train, X_test, y_train, y_test, _ = preprocess_client_data(client_data[:, i])
        if X_train is not None and len(X_train) > 100:
            clients.append(FederatedClient(i, X_train, y_train, X_test, y_test, name))
            print(f"  ✓ {name}: {len(X_train)} samples")
    
    print(f"\n✅ Using {len(clients)} clients")
    
    # Experiment 1: Find best aggregation layer (100 rounds - مطابق مقاله)
    print("\n" + "="*60)
    print("EXPERIMENT 1: Finding Best Aggregation Layer (100 rounds)")
    print("="*60)
    
    layer_results = {}
    for layer_idx, layer_name in [(0, 'Layer 1'), (1, 'Layer 2'), (2, 'Layer 3')]:
        print(f"\n--- {layer_name} ---")
        server = ASLAServer(len(clients), agg_layer=layer_idx)
        fresh = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                for c in clients]
        history = server.train(fresh, num_rounds=100)  # ✅ ۲۰ → ۱۰۰
        layer_results[layer_name] = history['mape'][-1] if history['mape'] else 100
        print(f"   Final MAPE: {layer_results[layer_name]:.2f}%")
    
    best_layer = min(layer_results, key=layer_results.get)
    print(f"\n🏆 Best layer: {best_layer} with MAPE = {layer_results[best_layer]:.2f}%")
    
    # Experiment 2: Quantization (50 rounds - مطابق مقاله بخش ۵.۳)
    print("\n" + "="*60)
    print("EXPERIMENT 2: Quantization Effects (50 rounds - مطابق مقاله)")
    print("="*60)
    
    quant_results = {}
    for bits, name in [(32, '32-bit'), (16, '16-bit'), (8, '8-bit')]:
        print(f"\n--- {name} System ---")
        server = ASLAServer(len(clients), agg_layer=1, quant_bits=bits)
        fresh = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name, bits) 
                for c in clients]
        history = server.train(fresh, num_rounds=50)  # ✅ طبق مقاله برای کوانتیزاسیون ۵۰ دور
        quant_results[name] = history
        print(f"   Final MAPE: {history['mape'][-1]:.2f}%")
    
    plot_quantization(quant_results)
    
    # Experiment 3: ASLA vs FedAvg (100 rounds - مطابق مقاله)
    print("\n" + "="*60)
    print("EXPERIMENT 3: ASLA vs FedAvg Comparison (100 rounds)")
    print("="*60)
    
    # FedAvg
    print("\n--- FedAvg (All Layers) ---")
    fedavg_server = FedAvgServer(len(clients))
    fedavg_clients = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                     for c in clients]
    fedavg_history = fedavg_server.train(fedavg_clients, num_rounds=100)  # ✅ ۲۰ → ۱۰۰
    
    # ASLA (best layer)
    print("\n--- ASLA (Single Layer - Layer 2) ---")
    asla_server = ASLAServer(len(clients), agg_layer=1)
    asla_clients = [FederatedClient(c.client_id, c.X_train, c.y_train, c.X_test, c.y_test, c.name) 
                   for c in clients]
    asla_history = asla_server.train(asla_clients, num_rounds=100)  # ✅ ۲۰ → ۱۰۰
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    model = create_model()
    model_size = calculate_model_size(model, 32)
    layer_size = (5000 + 50) * 4 / 1024
    
    print(f"\n📊 Communication per round (32-bit):")
    print(f"   FedAvg (full model): {model_size:.2f} KB")
    print(f"   ASLA (single layer): {layer_size:.2f} KB")
    print(f"   🚀 ASLA saves: {model_size/layer_size:.1f}x per round!")
    
    # محاسبه با دورهای واقعی
    fedavg_rounds = len(fedavg_history['mape'])
    asla_rounds = len(asla_history['mape'])
    
    print(f"\n📊 Total Communication ({len(clients)} clients):")
    fedavg_total = model_size * fedavg_rounds * len(clients) * 2
    asla_total = layer_size * asla_rounds * len(clients) * 2
    print(f"   FedAvg: {fedavg_total:.2f} KB ({fedavg_rounds} rounds)")
    print(f"   ASLA:   {asla_total:.2f} KB ({asla_rounds} rounds)")
    if asla_total > 0:
        print(f"   🚀 ASLA total saving: {fedavg_total/asla_total:.1f}x")
    
    print(f"\n📊 Accuracy (Final MAPE):")
    print(f"   FedAvg: {fedavg_history['mape'][-1]:.2f}%")
    print(f"   ASLA:   {asla_history['mape'][-1]:.2f}%")
    
    if asla_history['mape'][-1] < fedavg_history['mape'][-1]:
        improvement = fedavg_history['mape'][-1] - asla_history['mape'][-1]
        print(f"   🎯 ASLA is {improvement:.2f}% MORE accurate!")
    
    # Plot comparison
    plot_comparison(fedavg_history, asla_history)
    
    # Paper comparison
    print("\n" + "="*60)
    print("PAPER COMPARISON (Section 7)")
    print("="*60)
    print("\nPaper's claims (Data 1):")
    print("  • ASLA Final MAPE: ~3.3%")
    print("  • FedAvg Final MAPE: ~35-40%")
    print("  • Communication reduction: 829.2x")
    print("  • Memory saving (32-bit→8-bit): 75%")
    
    print(f"\nYour results with PJM data ({fedavg_rounds} rounds):")
    if fedavg_history['mape']:
        print(f"  • FedAvg Final MAPE: {fedavg_history['mape'][-1]:.2f}%")
    if asla_history['mape']:
        print(f"  • ASLA Final MAPE: {asla_history['mape'][-1]:.2f}%")
    print(f"  • ASLA communication saving: {fedavg_total/asla_total:.1f}x")
    print(f"  • Memory saving (32-bit→8-bit): 75%")
    
    return layer_results, quant_results, fedavg_history, asla_history
def run_asla_baseline(data_folder='.', num_rounds=100, agg_layer=1, quant_bits=32,
                      num_clients=10, plot=False, out_file='baseline_asla_results.txt',
                      dp_clip=0.0, dp_sigma=0.0, dp_method='none',
                      dp_diagnostics=False,
                      dp_diagnostics_out='dp_diagnostics.csv',
                      json_out=''):
    """ASLA-only baseline (Stage 1 minimal-fix run).

    Runs ONLY the single-layer ASLA baseline, skipping the layer sweep, the
    quantization sweep, the FedAvg comparison and the blocking plt.show().

    DECISION RECORDED (B5) - documented, not changed: only `agg_layer` is ever
    aggregated, so the reported MAPE is the mean over K *personalised hybrid*
    models (shared global selected layer + each client's own other layers),
    NOT the error of one single global model, and not directly comparable to
    FedAvg's metric. Left as-is because changing it would change the ASLA
    method rather than fix a bug.

    Documented only, not changed:
      B3 - NaNs are dropped before the lag features are built. The corrected
           loader now returns a gap-free hourly grid (missing hours are
           interpolated and counted), so the drop is a no-op and prev_day /
           prev_week are exactly 24 / 168 hours back.
      B6 - the client set is the genuine utility series that share a common
           interval (see load_pjm_multiple_files); PJM aggregates and the
           duplicate AEP copy are excluded, and the selected names, the shared
           interval and the per-client sample counts are printed and written
           to the report.
      B7 - per-client early stopping and the ">= 30% stopped" rule can end the
           run before `num_rounds`; the completed round count is reported.
      DP - client-side DP is opt-in via dp_method (default 'none' = OFF, so the
           default path is identical to the non-DP baseline). Methods: 'none'
           (apply_dp_none, returns the weights unchanged), 'gaussian'
           (apply_dp_gaussian on the DELTA against the round reference G that
           the server sent, parameterised by dp_clip / dp_sigma, both default
           0.0; the noised delta is added back to G, so the client still returns
           ABSOLUTE weights) and 'clip_only' (apply_dp_clip_only: the same delta
           definition and the same global L2 clipping as 'gaussian', but with NO
           noise and no random numbers consumed - an ablation control arm, NOT a
           privacy mechanism and NOT a DP guarantee). 'gaussian' is a per-round
           LOCAL mechanism on the transmitted layer only; no end-to-end privacy
           accounting across rounds is implemented.
      DP diagnostics - opt-in via dp_diagnostics (default False). It only makes
           sense for an ACTIVE method ('gaussian', or 'clip_only' with a positive
           clip bound); when enabled, the scalar norms of every client update
           (raw/clipped/noise/noisy L2 and the clip factor; the noise columns are
           0.0 in the clip_only arm) are appended to `dp_diagnostics_out` as
           CSV. It is a measurement layer only: the DP update, the RNG sequence,
           the clipping, the noise, the quantization and the aggregation are all
           untouched, and with the flag off nothing extra is computed.
      json_out - optional path for a machine-readable record of the run
           (status, configuration, data interval, per-round history and
           per-client final metrics). Empty by default = no JSON written.
    """
    print("=" * 60)
    # Console banner: the title must match the method actually in effect, so a
    # DP run is never labelled as a no-DP run and the noise-free ablation is
    # never labelled as Gaussian DP (same rule as the report below).
    if dp_method == 'clip_only':
        dp_label = 'clip-only (no noise)' if dp_clip > 0.0 else 'none'
        dp_state = ('ON, clip-only - NO Gaussian noise' if dp_clip > 0.0
                    else 'OFF')
    elif dp_method == 'gaussian' and dp_clip > 0.0 and dp_sigma > 0.0:
        dp_label = 'gaussian'
        dp_state = 'ON'
    else:
        dp_label = 'none'
        dp_state = 'OFF'
    print(f"ASLA BASELINE (single-layer, DP: {dp_label})")
    print("=" * 60)

    inspect_file_structure(data_folder)
    client_data, client_names, data_meta = load_pjm_multiple_files(
        data_folder, return_meta=True)

    clients = []
    for i, name in enumerate(client_names[:num_clients]):
        X_train, X_test, y_train, y_test, _ = preprocess_client_data(client_data[:, i])
        if X_train is not None and len(X_train) > 100:
            clients.append(FederatedClient(i, X_train, y_train, X_test, y_test,
                                           name, quant_bits, dp_clip, dp_sigma,
                                           dp_method, dp_diagnostics))
            print(f"  client {i}: {name}  train={len(X_train)}  test={len(X_test)}")

    if not clients:
        raise RuntimeError("No usable clients were built from the CSV files in "
                           f"'{os.path.abspath(data_folder)}'")

    # Stage 1B: log the deterministic client set so the run is reproducible.
    print(f"\n  selected clients ({len(clients)}, sorted): "
          + ", ".join(c.name for c in clients))

    probe = create_model()
    probe_layer = probe.layers[agg_layer]
    layer_params = sum(w.size for w in probe_layer.get_weights())

    print(f"\n  clients used     : {len(clients)}")
    print(f"  selected layer   : model.layers[{agg_layer}] = {probe_layer.name} "
          f"({layer_params} params)")
    print(f"  quantization     : {quant_bits}-bit")
    print(f"  rounds requested : {num_rounds}")
    print("  local epochs     : 1   batch size: 300   optimizer: Adam(1e-3)")
    print("  evaluation       : mean client-level MAPE and MAE (Stage 1B)")
    print("  note             : MAPE is a personalised-hybrid metric (docstring, B5)")

    # --- Optional DP diagnostics (opt-in; nothing happens when disabled) -----
    # Only an ACTIVE DP method produces anything to measure ('gaussian', or the
    # 'clip_only' ablation with a positive clip bound), so a request that cannot
    # be honoured is explained once and then ignored: the training behaviour is
    # left exactly as it is.
    diag_out_path = None
    if dp_diagnostics:
        if dp_diagnostics_active(dp_method, dp_clip, dp_sigma):
            diag_out_path = unique_output_path(dp_diagnostics_out)
            print(f"  DP diagnostics   : ON -> {os.path.abspath(diag_out_path)}")
        else:
            what = ('clip_only arm' if dp_method == 'clip_only'
                    else 'Gaussian mechanism')
            print("  DP diagnostics   : requested but NOT possible - the "
                  f"{what} is not active (method={dp_method}, "
                  f"clip={dp_clip}, sigma={dp_sigma}); no CSV will be written "
                  "and training is unchanged.")

    server = ASLAServer(len(clients), agg_layer=agg_layer, quant_bits=quant_bits)
    history = server.train(clients, num_rounds=num_rounds)

    mape = history['mape']
    mae = history.get('mae', [])
    participants = history.get('participants', [])

    # Per-client final metrics on the held-out test split. Each client model is
    # a personalised hybrid (the shared global selected layer + that client's
    # own other layers), so these are per-client hybrid results - not the error
    # of one single global model.
    per_client = []
    for c in clients:
        res = c.evaluate()
        per_client.append({
            'client': c.name,
            'mape': float(res[2]) if len(res) >= 3 else None,
            'mae': float(res[1]) if len(res) >= 2 else None,
            'mse': float(res[0]) if len(res) >= 1 else None,
            'train_samples': int(len(c.X_train)),
            'test_samples': int(len(c.X_test)),
        })

    report = []
    report.append("=" * 60)
    report.append("ASLA BASELINE RESULTS (Stage 1B)")
    report.append("=" * 60)
    report.append(f"  data folder        : {os.path.abspath(data_folder)}")
    report.append(f"  shared interval    : {data_meta['interval_start']} .. "
                  f"{data_meta['interval_end']}")
    report.append(f"  sampling           : hourly ({data_meta['hours']} hours)")
    report.append(f"  excluded series    : "
                  + (", ".join(data_meta['excluded']) if data_meta['excluded']
                     else "<none>")
                  + " (aggregates / duplicates / no shared interval)")
    report.append(f"  selected clients   : {', '.join(c.name for c in clients)}")
    report.append(f"  selected layer     : model.layers[{agg_layer}] = {probe_layer.name} "
                  f"({layer_params} params)")
    report.append(f"  quantization       : {quant_bits}-bit")
    report.append(f"  client-side DP     : method={dp_method}, clip={dp_clip}, "
                  f"sigma={dp_sigma} ({dp_state})")
    report.append("  DP scope           : per-round local mechanism only - no")
    report.append("                       end-to-end privacy accounting.")
    if dp_method == 'clip_only' and dp_clip > 0.0:
        report.append("  clip-only caveat   : this arm clips the update norm but")
        report.append("                       adds NO noise and gives NO DP")
        report.append("                       guarantee - ablation control only.")
    report.append("  local epochs=1, batch size=300, optimizer=Adam(1e-3)")
    report.append(f"  rounds requested   : {num_rounds}")
    report.append(f"  rounds completed   : {len(mape)}")
    if mape:
        report.append(f"  final MAPE         : {mape[-1]:.2f}%")
        report.append(f"  best  MAPE         : {min(mape):.2f}% (round {int(np.argmin(mape)) + 1})")
        report.append("  last 5 MAPE        : " + ", ".join(f"{v:.2f}" for v in mape[-5:]))
        report.append(f"  final MAE          : {mae[-1]:.4f}" if mae
                      else "  final MAE          : <not available>")
        report.append(f"  fresh clients/round: {participants}")
    else:
        report.append("  final MAPE         : <no round produced a finite MAPE>")
    report.append("  note               : MAPE/MAE are the mean over personalised-hybrid")
    report.append("                       client models (B5), NOT one fully global model.")
    report.append("  MAPE policy        : Keras MAPE is already a percentage; every")
    report.append("                       client load is > 500 MW, so no near-zero")
    report.append("                       denominator occurs and no clipping is applied.")
    report.append("")
    report.append("  per-client final metrics (personalised hybrid, held-out test):")
    report.append(f"    {'client':<8} {'MAPE %':>9} {'MAE':>10} {'MSE':>14} "
                  f"{'train':>7} {'test':>7}")
    for row in per_client:
        report.append(f"    {row['client']:<8} {row['mape']:>9.2f} {row['mae']:>10.2f} "
                      f"{row['mse']:>14.1f} {row['train_samples']:>7d} "
                      f"{row['test_samples']:>7d}")
    for line in report:
        print(line)

    try:
        with open(out_file, 'w', encoding='utf-8') as fh:
            fh.write("\n".join(report) + "\n")
        print(f"\n  results saved to   : {os.path.abspath(out_file)}")
    except OSError as exc:
        print(f"\n  WARNING: could not write '{out_file}': {exc}")

    # --- Optional machine-readable record of the run -------------------------
    if json_out:
        payload = {
            'status': 'completed' if mape else 'failed',
            'config': {
                'data_folder': os.path.abspath(data_folder),
                'rounds_requested': num_rounds,
                'rounds_completed': len(mape),
                'agg_layer': agg_layer,
                'agg_layer_name': probe_layer.name,
                'agg_layer_params': int(layer_params),
                'quant_bits': quant_bits,
                'dp_method': dp_method,
                'dp_clip': dp_clip,
                'dp_sigma': dp_sigma,
                'local_epochs': 1,
                'batch_size': 300,
                'optimizer': 'Adam(1e-3)',
                'loss': 'mse',
                'lookback': 24,
                'test_size': 0.3,
                'split': 'chronological (shuffle=False)',
                'scaler': 'StandardScaler fitted on the training split only',
            },
            'data': data_meta,
            'history': {
                'mape': [float(v) for v in mape],
                'mae': [float(v) for v in mae],
                'participants': [int(v) for v in participants],
            },
            'final': {
                'mape': float(mape[-1]) if mape else None,
                'mae': float(mae[-1]) if mae else None,
                'best_mape': float(min(mape)) if mape else None,
            },
            'per_client': per_client,
        }
        try:
            with open(json_out, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2)
            print(f"  JSON saved to      : {os.path.abspath(json_out)}")
        except OSError as exc:
            print(f"\n  WARNING: could not write '{json_out}': {exc}")

    # --- Write the optional DP diagnostics CSV (only when it was active) -----
    if diag_out_path is not None:
        rows = [r for c in clients for r in c.dp_diag_rows]
        # Group by communication round (client order is preserved inside it).
        rows.sort(key=lambda r: (r.get('round') if r.get('round') is not None
                                 else -1, r.get('client_id') or 0))
        # Re-checked right before writing: an existing file is never overwritten.
        diag_out_path = unique_output_path(diag_out_path)
        try:
            with open(diag_out_path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(','.join(DP_DIAGNOSTICS_HEADER) + '\n')
                for row in rows:
                    fh.write(format_dp_diagnostics_row(row) + '\n')
            print(f"\n  DP diagnostics saved: {os.path.abspath(diag_out_path)} "
                  f"({len(rows)} rows, {len(DP_DIAGNOSTICS_HEADER)} columns)")
        except OSError as exc:
            print(f"\n  WARNING: could not write DP diagnostics "
                  f"'{diag_out_path}': {exc}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(mape) + 1), mape, 'b-', linewidth=2)
        plt.xlabel('Communication Round')
        plt.ylabel('MAPE (%)')
        plt.title('ASLA Baseline (single layer, no DP)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('asla_baseline.png', dpi=150)
        plt.show()

    return history


def main():
    parser = argparse.ArgumentParser(
        description="ASLA framework. Default: full experiment suite. "
                    "Use --asla-only for the isolated Stage 1 baseline.")
    parser.add_argument('--asla-only', action='store_true',
                        help="run only the single-layer ASLA baseline "
                             "(no layer sweep, no quantization sweep, no FedAvg)")
    parser.add_argument('--rounds', type=int, default=100,
                        help="communication rounds for --asla-only (default: 100)")
    parser.add_argument('--seed', type=int, default=42,
                    help="random seed for reproducible experiments (default: 42)")
    parser.add_argument('--agg-layer', type=int, default=1,
                        help="layer index to aggregate (default: 1 = Dense(50))")
    parser.add_argument('--quant-bits', type=int, default=32, choices=[8, 16, 32],
                        help="quantization bits (default: 32 = none)")
    parser.add_argument('--clients', type=int, default=10,
                        help="maximum number of clients (default: 10)")
    parser.add_argument('--data-folder', default='.',
                        help="folder holding the client CSV files (default: .)")
    parser.add_argument('--plot', action='store_true',
                        help="save/show plots (off by default: plt.show() blocks)")
    parser.add_argument('--out', default='baseline_asla_results.txt',
                        help="file to write the baseline results to "
                             "(default: baseline_asla_results.txt)")
    parser.add_argument('--dp-method', default='none',
                        choices=['none', 'gaussian', 'clip_only'],
                        help="client-side DP method (default: none = disabled); "
                             "'clip_only' is a noise-free ablation that clips "
                             "the delta but gives NO privacy guarantee")
    parser.add_argument('--dp-clip', type=float, default=0.0,
                        help="client-side DP: global L2 clipping bound C on the "
                             "delta's flattened vector (0 = DP disabled)")
    parser.add_argument('--dp-sigma', type=float, default=0.0,
                        help="client-side DP: Gaussian noise multiplier, added "
                             "noise std = dp_sigma * clip (0 = DP disabled)")
    parser.add_argument('--dp-diagnostics', action='store_true',
                        help="record the scalar update norms (raw/clipped/"
                             "noise/noisy L2 + clip factor) of every client "
                             "update into a CSV; needs --dp-method gaussian "
                             "with --dp-clip and --dp-sigma > 0, or "
                             "--dp-method clip_only with --dp-clip > 0 (its "
                             "noise columns are recorded as 0.0); default: "
                             "off, so a normal run is completely unchanged)")
    parser.add_argument('--dp-diagnostics-out', default='dp_diagnostics.csv',
                        help="CSV file for --dp-diagnostics (default: "
                             "dp_diagnostics.csv; a _1/_2/... name is chosen "
                             "instead of overwriting an existing file)")
    parser.add_argument('--json-out', default='',
                        help="optional path for a machine-readable JSON record "
                             "of the run (status, config, data interval, "
                             "per-round history, per-client metrics); "
                             "default: no JSON written")
    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    if args.asla_only:
        run_asla_baseline(data_folder=args.data_folder,
                          num_rounds=args.rounds,
                          agg_layer=args.agg_layer,
                          quant_bits=args.quant_bits,
                          num_clients=args.clients,
                          plot=args.plot,
                          out_file=args.out,
                          dp_clip=args.dp_clip,
                          dp_sigma=args.dp_sigma,
                          dp_method=args.dp_method,
                          dp_diagnostics=args.dp_diagnostics,
                          dp_diagnostics_out=args.dp_diagnostics_out,
                          json_out=args.json_out)
        return

    run_experiment(args.data_folder)
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("Visualizations saved: asla_vs_fedavg.png, quantization_comparison.png")
    print("="*60)


if __name__ == "__main__":
    main()