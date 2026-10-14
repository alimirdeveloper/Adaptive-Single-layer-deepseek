"""
ASLA PRIVACY-ATTACK EXPERIMENT - PHASE 1: structure + replay validation
============================================================================

Standalone study of how much private information the ASLA selected-layer
update leaks.  This file is ADDITIVE: it imports the stock implementation in
`asla_complete.py` READ-ONLY (it never edits it) and it never overwrites an
existing artifact (baseline_*.txt, dp_diag_*.txt, *.png).  Every file it
writes goes through `asla.unique_output_path` plus a protected-name guard.

PHASE 1 SCOPE (what this file does TODAY)
-----------------------------------------
Only the replay-validation mode is implemented.  It answers ONE question:

    Can an attacker who knows the public pipeline (architecture, features,
    scaler fit, optimizer, epochs, batch size, shuffle) and a CANDIDATE load
    series reproduce the selected-layer update a client actually transmitted
    - and how much of that reproduction depends on information the attacker
    should NOT have?

`--mode full` is deliberately NOT implemented: the 100-round,
four-configuration experiment must not run before this validation output has
been reviewed.

THREAT MODEL (honest-but-curious server; the stock code has no secure
aggregation, no masking and no shuffling - ASLAServer.do_round receives every
client vector and only afterwards averages them)
-----------------------------------------------------------------------
For every client i and round t the server observes W_i^t, the transmitted
selected-layer vector.  It sent the round reference G^t itself, so it can
compute the exact delta

    delta_obs = W_i^t - G^t

(quant_bits=32 is lossless: get_layer_weights returns the arrays unchanged).
It also knows the public code path, the 5 features, the StandardScaler fit on
the training split, Adam(1e-3), 1 epoch, batch size 300 and the seed policy.

TWO KNOWLEDGE SETS, kept strictly apart and never mixed:
  A "full state" - ORACLE / UPPER BOUND.  Additionally given the client's
      complete pre-round model state (all three layers), its optimizer slot
      variables and its per-fit seed.  That is everything EXCEPT the private
      series, so it is the correct adversary for a privacy study and it
      bounds what replay can achieve at all.
  B "realistic" - knows G^t, the public pipeline and the candidate series
      only.  The client's non-selected layers and its Adam slot state are not
      observable in any transmission, so they are guessed (the same public
      build seed, fresh optimizer).  Those guesses are exactly right at round
      1, because the client's hidden layers have not been updated yet and the
      initial weights are a deterministic public function of the published
      seed policy - so B is EXPECTED to be exact at round 1 and to degrade from
      round 2 on.  Variant B is exact only to the extent those guesses happen
      to be right; the validation mode MEASURES the resulting gap instead of
      assuming it away.

WHAT THIS FILE DOES NOT CLAIM
-----------------------------
* No differential-privacy claim, no epsilon/delta, no end-to-end accounting.
* A failed replay is NOT evidence of privacy.  It is evidence that the replay
  is inexact, and the report names the component responsible.
* A successful identity match is NOT proof that the whole series is
  recoverable.  Identification and reconstruction are reported separately.
* Clipping alone is not a privacy mechanism (it preserves direction exactly)
  and a reduced attack score is not a DP guarantee.

MEASURED FACTS THIS FILE RELIES ON (re-derived at run time, not assumed)
------------------------------------------------------------------------
* MODEL BUILD.  Keras 3's seedless initializers (`Dense` with the default
  GlorotUniform) do NOT draw from the TF global RNG.  Measured: two builds
  under the same `tf.random.set_seed` differed even INSIDE one process, and
  every process got its own initial weights - so the recorded pass, and with it
  the gate verdict, was a fresh sample on each run.  They draw from PYTHON's
  `random` module: `random.seed(s) + tf.random.set_seed(s)` reproduces the
  initial weights exactly (identical digest in separate processes).
  `seed_for_build()` applies both and deliberately does NOT reseed NumPy.
* FIT.  `train_local` -> `model.fit(..., shuffle=True)` draws its batch
  permutation from the global TF RNG.  With `tf.random.set_seed(s)`
  immediately before the fit the result is determined by (starting weights,
  optimizer state, s) alone: measured, two fresh models with identical initial
  weights produced identical weights under the same TF seed while the Python
  and NumPy states differed, and `fit` consumed neither of those streams.  The
  shim therefore sets a public per-(client, round) TF seed before every fit and
  leaves NumPy untouched, so the client-side DP noise keeps the stock sequence
  and the Gaussian arm's residual is genuine injected noise, not an artefact.
* `model.set_weights(pre_round_state)` plus `optimizer.build(...)`/`assign`
  makes a replay of the SAME round bit-exact (rel-L2 = 0.0); without the
  optimizer slots the residual is ~28% of ||delta|| and without the hidden
  layers ~35%.
* The build determinism is re-derived at run time by
  `verify_build_determinism()`; the gate ABORTS if it fails, because a
  non-reproducible build makes every number below - and the verdict - a single
  unreproducible sample.

USAGE
-----
  python asla_privacy_attack.py                       # Phase 1 validation
  python asla_privacy_attack.py --rounds 2 --configs none --victims 1
  python asla_privacy_attack.py --mode full           # refuses (by design)
"""

# ---------------------------------------------------------------------------
# PART 0: DETERMINISM SHIM - must run BEFORE tensorflow is imported.
# ---------------------------------------------------------------------------
# TF reads several of these at import / op-registration time, so setting them
# after `import tensorflow` would have no effect.  `asla_complete` (which
# imports tensorflow itself) is imported further down, so this block is the
# earliest point available in this file.
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Bit-level reproducibility of the replay is the entire point of Phase 1.
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('PYTHONHASHSEED', '0')
# One thread in each pool: floating-point reductions then run in a fixed
# order.  configure_determinism() applies the same values through the public
# API as well.
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
# This script never plots, so never let a GUI backend be selected.
os.environ.setdefault('MPLBACKEND', 'Agg')
# TF_ENABLE_ONEDNN_OPTS is deliberately NOT touched: the stock baseline files
# were produced with oneDNN enabled, and the stock numbers in this study were
# measured to be unaffected by it.  Toggling it was also measured to NOT change
# cross-process reproducibility - it is not the source of the run-to-run
# difference that the shim below fixes (that was the unseeded Keras
# initializer; see seed_for_build()).

import argparse
import hashlib
import random
import sys

import numpy as np
import tensorflow as tf

# The stock implementation is imported READ-ONLY.  Importing the module runs
# only np.random.seed(42) / tf.random.set_seed(42) / a warnings filter, and
# main() is guarded by `if __name__ == "__main__"`, so nothing else executes.
import asla_complete as asla

# The stock module prints non-ASCII status characters (a check mark, and
# Persian/Arabic comments in its banners).  A Windows console, or any piped
# stdout, defaults to a legacy code page and would raise UnicodeEncodeError
# inside the stock loader before this study can even start.  Forcing UTF-8 on
# our own streams fixes that without touching the stock module; errors are
# replaced rather than raised so a stray glyph can never abort a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):      # pragma: no cover - old Python
        pass
del _stream

SCRIPT_NAME = 'asla_privacy_attack.py'
PHASE = 'Phase 1: structure + replay validation only'

# ---------------------------------------------------------------------------
# PART 1: THE FOUR CONFIGURATIONS UNDER STUDY
# ---------------------------------------------------------------------------
# Names, clipping bound and noise multipliers are exactly the ones used by the
# existing 100-round baselines, so the arms stay comparable:
#   baseline_100r_none.txt, baseline_100r_clip_only_c2.txt,
#   baseline_100r_gaussian_c2_s0.001.txt, baseline_100r_gaussian_c2_s0.01.txt
# clip_only / gaussian are active only with clip > 0, and gaussian
# additionally needs sigma > 0 - the same rule as apply_dp_gaussian and the
# same rule as the stock console banner / report.
#   (name, dp_method, dp_clip, dp_sigma)
DP_CONFIGS = (
    ('none',               'none',      0.0, 0.0),
    ('clip_only_c2',       'clip_only', 2.0, 0.0),
    ('gaussian_c2_s0.001', 'gaussian',  2.0, 0.001),
    ('gaussian_c2_s0.01',  'gaussian',  2.0, 0.01),
)

CONFIG_NAMES = tuple(c[0] for c in DP_CONFIGS)

def get_config(name):
    """Look up one configuration by name (case-insensitive)."""
    for cfg in DP_CONFIGS:
        if cfg[0].lower() == str(name).strip().lower():
            return cfg
    raise ValueError("unknown config '" + str(name) + "'; known: "
                     + ", ".join(CONFIG_NAMES))

def config_is_active(dp_method, dp_clip, dp_sigma):
    """True when the client-side mechanism actually alters the update.

    Mirrors asla.dp_diagnostics_active: 'clip_only' needs clip > 0, 'gaussian'
    needs clip > 0 AND sigma > 0.  Used only for labelling and for deciding
    whether a residual is expected to be non-zero.
    """
    if dp_method == 'clip_only':
        return dp_clip > 0.0
    if dp_method == 'gaussian':
        return dp_clip > 0.0 and dp_sigma > 0.0
    return False

# Knowledge levels, in decreasing order of what the attacker is given.
# 'A_full_state' is the oracle upper bound; 'B_realistic' is the adversary
# that only sees public information.  The two ablation levels isolate WHICH
# piece of hidden state a mismatch comes from, so a failure can be diagnosed
# instead of merely reported.
KNOWLEDGE_LEVELS = ('A_full_state', 'B_realistic',
                    'ablate_weights_only', 'ablate_optimizer_only')

def knowledge_grants(knowledge):
    """(restore_hidden_weights, restore_optimizer_state) for a knowledge level.

    A: both.  B: neither (public init + fresh optimizer).
    ablate_weights_only: hidden layers yes, optimizer no.
    ablate_optimizer_only: hidden layers no, optimizer yes.
    """
    table = {
        'A_full_state': (True, True),
        'B_realistic': (False, False),
        'ablate_weights_only': (True, False),
        'ablate_optimizer_only': (False, True),
    }
    if knowledge not in table:
        raise ValueError("unknown knowledge level '" + str(knowledge) + "'")
    return table[knowledge]

# ---------------------------------------------------------------------------
# PART 2: OUTPUT SAFETY
# ---------------------------------------------------------------------------
# This study is additive.  These prefixes belong to earlier stages of the
# project and must never be written by this file.
PROTECTED_PREFIXES = ('baseline_', 'dp_diag', 'dp_diagnostics',
                      'asla_vs_fedavg', 'quantization_', 'layer_comparison',
                      'fedavg_', 'client_data_distribution', 'asla_baseline')

def safe_output_path(path):
    """Refuse to write a protected artifact, else pick a free name.

    Reuses the stock `unique_output_path` helper, so an existing file is never
    reused silently.  The guard is a second line of defence: this file must
    never be able to overwrite the Stage 1 / DP-baseline artifacts.
    """
    base = os.path.basename(path)
    for prefix in PROTECTED_PREFIXES:
        if base.startswith(prefix):
            raise RuntimeError(
                "refusing to write '" + base + "': it belongs to an earlier "
                "stage of the project and must not be overwritten")
    return asla.unique_output_path(path)

# ---------------------------------------------------------------------------
# PART 3: DETERMINISM AND THE PUBLIC PER-FIT SEED
# ---------------------------------------------------------------------------

def configure_determinism(seed):
    """Apply the runtime half of the determinism shim (the env half is above).

    Returns a list of human-readable notes for the report, so what was
    actually applied is recorded rather than assumed.
    """
    notes = []
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        notes.append("thread pools: inter=1 intra=1")
    except Exception as exc:                      # pragma: no cover - env dependent
        notes.append("thread pools: NOT set (" + str(exc) + ")")
    try:
        tf.config.experimental.enable_op_determinism()
        notes.append("tf.config.experimental.enable_op_determinism(): ON")
    except Exception as exc:                      # pragma: no cover - version dependent
        notes.append("enable_op_determinism(): UNAVAILABLE (" + str(exc) + ")")
    # Python's `random` FIRST: Keras 3's initializers draw from it (measured),
    # so a TF-only seed leaves the initial weights different in every process.
    # Then the stock order - numpy, then tensorflow.  The NumPy seed here is the
    # stock `np.random.seed(args.seed)` of asla_complete.main(), applied at the
    # same point, so the client-side Gaussian DP noise stream stays stock.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    notes.append("random.seed(%d) + np.random.seed(%d) + tf.random.set_seed(%d)"
                 % (seed, seed, seed))
    return notes

def set_tf_seed(seed):
    """The per-fit shim: seed ONLY the TF generator, leave NumPy alone.

    Measured: with this immediately before `model.fit` the result is determined
    by (starting weights, optimizer state, seed) alone - two fresh models with
    identical initial weights agreed bit for bit under the same TF seed while
    the Python and NumPy states differed, and the fit consumed neither of those
    streams.  So this suffices for a fit, and leaving NumPy alone keeps the
    client-side DP noise on the stock sequence: the attacker cannot reproduce
    that noise, so a Gaussian residual stays a genuine DP effect.
    """
    tf.random.set_seed(int(seed))

def seed_for_build(seed):
    """Seed a MODEL BUILD: Python's `random` and TF - never NumPy.

    Measured fact this exists for: Keras 3's seedless initializers (`Dense` ->
    GlorotUniform(seed=None)) draw from PYTHON's `random` module, not from
    tf.random.  Seeding only the TF generator therefore left the initial weights
    different in every process (two builds under the same TF seed differed even
    inside one process), which made the whole recorded pass - and hence the gate
    verdict - a fresh sample on every run.  NumPy is deliberately not reseeded
    here: it carries the stock client-side DP noise stream.
    """
    random.seed(int(seed))
    tf.random.set_seed(int(seed))

def weight_digest(model):
    """Short fingerprint of a model's weights, for the determinism self-check."""
    flat = np.concatenate([np.asarray(w).ravel() for w in model.get_weights()])
    return hashlib.sha256(flat.tobytes()).hexdigest()[:16]

def verify_build_determinism(build_seed):
    """Build the stock model twice under the same seed and compare.

    Returns (ok, digest).  This turns the measurement behind `seed_for_build()`
    into a permanent guard: if a future Keras/TF changes which generator the
    initializers draw from, the run says so instead of silently producing
    numbers that cannot be reproduced.
    """
    digests = []
    for _ in range(2):
        seed_for_build(build_seed)
        model = asla.create_model()          # exactly what the pass builds
        model.build((None, 5))
        digests.append(weight_digest(model))
    seed_for_build(build_seed)               # leave the RNGs in a known state
    return digests[0] == digests[1], digests[0]

def derive_seed(base_seed, client_id, round_index):
    """Public, reproducible per-(client, round) fit seed.

    Deliberately a pure function of information the server ALREADY holds (the
    run seed plus the public client index and round number), so the attacker
    can recompute it exactly.  It replaces the implicit, unobservable shuffle
    seed Keras would otherwise draw from the global RNG.

    DOCUMENTED DEVIATION: epochs, batch size, optimizer, data, loss and
    `shuffle=True` are untouched, but the batch ORDER differs from an
    unshimmed run, so this script's MAPE is NOT bit-comparable to the numbers
    in baseline_100r_*.txt.  `--no-seed-shim` disables this and demonstrates
    empirically that the unshimmed replay is not reproducible.
    """
    return (int(base_seed) * 1000003 + int(client_id) * 1009
            + int(round_index)) % (2 ** 31 - 1)

# ---------------------------------------------------------------------------
# PART 4: SMALL ARRAY HELPERS (shape-agnostic, kernel+bias flattened)
# ---------------------------------------------------------------------------

def flatten(arrays):
    """Flatten a list of arrays (kernel, bias, ...) into one 1-D vector."""
    return np.concatenate([np.asarray(a, dtype=float).reshape(-1)
                           for a in arrays])

def l2_norm(arrays):
    """Euclidean norm of the flattened parameter vector."""
    return float(np.linalg.norm(flatten(arrays)))

def delta_of(a, b):
    """a - b, element-wise, as a new list of arrays."""
    return [np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
            for x, y in zip(a, b)]

def cosine(a, b):
    """Cosine similarity of two flattened parameter vectors (0.0 if degenerate)."""
    va, vb = flatten(a), flatten(b)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def rel_l2(obs, pred):
    """||obs - pred|| / ||obs|| (0.0 when obs is exactly reproduced)."""
    den = l2_norm(obs)
    if den == 0.0:
        return float('nan')
    return l2_norm(delta_of(obs, pred)) / den

def arrays_equal(a, b):
    """Exact element-wise equality of two parameter lists."""
    if len(a) != len(b):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(a, b))

def copy_arrays(arrays):
    """Independent deep copy of a parameter list (never shares buffers)."""
    return [np.array(a, copy=True) for a in arrays]

# ---------------------------------------------------------------------------
# PART 5: THE RECORDING CLIENT (observation only - never changes the update)
# ---------------------------------------------------------------------------

class RecordingClient(asla.FederatedClient):
    """A stock FederatedClient that additionally OBSERVES what it transmits.

    Contract, deliberately narrow:
      * every override calls `super()` exactly once with the arguments it was
        given and returns the stock result unchanged;
      * `get_layer_weights` records AFTER the stock DP + quantization path has
        produced the value, so the recorded vector is literally the vector the
        server receives (and at quant_bits=32 that path is lossless);
      * the pre-round state is captured BEFORE `super().set_layer_weights`
        mutates the selected layer;
      * the only non-stock side effect is the per-fit TF seed in `train_local`
        when a seed is supplied (the documented reproducibility shim);
      * every copy uses np.array(..., copy=True), so a recorded array can never
        be mutated later by an in-place update.
    """

    def __init__(self, *args, **kwargs):
        # Shim controls (consumed here, never forwarded to the stock __init__).
        self.base_seed = kwargs.pop('base_seed', 42)
        self.seed_shim = kwargs.pop('seed_shim', True)
        self.fit_seed_override = kwargs.pop('fit_seed_override', None)
        self.fit_seed = None
        self.records = []
        self._round = 0
        self._pending = None
        super().__init__(*args, **kwargs)

    # -- observation points ------------------------------------------------
    def train_local(self, epochs=1, batch_size=300):
        """Stock training, preceded by the public per-(client, round) TF seed.

        The round index is finalised HERE, not in set_layer_weights:
        ASLAServer.do_round calls set_layer_weights once more per round for its
        evaluation sweep, so counting setter calls would count evaluation
        passes as training rounds and mislabel every recorded round.  Counting
        train_local calls counts exactly the rounds this client trained in.
        """
        self._round += 1
        if self.fit_seed_override is not None:
            self.fit_seed = (int(self.fit_seed_override)
                             if self.seed_shim else None)
        elif self.seed_shim:
            self.fit_seed = derive_seed(self.base_seed, self.client_id,
                                        self._round)
        else:
            self.fit_seed = None
        if self._pending is not None:
            self._pending['round'] = self._round
            self._pending['fit_seed'] = self.fit_seed
        if self.seed_shim and self.fit_seed is not None:
            set_tf_seed(self.fit_seed)
        return super().train_local(epochs, batch_size)

    def set_layer_weights(self, layer_idx, weights):
        """Capture the full pre-round state, then run the stock setter.

        The capture MUST happen first: super().set_layer_weights overwrites
        model.layers[layer_idx], which is exactly the state needed before it is
        touched.
        """
        self._capture_pre_state(layer_idx, weights)
        super().set_layer_weights(layer_idx, weights)

    def get_layer_weights(self, layer_idx):
        """Run the stock DP + quantization path, then record the result."""
        w = super().get_layer_weights(layer_idx)
        self._record_transmission(layer_idx, w)
        return w

    # -- internals ---------------------------------------------------------
    def _capture_pre_state(self, layer_idx, incoming_weights):
        """Snapshot everything the round is about to start from.

        The round number and the fit seed are NOT finalised here - train_local
        fills them in, because only a real training call marks a round.
        """
        stock_round = self.dp_diag_round
        self._pending = {
            'round': None,                 # finalised by train_local
            # Cross-check of our own counter against the stock tag (recorded,
            # never trusted): do_round assigns dp_diag_round unconditionally,
            # but only AFTER this setter has already run, so at capture time it
            # still holds the previous round.
            'stock_round': (int(stock_round)
                            if isinstance(stock_round, int) else None),
            'client_id': self.client_id,
            'name': self.name,
            'layer': layer_idx,
            'quant_bits': self.quant_bits,
            'dp_method': self.dp_method,
            'dp_clip': self.dp_clip,
            'dp_sigma': self.dp_sigma,
            'fit_seed': None,              # finalised by train_local
            # What the server sent this round (the reference G).
            'G': copy_arrays(incoming_weights),
            # ORACLE-ONLY state: all three layers, before the setter ran.
            'pre_weights': copy_arrays(self.model.get_weights()),
            # ORACLE-ONLY state: Adam slots (empty list before the first fit).
            'pre_optimizer': copy_arrays(self.model.optimizer.variables),
        }

    def _record_transmission(self, layer_idx, w):
        """Attach the transmitted vector to the snapshot taken this round."""
        pending = self._pending
        if pending is None or pending.get('layer') != layer_idx:
            return                      # nothing captured: record nothing
        record = dict(pending)
        record['sent'] = copy_arrays(w)
        record['diag'] = self._matching_diag_row(layer_idx, record['round'])
        self.records.append(record)
        self._pending = None

    def _matching_diag_row(self, layer_idx, round_index):
        """The stock DP diagnostics row for this very update, if any.

        Read-only: the row is copied, never removed, so the stock diagnostics
        sink stays intact for anyone else who reads it.
        """
        for row in reversed(self.dp_diag_rows):
            if (row.get('layer_index') == layer_idx
                    and row.get('round') == round_index):
                return dict(row)

# ---------------------------------------------------------------------------
# PART 6: THE REPLAY
# ---------------------------------------------------------------------------

def restore_optimizer_state(model, saved_variables):
    """Load recorded Adam slots into a FRESH optimizer (best effort).

    A fresh Adam owns no variables until it is built, so `saved_variables` is
    empty before the first fit: there is simply nothing to restore at round 1,
    which is itself a finding (variant B can only be exact at round 1).  The
    slots cannot be assigned into an unbuilt optimizer, so it is built from the
    model's trainable variables first, then assigned.

    Returns a short status string that the report records verbatim.
    """
    if not saved_variables:
        return 'empty(nothing-to-restore)'
    optimizer = model.optimizer
    if len(optimizer.variables) != len(saved_variables):
        try:
            optimizer.build(model.trainable_variables)
        except TypeError:                     # pragma: no cover - API drift
            optimizer.build([v.path for v in model.trainable_variables])
    if len(optimizer.variables) != len(saved_variables):
        return 'FAILED(count %d != %d)' % (len(optimizer.variables),
                                           len(saved_variables))
    try:
        for var, value in zip(optimizer.variables, saved_variables):
            var.assign(np.asarray(value))
    except Exception as exc:                  # pragma: no cover - API drift
        return 'FAILED(%s)' % exc
    return 'restored(%d vars)' % len(saved_variables)

def predict_transmitted_delta(delta_raw, dp_method, dp_clip, dp_sigma):
    """The attacker's prediction of the DELTA the server receives.

    For an inactive configuration the transmitted delta is the raw local
    delta.  For 'clip_only' and 'gaussian' it is the deterministically clipped
    delta, computed with the stock function itself so the prediction cannot
    drift from the implementation.

    For 'gaussian' this is deliberately the NOISE-BLIND prediction: the noise
    is drawn from the client's private NumPy stream (the shim above leaves that
    stream alone precisely so it stays private), so a residual on the Gaussian
    arms is the DP noise itself.  That is the honest attacker: one that cannot
    reproduce the noise.
    """
    if not config_is_active(dp_method, dp_clip, dp_sigma):
        return copy_arrays(delta_raw)
    return asla.apply_dp_clip_only(copy_arrays(delta_raw), dp_clip)

def replay_update(record, data, layer_idx, knowledge, seed_shim=True,
                  build_seed=7, epochs=1, batch_size=300):
    """Replay ONE recorded client/round update and predict what it transmitted.

    `data` is the candidate series already preprocessed the same public way,
    as (X_train, y_train, X_test, y_test).  Returns

        (delta_raw, delta_pred, info)

    where delta_raw is the replayed LOCAL delta (w_local - G), delta_pred is
    the prediction of the transmitted delta for this record's configuration,
    and info carries the restore statuses and the replay's own training loss.

    The replay client is always built with dp_method='none' and
    quant_bits=32: the DP/quantization step is then applied by the PREDICTOR
    above, so the replay never consumes the client's private noise stream and
    a residual cannot be mis-attributed to the replay itself.
    """
    restore_weights, restore_optimizer = knowledge_grants(knowledge)
    X_train, y_train, X_test, y_test = data
    layer = record.get('layer', layer_idx)

    # Seeded construction: the attacker can rebuild the same untrained model.
    # seed_for_build also seeds Python's random, which is where Keras 3 draws
    # the initial weights from - tf.random.set_seed alone would not reproduce
    # the attacker's own model.
    seed_for_build(build_seed)
    client = RecordingClient(
        record['client_id'], X_train, y_train, X_test, y_test,
        name=record.get('name', ''), quant_bits=32,
        dp_clip=0.0, dp_sigma=0.0, dp_method='none', dp_diagnostics=False,
        seed_shim=seed_shim, fit_seed_override=record.get('fit_seed'))
    client.set_layer_weights(layer, record['G'])

    info = {'knowledge': knowledge, 'weights': 'public-init',
            'optimizer': 'fresh'}
    if restore_weights:
        client.model.set_weights(record['pre_weights'])
        # set_weights rewrites the selected layer too; re-assert the reference
        # so the replay starts from exactly (hidden state, G).
        asla.set_layer_weights(client.model, layer, record['G'])
        info['weights'] = 'restored(all layers)'
    if restore_optimizer:
        info['optimizer'] = restore_optimizer_state(client.model,
                                                    record['pre_optimizer'])

    loss = client.train_local(epochs, batch_size)
    info['loss'] = loss
    info['seeded'] = bool(seed_shim and client.fit_seed is not None)
    info['fit_seed'] = client.fit_seed

    delta_raw = delta_of(client.get_layer_weights(layer), record['G'])
    delta_pred = predict_transmitted_delta(delta_raw, record['dp_method'],
                                           record['dp_clip'],
                                           record['dp_sigma'])
    return delta_raw, delta_pred, info

# ---------------------------------------------------------------------------
# PART 7: METRICS AND THE SEPARATION STATISTICS
# ---------------------------------------------------------------------------

def observed_delta(record):
    """delta_obs = transmitted vector - round reference G.

    Both terms are exactly what the server sees: it chose G itself and it
    received the transmitted vector.  At quant_bits=32 the stock path is
    lossless, so this is the true local delta.
    """
    return delta_of(record['sent'], record['G'])

def compare(obs, pred):
    """The headline numbers of one replay.

    rel_l2 is the relative residual ||obs - pred|| / ||obs|| (0.0 = bit-exact
    reproduction, 1.0 = the prediction is no better than predicting nothing);
    cosine is the direction agreement, which is what a ranking attack needs.
    """
    return {'rel_l2': rel_l2(obs, pred),
            'cosine': cosine(obs, pred),
            'bit_equal': arrays_equal(obs, pred),
            'obs_l2': l2_norm(obs),
            'pred_l2': l2_norm(pred)}

def pairwise_cosines(deltas):
    """Cosines between every unordered pair of deltas (same round, all clients).

    This is the feasibility gate for any later ranking stage.  Two clients
    whose true deltas were nearly collinear could not be told apart by cosine
    similarity, so a high mean here would invalidate the identity experiment
    no matter how exact the replay is.  (A synthetic look-alike profile was
    measured at cos = 0.999935 - that is the trap this statistic exists to
    catch.  The real PJM zones are nearly orthogonal.)
    """
    vectors = [flatten(d) for d in deltas]
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            out.append(cosine(vectors[i], vectors[j]))
    return out

def describe(values, digits=4):
    """'n=.. mean=.. min=.. max=..' for a list of floats (empty-safe)."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return 'n=0'
    return ('n=%d mean=%.*f min=%.*f max=%.*f'
            % (len(clean), digits, float(np.mean(clean)),
               digits, float(np.min(clean)), digits, float(np.max(clean))))

def mean_of(values, default=float('nan')):
    """Mean of the non-NaN entries (default when there is nothing to average)."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return default
    return float(np.mean(clean))

def truthy_rate(flags):
    """Fraction of True entries (NaN when the list is empty)."""
    clean = [bool(f) for f in flags]
    if not clean:
        return float('nan')
    return float(sum(clean)) / len(clean)

# ---------------------------------------------------------------------------
# PART 8: THE VALIDATION RUNNER
# ---------------------------------------------------------------------------

def build_series_pool(client_data, client_names, max_clients):
    """Preprocess candidate series exactly as the stock code does.

    Returns (usable, series): `usable` is the list of column indices that pass
    the stock admission rule (a training split exists and is longer than 100),
    and `series[i]` is (X_train, y_train, X_test, y_test).

    Preprocessing is pure: it consumes no random numbers, so preparing extra
    series (for the decoy arm) cannot perturb the run in any way.
    """
    usable, series = [], {}
    for i in range(min(len(client_names), max_clients)):
        X_train, X_test, y_train, y_test, _ = asla.preprocess_client_data(
            client_data[:, i])
        if X_train is None or len(X_train) <= 100:
            continue
        usable.append(i)
        series[i] = (X_train, y_train, X_test, y_test)
    return usable, series

def run_config_pass(series, victim_indices, client_names, cfg, args, layer):
    """One stock ASLA run under one DP configuration, with recording.

    The client set, construction order, server, aggregation, evaluation and
    stopping rule are the stock ones.  The only additions are that each client
    is a RecordingClient (observe-only) and that each fit is preceded by its
    public per-(client, round) seed.
    """
    _, dp_method, dp_clip, dp_sigma = cfg
    clients = []
    for i in victim_indices:
        X_train, y_train, X_test, y_test = series[i]
        clients.append(RecordingClient(
            i, X_train, y_train, X_test, y_test, name=client_names[i],
            quant_bits=args.quant_bits, dp_clip=dp_clip, dp_sigma=dp_sigma,
            dp_method=dp_method,
            dp_diagnostics=config_is_active(dp_method, dp_clip, dp_sigma),
            base_seed=args.seed, seed_shim=not args.no_seed_shim))
    server = asla.ASLAServer(len(clients), agg_layer=layer,
                             quant_bits=args.quant_bits)
    history = server.train(clients, num_rounds=args.rounds)
    records = [rec for c in clients for rec in c.records]
    for rec in records:
        rec['config'] = cfg[0]
    records.sort(key=lambda r: (r['round'], r['client_id']))
    return clients, history, records

def pick_decoy(usable, victim_indices):
    """The decoy candidate: a real series that is NOT one of the victims.

    Chosen deterministically as the last usable series, so the control arm is
    reproducible.  Returns None when every usable series is already a victim.
    """
    candidates = [i for i in usable if i not in set(victim_indices)]
    return candidates[-1] if candidates else None

def summarize_separation(records, rounds):
    """Per-round pairwise cosine statistics of the TRUE observed deltas."""
    out = {}
    for r in rounds:
        deltas = [observed_delta(rec) for rec in records
                  if rec['round'] == r]
        if len(deltas) >= 2:
            out[r] = pairwise_cosines(deltas)
    return out

# --- the replay sweep -------------------------------------------------------
REPLAY_CSV_HEADER = ('config', 'round', 'client_id', 'client_name', 'knowledge',
                     'role', 'rel_l2', 'cosine', 'bit_equal', 'obs_l2',
                     'pred_l2', 'seeded', 'restore_weights',
                     'restore_optimizer', 'fit_seed')

def evaluate_replays(records, series, victim_ids, decoy_index, layer, args):
    """Replay every selected (round, client) cell at every knowledge level.

    Two arms per cell: 'true_series' (the victim's own data) and 'decoy_series'
    (a real but different series, given the same oracle state and the same
    seed).  The decoy arm is the control: if it scored as well as the true arm,
    the replay would be measuring the pipeline, not the series.
    """
    levels = ['A_full_state', 'B_realistic']
    if args.ablate:
        levels += ['ablate_weights_only', 'ablate_optimizer_only']
    rows = []
    for record in records:
        if (args.eval_rounds is not None
                and record['round'] not in args.eval_rounds):
            continue
        if victim_ids is not None and record['client_id'] not in victim_ids:
            continue
        obs = observed_delta(record)
        arms = [('true_series', series[record['client_id']])]
        if decoy_index is not None:
            arms.append(('decoy_series', series[decoy_index]))
        for knowledge in levels:
            for role, data in arms:
                _, pred, info = replay_update(
                    record, data, layer, knowledge,
                    seed_shim=not args.no_seed_shim,
                    build_seed=args.build_seed, epochs=1, batch_size=300)
                metrics = compare(obs, pred)
                rows.append({
                    'config': record['config'], 'round': record['round'],
                    'client_id': record['client_id'],
                    'client_name': record.get('name', ''), 'knowledge': knowledge,
                    'role': role, 'rel_l2': metrics['rel_l2'],
                    'cosine': metrics['cosine'],
                    'bit_equal': metrics['bit_equal'],
                    'obs_l2': metrics['obs_l2'], 'pred_l2': metrics['pred_l2'],
                    'seeded': info.get('seeded'),
                    'restore_weights': info.get('weights'),
                    'restore_optimizer': info.get('optimizer'),
                    'fit_seed': info.get('fit_seed'),
                    'dp_method': record['dp_method'],
                    'dp_clip': record['dp_clip'],
                    'dp_sigma': record['dp_sigma'],
                    'diag': record.get('diag'),
                })
    return rows

# --- the go / no-go gate ----------------------------------------------------
# How far residual/||noise|| may sit from 1.00 and still count as "the oracle
# residual IS the injected noise".  Measured values were 1.0027 and 0.9951; the
# tolerance only has to absorb the sampling spread of ||noise|| across the few
# recorded cells.  A broken replay lands orders of magnitude outside it.
NOISE_MATCH_TOL = 0.10

def evaluate_gate(rows, separation, diag_by_config=None):
    """Turn the measurements into a per-configuration go / no-go verdict.

    THREE independent conditions must hold before any ranking stage may run:
      1. the oracle replay must reproduce the update - otherwise the pipeline
         model itself is wrong and no ranking number could mean anything.
         This test is MECHANISM-AWARE, because "reproduce" means different
         things in the two cases:
           * noise-free arm (dp inactive or clipping only): EXACT - every cell
             bit-identical, or at worst identical to float rounding (the
             clipping arm reaches rel_L2 ~1e-8 with 0% of cells bit-identical,
             which still counts as exact).  A near-miss (a ~1% residual, cosine
             0.9999) means some hidden state
             is still unmodelled, and that must be diagnosed before any ranking
             number is trusted - the no-seed-shim demonstration produces exactly
             such a near-miss.
           * noise-injecting arm: bit-exactness is IMPOSSIBLE by construction -
             the attacker cannot observe the noise draw, and it is not
             reproducible even by an attacker holding the whole pre-round state
             (that is the point of the mechanism).  So the test becomes "is the
             oracle residual explained by the injected noise?", i.e.
             residual/||noise|| == 1.  That cross-check is what separates
             injected noise from a broken replay: an unmodelled hidden state
             would make the residual miss ||noise||, not match it.
      2. the true deltas of different clients must be separable - otherwise
         even a perfect replay could not tell two clients apart;
      3. the true series must OUTSCORE the decoy control - otherwise the cell
         carries no series-specific information at all.  This is the condition
         that catches round 1, where the local delta is dominated by the
         pre-round state and a wrong series reproduces it just as well.
    """
    diag_by_config = diag_by_config or {}
    verdicts = []
    for config in sorted({r['config'] for r in rows}):
        mine = [r for r in rows if r['config'] == config]
        a_true = [r for r in mine if r['knowledge'] == 'A_full_state'
                  and r['role'] == 'true_series']
        b_true = [r for r in mine if r['knowledge'] == 'B_realistic'
                  and r['role'] == 'true_series']
        b_decoy = [r for r in mine if r['knowledge'] == 'B_realistic'
                   and r['role'] == 'decoy_series']
        oracle_exact = truthy_rate([r['bit_equal'] for r in a_true])
        oracle_rel = mean_of([r['rel_l2'] for r in a_true])
        oracle_cos = mean_of([r['cosine'] for r in a_true])
        bit_exact = bool(oracle_exact >= 0.999 or oracle_rel <= 1e-6)
        # "Bit-exact" covers two different things: every cell identical, or
        # identical to float rounding.  The clipping arm lands on the latter
        # (rel_L2 ~1e-8, yet 0% of cells bit-identical), so the two must not be
        # reported with the same word or the report would overstate the replay.
        fully_exact = bool(oracle_exact >= 0.999)
        # Condition 1, mechanism-aware: see the docstring above.
        diag = diag_by_config.get(config) or {}
        noise_l2 = diag.get('noise_l2')
        resid_l2 = diag.get('residual_l2')
        noise_ratio = (resid_l2 / noise_l2
                       if noise_l2 and resid_l2 is not None and noise_l2 > 0.0
                       else None)
        noise_explained = bool(noise_ratio is not None
                               and abs(noise_ratio - 1.0) <= NOISE_MATCH_TOL)
        oracle_ok = bool(bit_exact or noise_explained)
        b_cos = mean_of([r['cosine'] for r in b_true])
        decoy_cos = mean_of([r['cosine'] for r in b_decoy])
        margin = b_cos - decoy_cos
        # Condition 3, per round.  The mean alone is NOT enough: with a
        # knife-edge margin the mean's sign is decided by the last decimal, and
        # a verdict that flips on that is worthless.  A round whose margin is
        # not positive is a round in which the decoy series reproduces the
        # transmitted update exactly as well as the true one, i.e. a round that
        # carries no series-specific information at all - so such a round must
        # block the ranking stage, however favourable the other rounds look.
        round_margins = {}
        for index in sorted({r['round'] for r in b_true}):
            true_cell = [r['cosine'] for r in b_true if r['round'] == index]
            decoy_cell = [r['cosine'] for r in b_decoy if r['round'] == index]
            if true_cell and decoy_cell:
                round_margins[index] = mean_of(true_cell) - mean_of(decoy_cell)
        losing = sorted(i for i, m in round_margins.items() if not m > 0.0)
        margin_ok = bool(margin > 0.0 and not losing)
        # The ORACLE-level decoy margin is the SIGNAL CEILING: the same
        # comparison with the pre-round state known, i.e. how much series
        # information the update carries at all.  It separates two very
        # different ABORT causes that would otherwise look identical:
        #   * ceiling ~ 0 too -> the series carries no recoverable information;
        #   * ceiling large    -> the information IS there, but the unmodelled
        #     pre-round state buries it at the attacker's knowledge level.
        a_decoy = [r for r in mine if r['knowledge'] == 'A_full_state'
                   and r['role'] == 'decoy_series']
        oracle_margin = mean_of([r['cosine'] for r in a_true]) - mean_of(
            [r['cosine'] for r in a_decoy])
        oracle_round_margins = {}
        for index in sorted({r['round'] for r in a_true}):
            true_cell = [r['cosine'] for r in a_true if r['round'] == index]
            decoy_cell = [r['cosine'] for r in a_decoy if r['round'] == index]
            if true_cell and decoy_cell:
                oracle_round_margins[index] = (mean_of(true_cell)
                                               - mean_of(decoy_cell))
        # "Buried" = a real ceiling exists but the threat-model margin keeps
        # less than a tenth of it.
        buried = bool(oracle_margin > 0.0 and margin < 0.1 * oracle_margin)
        sep = mean_of([c for values in separation.values() for c in values])
        separated = bool(not np.isnan(sep) and sep < 0.5)
        diagnosis = ''
        if not oracle_ok:
            for level in ('ablate_weights_only', 'ablate_optimizer_only'):
                cell = [r for r in mine if r['knowledge'] == level
                        and r['role'] == 'true_series']
                if cell:
                    diagnosis += ('%s cos=%.4f; ' % (level, mean_of(
                        [r['cosine'] for r in cell])))
        per_round = ' '.join('%d:%+.4f' % (i, m)
                             for i, m in sorted(round_margins.items()))
        if not oracle_ok:
            why = ('residual/noise %.4f, which is not ~1.00 for a '
                   'noise-injecting arm' % noise_ratio
                   if noise_ratio is not None else
                   'this arm injects no noise, so the replay must be bit-exact')
            verdict = ('ABORT: the oracle replay does not reproduce the update '
                       '(bit-identical in %.0f%% of cells, mean rel_L2 %.6f; '
                       '%s) - ranking must '
                       'not run. ' % (100.0 * oracle_exact, oracle_rel, why)
                       + diagnosis)
        elif not separated:
            verdict = ('ABORT: client deltas are not separable (mean pairwise '
                       'cosine %.4f) - ranking would be uninformative.' % sep)
        elif not margin_ok:
            verdict = ('ABORT: the true series does not outscore the decoy '
                       'control in every round (threat-model margin %s; mean '
                       '%+.4f) - round(s) %s carry no series-specific '
                       'information, so a ranking would be meaningless there.  '
                       'The decoy is another real client, so beating it requires '
                       'the series to drive the update more than the '
                       'unobservable pre-round state does; round 1 is the '
                       'classic case, where that state dominates the local '
                       'delta.'
                       % (per_round, margin,
                          ', '.join(str(i) for i in losing) if losing else 'all'))
            if buried:
                verdict += ('  DIAGNOSIS: the signal itself exists - with the '
                            'pre-round state known (oracle arm) the true series '
                            'beats the decoy by %+.4f (mean), %s per round - but '
                            'at this knowledge level the unmodelled state gap '
                            'buries it (%.1f%% of the ceiling survives).  So a '
                            'ranking stage needs a stronger knowledge model, '
                            'not a better candidate list.'
                            % (oracle_margin,
                               ' '.join('%d:%+.4f' % (i, m) for i, m in sorted(
                                   oracle_round_margins.items())),
                               100.0 * margin / oracle_margin))
            else:
                verdict += ('  DIAGNOSIS: the oracle-level margin is %+.4f too, '
                            'so the update carries no recoverable series '
                            'information at all.' % oracle_margin)
        else:
            if fully_exact:
                basis = 'bit-exact in %.0f%% of cells' % (100.0 * oracle_exact)
            elif bit_exact:
                basis = ('exact to float rounding (mean rel_L2 %.2e; only %.0f%%'
                         ' of cells bit-identical)'
                         % (oracle_rel, 100.0 * oracle_exact))
            else:
                basis = ('residual/noise %.4f, i.e. the residual IS the '
                         'injected noise' % noise_ratio)
            verdict = ('PROCEED: replay validated (%s) and the true series '
                       'outscores the decoy control in every round (per-round '
                       'margin %s; mean %+.4f)' % (basis, per_round, margin))
            if oracle_margin > 0.0:
                verdict += ('  Signal ceiling (the same comparison with the '
                            'pre-round state known, i.e. what the update carries '
                            'at all): %+.4f mean, %s per round; this knowledge '
                            'level keeps %.1f%% of it.'
                            % (oracle_margin,
                               ' '.join('%d:%+.4f' % (i, m) for i, m in sorted(
                                   oracle_round_margins.items())),
                               100.0 * margin / oracle_margin))
        verdicts.append({
            'config': config, 'oracle_bit_exact_rate': oracle_exact,
            'oracle_mean_rel_l2': oracle_rel,
            'oracle_mean_cosine': oracle_cos, 'oracle_ok': oracle_ok,
            'oracle_bit_exact': bit_exact,
            'oracle_fully_exact': fully_exact,
            'oracle_noise_ratio': noise_ratio,
            'oracle_noise_explained': noise_explained,
            'oracle_noise_l2': noise_l2, 'oracle_residual_l2': resid_l2,
            'oracle_margin': oracle_margin,
            'oracle_round_margins': oracle_round_margins,
            'signal_buried': buried,
            'realistic_mean_cosine': b_cos, 'decoy_mean_cosine': decoy_cos,
            'realistic_margin': margin, 'margin_ok': margin_ok,
            'round_margins': round_margins, 'losing_rounds': losing,
            'separation_mean_cosine': sep, 'separated': separated,
            'verdict': verdict,
        })
    return verdicts

# --- identification (a separate claim from reconstruction) ------------------
IDENTITY_CSV_HEADER = ('config', 'round', 'victim_id', 'victim_name', 'knowledge',
                       'top1_correct', 'true_rank', 'candidates',
                       'best_candidate_id', 'best_cosine', 'true_cosine',
                       'margin', 'chance', 'shuffled_top1_correct')

def shuffled_identity_map(candidate_ids, seed):
    """A fixed label permutation, for the shuffled-label chance control.

    The score matrix is left untouched; only the mapping "candidate -> the
    client it is claimed to be" is permuted.  A ranking with no real signal
    must then score about chance (1/K), which is what makes the control
    informative rather than decorative.
    """
    order = list(candidate_ids)
    rng = np.random.RandomState(int(seed) + 20260101)
    permuted = list(order)
    rng.shuffle(permuted)
    return dict(zip(order, permuted))

def run_identity(records, series, candidate_ids, layer, knowledge, args):
    """K x K cosine ranking of candidate series against each observed update.

    Deliberately reported as a SEPARATE claim: knowing WHICH series produced an
    update is weaker and different from reproducing the update itself.
    """
    shuffled = shuffled_identity_map(candidate_ids, args.seed)
    chance = 1.0 / len(candidate_ids) if candidate_ids else float('nan')
    rows = []
    for record in records:
        if (args.eval_rounds is not None
                and record['round'] not in args.eval_rounds):
            continue
        if args.victim_ids is not None and record['client_id'] not in args.victim_ids:
            continue
        obs = observed_delta(record)
        scores = []
        for cid in candidate_ids:
            _, pred, _ = replay_update(record, series[cid], layer, knowledge,
                                       seed_shim=not args.no_seed_shim,
                                       build_seed=args.build_seed)
            scores.append((cosine(obs, pred), cid))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        best_cosine, best_candidate = scores[0]
        true_cosine = next(c for c, cid in scores
                           if cid == record['client_id'])
        true_rank = 1 + sum(1 for c, _ in scores if c > true_cosine)
        second_best = next((c for c, cid in scores
                            if cid != record['client_id']), float('nan'))
        rows.append({
            'config': record['config'], 'round': record['round'],
            'victim_id': record['client_id'],
            'victim_name': record.get('name', ''), 'knowledge': knowledge,
            'top1_correct': bool(best_candidate == record['client_id']),
            'true_rank': true_rank, 'candidates': len(candidate_ids),
            'best_candidate_id': best_candidate,
            'best_cosine': best_cosine, 'true_cosine': true_cosine,
            'margin': true_cosine - second_best,
            'chance': chance,
            'shuffled_top1_correct': bool(
                best_candidate == shuffled[record['client_id']]),
        })
    return rows

# ---------------------------------------------------------------------------
# PART 9: OUTPUT (text report + CSV)
# ---------------------------------------------------------------------------

def csv_cell(value):
    """Render one CSV cell (no quoting needed: no field can contain a comma).

    NumPy scalars are normalised explicitly: np.float64 does subclass float,
    but NumPy 2's repr() prints 'np.float64(0.5)', which would land verbatim in
    the CSV and break any consumer that parses the column as a number.
    """
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return 'True' if bool(value) else 'False'
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)

def write_csv(path, header, rows):
    """Write a small CSV through the safe-path guard.  Returns the path used."""
    target = safe_output_path(path)
    with open(target, 'w', encoding='utf-8', newline='') as handle:
        handle.write(','.join(header) + '\n')
        for row in rows:
            handle.write(','.join(csv_cell(row.get(key))
                                  for key in header) + '\n')
    return target

def write_text(path, lines):
    """Write the text report through the safe-path guard.  Returns the path."""
    target = safe_output_path(path)
    with open(target, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return target

# ---------------------------------------------------------------------------
# PART 10: THE TEXT REPORT
# ---------------------------------------------------------------------------

def _header_lines(results):
    """Run identity, knobs and the exact determinism settings in effect."""
    args = results['args']
    shim = 'OFF (--no-seed-shim: the replay is expected NOT to reproduce)' \
        if args.no_seed_shim else \
        'ON (public per-(client, round) tf.random.set_seed before each fit)'
    lines = [
        '=' * 72,
        'ASLA PRIVACY-ATTACK STUDY - PHASE 1: REPLAY VALIDATION',
        '=' * 72,
        'phase            : ' + PHASE,
        'script           : ' + SCRIPT_NAME,
        'stock module     : asla_complete.py (imported read-only, untouched)',
        'attack mode      : replay validation only'
        + (' + identity ranking (--identity)' if args.identity else ''),
        '--mode full      : NOT IMPLEMENTED - it refuses and exits non-zero',
        'run seed         : ' + str(args.seed),
        'model build seed : ' + str(args.build_seed),
        'build check      : ' + (
            ('OK (initial weights digest %s, identical in two builds)'
             % results['build_check'][1]) if results.get('build_check')
            else 'NOT RUN'),
        'seed shim        : ' + shim,
        'selected layer   : model.layers[' + str(results['layer']) + ']',
        'quant bits       : ' + str(args.quant_bits)
        + ('' if args.quant_bits == 32 else
           ' (lossy: both sides are compared in the quantized domain)'),
        'data folder      : ' + os.path.abspath(args.data_folder),
        'rounds requested : ' + str(args.rounds),
        'configs          : ' + ', '.join(results['config_order']),
        'clients in pass  : ' + str(len(results['victim_indices'])),
        'victims evaluated: ' + (', '.join(str(i) for i in results['victim_ids'])
                                 if results['victim_ids'] is not None
                                 else 'all clients of the pass'),
        'eval rounds      : ' + (', '.join(str(r) for r in sorted(
                                 results['eval_rounds']))
                                 if results['eval_rounds'] is not None
                                 else 'all completed rounds'),
        'decoy candidate  : ' + (results['decoy_name']
                                 if results['decoy_index'] is not None
                                 else 'NONE (no series left outside the pass)'),
        'ablation levels  : ' + ('ON' if args.ablate else 'OFF'),
        'determinism      : TF_DETERMINISTIC_OPS='
        + os.environ.get('TF_DETERMINISTIC_OPS', '<unset>')
        + ', PYTHONHASHSEED=' + os.environ.get('PYTHONHASHSEED', '<unset>'),
        '                   TF_NUM_INTRAOP_THREADS='
        + os.environ.get('TF_NUM_INTRAOP_THREADS', '<unset>')
        + ', TF_NUM_INTEROP_THREADS='
        + os.environ.get('TF_NUM_INTEROP_THREADS', '<unset>'),
    ]
    for note in results['determinism_notes']:
        lines.append('                   ' + note)
    lines += [
        'oneDNN           : TF_ENABLE_ONEDNN_OPTS='
        + os.environ.get('TF_ENABLE_ONEDNN_OPTS', '<unset>') + ' (left at the',
        '                   default on purpose: the stock baselines were made',
        '                   with oneDNN enabled and this environment was',
        '                   measured to reproduce them bit-identically)',
        'seed shim caveat : the batch ORDER differs from an unshimmed run, so',
        '                   this run\'s MAPE is NOT bit-comparable to',
        '                   baseline_100r_*.txt.  Semantics are unchanged:',
        '                   same data, loss, optimizer, epochs, batch size and',
        '                   shuffle=True; only the permutation is pinned.',
        'np.random        : deliberately NOT reseeded per fit, so the',
        '                   client-side DP noise stream stays private and the',
        '                   Gaussian arms are judged by a NOISE-BLIND attacker',
        '                   (its residual is the injected noise itself).',
        '',
    ]
    return lines

def _client_lines(results):
    """Which series were used, and which one is the decoy control."""
    lines = ['-' * 72, 'SERIES USED', '-' * 72]
    lines.append('%-6s %-22s %-10s %-10s' % ('index', 'name', 'train', 'test'))
    for i in results['victim_indices']:
        train, test = results['series_sizes'][i]
        lines.append('%-6d %-22s %-10d %-10d' % (i, results['client_names'][i],
                                                 train, test))
    if results['decoy_index'] is not None:
        train, test = results['series_sizes'][results['decoy_index']]
        lines.append('%-6d %-22s %-10d %-10d  <- DECOY CONTROL (not a client)'
                     % (results['decoy_index'], results['decoy_name'],
                        train, test))
    lines.append('')
    return lines

def _utility_lines(results):
    """Per-configuration training outcome of the pass that produced the data.

    Reported for context only.  Because of the seed shim these numbers are not
    bit-comparable to baseline_100r_*.txt, and the MAPE itself is the B5
    personalised-hybrid metric (shared selected layer + each client's own other
    layers), not the error of one fully global model.
    """
    lines = ['-' * 72, 'UTILITY OF THE RECORDED PASS (context only)',
             '-' * 72]
    lines.append('%-22s %-8s %-10s %-10s %-10s %-14s'
                 % ('config', 'rounds', 'final MAPE', 'best MAPE', 'final MAE',
                    'fresh/round'))
    for block in results['configs']:
        mape = block['history'].get('mape', [])
        mae = block['history'].get('mae', [])
        participants = block['history'].get('participants', [])
        lines.append('%-22s %-8d %-10s %-10s %-10s %-14s' % (
            block['name'], len(mape),
            ('%.2f%%' % mape[-1]) if mape else '<none>',
            ('%.2f%%' % min(mape)) if mape else '<none>',
            ('%.4f' % mae[-1]) if mae else '<none>',
            str(participants[-1]) if participants else '<none>'))
    lines += [
        'note: MAPE/MAE are means over personalised-hybrid client models (B5),',
        '      not one fully global model, and are NOT bit-comparable to',
        '      baseline_100r_*.txt (see the seed-shim caveat above).',
        '',
    ]
    return lines

def _replay_lines(results):
    """The Phase 1 question, answered per configuration and knowledge level."""
    lines = ['-' * 72, 'REPLAY VALIDATION', '-' * 72,
             'rel_L2 = ||delta_obs - delta_pred|| / ||delta_obs||',
             '         0.0 = bit-exact reproduction, 1.0 = no better than',
             '         predicting nothing at all.',
             'cosine = direction agreement, i.e. what a ranking attack needs.',
             'decoy_series = the SAME oracle state and seed, a DIFFERENT real',
             '         series: the control that shows the score comes from the',
             '         series and not from the pipeline.',
             'note: when a mechanism is ACTIVE the client transmits G + d, so the',
             '      server recovers d as (G + d) - G.  Floating-point addition',
             '      and subtraction are not exactly inverse, so bit-exact can be',
             '      False while rel_L2 sits at the ~1e-7 numerical floor; the gate',
             '      therefore accepts rel_L2 <= 1e-6 as exact.  The none arm has no',
             '      such rounding and can be bit-exact.', '']
    for block in results['configs']:
        state = ('ACTIVE' if block['active'] else
                 'INACTIVE (the update is not modified at all)')
        lines.append('config %s : dp_method=%s clip=%g sigma=%g -> %s'
                     % (block['name'], block['dp_method'], block['dp_clip'],
                        block['dp_sigma'], state))
        lines.append('  %-22s %-13s %-5s %-14s %-14s %-10s'
                     % ('knowledge', 'role', 'n', 'rel_L2 mean',
                        'cosine mean', 'bit-exact'))
        for knowledge in results['knowledge_levels']:
            for role in ('true_series', 'decoy_series'):
                cell = [r for r in block['replays']
                        if r['knowledge'] == knowledge and r['role'] == role]
                if not cell:
                    continue
                lines.append('  %-22s %-13s %-5d %-14.6f %-14.6f %-10s' % (
                    knowledge, role, len(cell),
                    mean_of([r['rel_l2'] for r in cell]),
                    mean_of([r['cosine'] for r in cell]),
                    '%.0f%%' % (100.0 * truthy_rate(
                        [r['bit_equal'] for r in cell]))))
        lines.append('  per-cell detail (true series; the ablation levels are '
                     'in the table above and in the CSV):')
        for r in block['replays']:
            if r['role'] != 'true_series':
                continue
            if r['knowledge'] not in ('A_full_state', 'B_realistic'):
                continue
            lines.append('    round %-3d client %-14s %-14s rel_L2=%10.6f  '
                         'cos=%9.6f  bit-exact=%s'
                         % (r['round'], r['client_name'], r['knowledge'],
                            r['rel_l2'], r['cosine'], r['bit_equal']))
        lines.append('')
    return lines

def _separation_lines(results):
    """How distinguishable the true updates of different clients are."""
    lines = ['-' * 72, 'SEPARATION OF THE TRUE UPDATES (ranking feasibility)',
             '-' * 72,
             'Mean pairwise cosine between the observed deltas of DIFFERENT',
             'clients in the same round.  Near 0 = nearly orthogonal, so',
             'cosine similarity can tell clients apart; near 1 = the ranking',
             'stage would be meaningless however exact the replay is.', '']
    for block in results['configs']:
        lines.append('config %s' % block['name'])
        for r in sorted(block['separation']):
            values = block['separation'][r]
            lines.append('  round %-3d pairwise cosines : %s'
                         % (r, describe(values, 4)))
        norms = block['delta_norms']
        if norms:
            lines.append('  observed ||delta||         : %s' % describe(norms, 4))
        lines.append('')
    return lines

def _gate_lines(results):
    """The go / no-go verdict that guards the ranking stage."""
    lines = ['-' * 72, 'FEASIBILITY GATE (may the ranking stage run?)',
             '-' * 72,
             'PRECONDITION: the run must be reproducible.  Two model builds',
             'under the build seed must give identical initial weights, else',
             'every number below - and the verdict - is one unreproducible',
             'sample and no ranking stage may be built on it.',
             'THREE INDEPENDENT conditions: (1) the oracle replay must',
             'reproduce the update exactly for a noise-free arm (every cell',
             'bit-identical, or at worst to float rounding), or with a',
             'residual equal to the injected noise for a noise-injecting one -',
             'else the pipeline model is wrong; (2) the',
             'true deltas must be separable, else no ranking could work; (3)',
             'the true series must outscore the decoy control in EVERY',
             'evaluated round - a mean alone can be decided by the last',
             'decimal, and a round that the decoy reproduces just as well',
             'carries no series-specific information at all.', '']
    check = results.get('build_check')
    if check is None:
        lines.append('precondition: build determinism NOT CHECKED')
    elif check[0]:
        lines.append('precondition: build determinism OK (initial weights digest '
                     '%s)' % check[1])
    else:
        lines.append('precondition: build determinism FAILED (initial weights '
                     'digest %s)' % check[1])
    lines.append('')
    lines.append('%-22s %-13s %-12s %-10s %-11s %-9s %-11s'
                 % ('config', 'oracle exact', 'oracle cos', 'B cos', 'decoy cos',
                    'margin', 'separation'))
    for block in results['configs']:
        verdict = block['verdict']
        if verdict.get('oracle_bit_exact'):
            exact_cell = ('%.0f%%' % (100.0 * verdict['oracle_bit_exact_rate'])
                          if verdict.get('oracle_fully_exact') else '~exact')
        elif verdict.get('oracle_noise_explained'):
            exact_cell = 'n/a (noise)'
        else:
            exact_cell = ('%.0f%%'
                          % (100.0 * verdict['oracle_bit_exact_rate']))
        lines.append('%-22s %-13s %-12.6f %-10.6f %-11.6f %-9.4f %-11.4f' % (
            block['name'], exact_cell,
            verdict['oracle_mean_cosine'], verdict['realistic_mean_cosine'],
            verdict['decoy_mean_cosine'], verdict['realistic_margin'],
            verdict['separation_mean_cosine']))
    lines.append('')
    lines.append('condition 1 (did the oracle replay reproduce the update?) is')
    lines.append('mechanism-aware: a noise-free arm must reproduce the update')
    lines.append('exactly (at worst to float rounding), while a')
    lines.append('noise-injecting arm CANNOT be - the draw is not observable and')
    lines.append('not reproducible even with the whole pre-round state - so there')
    lines.append('the residual must instead equal the injected noise, i.e.')
    lines.append('residual/||noise|| = 1.00 (tolerance %.2f):' % NOISE_MATCH_TOL)
    for block in results['configs']:
        verdict = block['verdict']
        ratio = verdict.get('oracle_noise_ratio')
        if verdict.get('oracle_bit_exact'):
            lines.append('  %-22s %s (%.0f%% of cells bit-identical, mean '
                         'rel_L2 %.2e)'
                         % (block['name'],
                            'bit-exact' if verdict.get('oracle_fully_exact')
                            else 'exact to float rounding',
                            100.0 * verdict['oracle_bit_exact_rate'],
                            verdict['oracle_mean_rel_l2']))
        elif verdict.get('oracle_noise_explained'):
            lines.append('  %-22s noise-explained: residual/noise = %.4f '
                         '(||noise|| %.6f)'
                         % (block['name'], ratio, verdict['oracle_noise_l2']))
        elif ratio is not None:
            lines.append('  %-22s NOT reproduced: residual/noise = %.4f, which '
                         'must be ~1.00' % (block['name'], ratio))
        else:
            lines.append('  %-22s NOT reproduced: bit-identical in %.0f%% of '
                         'cells, mean rel_L2 %.6f, and no noise to explain it'
                         % (block['name'],
                            100.0 * verdict['oracle_bit_exact_rate'],
                            verdict['oracle_mean_rel_l2']))
    lines.append('')
    lines.append('per-round decoy margin (realistic arm: true minus decoy '
                 'cosine) - a')
    lines.append('round with a negative margin carries no series '
                 'information:')
    for block in results['configs']:
        rounds = sorted({r['round'] for r in block['replays']})
        parts = []
        for index in rounds:
            true_cell = [x['cosine'] for x in block['replays']
                         if x['round'] == index
                         and x['knowledge'] == 'B_realistic'
                         and x['role'] == 'true_series']
            decoy_cell = [x['cosine'] for x in block['replays']
                          if x['round'] == index
                          and x['knowledge'] == 'B_realistic'
                          and x['role'] == 'decoy_series']
            if true_cell and decoy_cell:
                parts.append('round %d: %+.4f'
                             % (index, mean_of(true_cell)
                                - mean_of(decoy_cell)))
        lines.append('  %-22s %s' % (block['name'],
                                     '   '.join(parts) if parts else 'n/a'))
    lines.append('')
    lines.append('signal ceiling - the SAME comparison with the pre-round state')
    lines.append('known (oracle arm), i.e. how much series information the update')
    lines.append('carries at all.  A large ceiling next to a ~0 threat-model')
    lines.append('margin means the information exists but the unmodelled')
    lines.append('pre-round state buries it, so the obstacle is the knowledge')
    lines.append('model, not the candidate list:')
    for block in results['configs']:
        verdict = block['verdict']
        ceiling = verdict.get('oracle_round_margins') or {}
        if not ceiling:
            lines.append('  %-22s n/a' % block['name'])
            continue
        parts = '   '.join('round %d: %+.4f' % (i, m)
                           for i, m in sorted(ceiling.items()))
        if verdict.get('oracle_margin'):
            keep = ('   (threat model keeps %.1f%% of it)'
                    % (100.0 * verdict['realistic_margin']
                       / verdict['oracle_margin']))
        else:
            keep = ''
        lines.append('  %-22s %s%s' % (block['name'], parts, keep))

    lines.append('')
    reproducible = (results.get('build_check') or (True, ''))[0]
    for block in results['configs']:
        prefix = ('EXPECTED-FAILURE DEMO: '
                  if results['args'].no_seed_shim else '')
        if not reproducible:
            lines.append('  %-22s ABORT: the model build is not reproducible, so '
                         'this verdict is not' % block['name'])
            lines.append('  %-22s        meaningful - fix the build seed first.'
                         % '')
            continue
        lines.append('  %-22s %s%s' % (block['name'], prefix,
                                       block['verdict']['verdict']))
    if results['args'].no_seed_shim:
        lines += [
            '',
            'NOTE: --no-seed-shim was used, so the per-fit seed was NOT set.',
            'The stock fit draws its batch permutation from the global RNG,',
            'which depends on the whole call history, so a failure to',
            'reproduce here is the EXPECTED result and is the empirical',
            'justification for the shim.  It is not evidence of privacy.',
        ]
    lines.append('')
    return lines

def _dp_lines(results):
    """Cross-check: is a Gaussian residual the noise, or a replay failure?

    The stock diagnostics record the norm of the update, of the clipped update
    and of the injected noise.  If the replay residual equals the noise norm,
    the residual IS the mechanism, not a bug in the replay.
    """
    lines = ['-' * 72, 'DP DIAGNOSTICS CROSS-CHECK', '-' * 72]
    any_active = False
    for block in results['configs']:
        diag = block['diag_summary']
        if not diag:
            continue
        any_active = True
        lines.append('config %s' % block['name'])
        lines.append('  updates measured       : %d' % diag['n'])
        lines.append('  ||raw delta|| mean     : %.6f' % diag['raw_l2'])
        lines.append('  ||clipped delta|| mean : %.6f' % diag['clipped_l2'])
        lines.append('  ||noise|| mean         : %.6f' % diag['noise_l2'])
        lines.append('  noise/raw delta ratio  : %.6f' % diag['noise_ratio'])
        lines.append('  clipping bound         : %.2f%% of updates were clipped'
                     % (100.0 * diag['clipped_rate']))
        if diag['residual_l2'] is not None:
            lines.append('  residual, ORACLE arm   : %.6f  (full hidden state '
                         'restored, so only the' % diag['residual_l2'])
            lines.append('                           mechanism can separate '
                         'prediction from transmission)')
            if diag['noise_l2'] > 0:
                lines.append('  -> residual/noise      : %.4f   (1.00 means the '
                             'residual IS the noise)'
                             % (diag['residual_l2'] / diag['noise_l2']))
            else:
                lines.append('  -> this arm adds NO noise (clipping only), so a '
                             'residual here')
                lines.append('     would be a replay error, not a mechanism '
                             'effect.')
        if diag.get('residual_l2_realistic') is not None:
            lines.append('  residual, REALISTIC arm: %.6f  (also carries the '
                         'hidden-state gap)'
                         % diag['residual_l2_realistic'])
        lines.append('')
    if not any_active:
        lines.append('No active mechanism in this run: there is no noise to')
        lines.append('cross-check (none / clip_only add none by design).')
        lines.append('')
    return lines

def _identity_lines(results):
    """Identification results, kept separate from the reconstruction results."""
    lines = ['-' * 72, 'IDENTIFICATION (a separate claim from reconstruction)',
             '-' * 72]
    if not results['identity_rows']:
        lines += ['Not run (add --identity).  Identification asks WHICH series',
                  'produced an update; reconstruction asks whether the update',
                  'itself can be reproduced.  They are different claims and are',
                  'reported in different CSVs.', '']
        return lines
    lines.append('%-22s %-14s %-10s %-8s %-10s %-14s %-11s %s'
                 % ('config', 'knowledge', 'top-1', 'chance', 'shuffled',
                    'mean true rank', 'mean margin', 'cells'))
    keys = []
    for row in results['identity_rows']:
        key = (row['config'], row['knowledge'])
        if key not in keys:
            keys.append(key)
    for config, knowledge in keys:
        cell = [r for r in results['identity_rows']
                if r['config'] == config and r['knowledge'] == knowledge]
        lines.append('%-22s %-14s %-10s %-8s %-10s %-14s %-11s %d' % (
            config, knowledge,
            '%.1f%%' % (100.0 * truthy_rate([r['top1_correct'] for r in cell])),
            '%.1f%%' % (100.0 * mean_of([r['chance'] for r in cell])),
            '%.1f%%' % (100.0 * truthy_rate(
                [r['shuffled_top1_correct'] for r in cell])),
            '%.2f' % mean_of([r['true_rank'] for r in cell]),
            '%.4f' % mean_of([r['margin'] for r in cell]),
            len(cell)))
    lines += ['',
              'The shuffled column permutes only the candidate->client label',
              'mapping, leaving every score untouched: with no real signal it',
              'must fall to chance.  If top-1 and shuffled are both near',
              'chance, the ranking carries no identity information at all.']
    # Sample-size guard: a top-1 rate is a proportion over `cells` observations,
    # so its resolution is 1/cells.  With a handful of cells the rate can only
    # take a few values and a "50% vs 10% chance" line says nothing at all -
    # that is exactly the kind of number this section must not let a reader
    # over-read.
    smallest = min(len([r for r in results['identity_rows']
                        if r['config'] == c and r['knowledge'] == k])
                   for c, k in keys)
    if smallest < 10:
        lines += ['',
                  'WARNING: the smallest cell above holds only %d identity '
                  'cell(s).' % smallest,
                  'A top-1 rate over %d observation(s) can only take the value'
                  % smallest,
                  '%s, so it cannot separate a real effect from luck'
                  % ' or '.join('%.0f%%' % (100.0 * i / smallest)
                                for i in range(smallest + 1)),
                  '(chance is %.0f%%).  Treat this table as a smoke test of the'
                  % (100.0 * mean_of([r['chance'] for r in
                                     results['identity_rows']])),
                  'identity pipeline, not as evidence, and raise --rounds or',
                  '--victims before quoting any top-1 number.']
    lines.append('')
    return lines

def _claims_lines(results):
    """Explicitly separate what is claimed from what is not."""
    return [
        '-' * 72, 'CLAIMED vs NOT CLAIMED', '-' * 72,
        'CLAIMED (Phase 1):',
        '  * whether a candidate load series can reproduce the selected-layer',
        '    update a client transmitted, and how much of that reproduction',
        '    depends on state the attacker should not have (oracle vs public);',
        '  * the same measured against a decoy series and, for the identity',
        '    stage, against chance with a shuffled-label control;',
        '  * the separation of different clients\' true updates, which is what',
        '    bounds any later ranking.',
        'NOT CLAIMED:',
        '  * no differential-privacy guarantee, no epsilon/delta, no end-to-end',
        '    privacy accounting of any kind;',
        '  * a poor replay score is NOT privacy: it says this replay was',
        '    inexact, and the ablation levels name the responsible component;',
        '  * identifying a series is NOT recovering it (separate metrics and',
        '    separate CSV columns);',
        '  * clipping alone hides nothing: it preserves the direction exactly;',
        '  * this run\'s MAPE is not bit-comparable to baseline_100r_*.txt',
        '    (documented seed-shim deviation);',
        '  * the Gaussian arms are judged by a noise-blind attacker, so their',
        '    residual is the injected noise, not a failed replay: condition 1',
        '    checks residual/||noise|| = 1.00 for them, since bit-exactness is',
        '    impossible by construction once a noise draw is injected.',
        '',
    ]

def render_report(results):
    """Assemble the full text report."""
    lines = []
    lines += _header_lines(results)
    lines += _client_lines(results)
    lines += _utility_lines(results)
    lines += _replay_lines(results)
    lines += _separation_lines(results)
    lines += _gate_lines(results)
    lines += _dp_lines(results)
    lines += _identity_lines(results)
    lines += _claims_lines(results)
    lines += ['-' * 72, 'FILES WRITTEN BY THIS RUN', '-' * 72]
    for label, path in results['written']:
        lines.append('  %-22s %s' % (label, os.path.abspath(path)))
    lines += ['',
              'Existing artifacts (asla_complete.py, baseline_100r_*.txt,',
              'dp_diag_*.txt, *.png) were never opened for writing.',
              '']
    return lines

# ---------------------------------------------------------------------------
# PART 11: THE RUNNER
# ---------------------------------------------------------------------------

def summarize_diag(records, replays):
    """Scalar DP summary of one configuration, from the STOCK diagnostics.

    Nothing here is assumed: the norms come from the stock diagnostics rows
    (which measure the very arrays that were transmitted).  Two residuals are
    reported, because they answer different questions:

      * oracle arm - full hidden state restored, so the ONLY thing separating
        the prediction from the transmitted update is the mechanism itself.
        For a Gaussian arm this residual must equal ||noise||; that equality is
        the cross-check that a large residual is injected noise rather than a
        broken replay.
      * realistic arm - also carries the hidden-state gap, i.e. what an
        attacker without the pre-round state actually sees.
    """
    rows = [r['diag'] for r in records if r.get('diag')]
    if not rows:
        return {}

    def residual(level):
        cell = [r for r in replays if r['knowledge'] == level
                and r['role'] == 'true_series']
        return (mean_of([r['rel_l2'] * r['obs_l2'] for r in cell])
                if cell else None)

    return {
        'n': len(rows),
        'raw_l2': mean_of([d.get('raw_delta_l2') for d in rows]),
        'clipped_l2': mean_of([d.get('clipped_delta_l2') for d in rows]),
        'noise_l2': mean_of([d.get('noise_l2') for d in rows], default=0.0),
        'noise_ratio': mean_of(
            [d.get('noise_to_raw_delta_ratio') for d in rows
             if d.get('noise_to_raw_delta_ratio') is not None], default=0.0),
        'clipped_rate': truthy_rate([(d.get('clip_factor') or 1.0) < 1.0
                                     for d in rows]),
        'residual_l2': residual('A_full_state'),
        'residual_l2_realistic': residual('B_realistic'),
    }

UTILITY_CSV_HEADER = ('config', 'dp_method', 'dp_clip', 'dp_sigma', 'active',
                      'rounds_completed', 'final_mape', 'best_mape',
                      'final_mae', 'participants_last')

def run_validation(args):
    """The Phase 1 experiment: record, replay, measure, gate, report."""
    determinism_notes = configure_determinism(args.seed)
    layer = args.agg_layer
    configs = [get_config(name) for name in (args.configs or CONFIG_NAMES)]

    # Precondition for every number below.  Without this check an
    # unreproducible build shows up only as a gate verdict that changes from
    # run to run, which is exactly the failure mode this study must not have.
    build_ok, build_digest = verify_build_determinism(args.build_seed)
    determinism_notes.append(
        'build determinism self-check: %s (weights digest %s under seed %d)'
        % ('OK' if build_ok else 'FAILED', build_digest, args.build_seed))
    print('build determinism: %s (weights digest %s under build seed %d)'
          % ('OK' if build_ok else 'FAILED', build_digest, args.build_seed))
    if not build_ok:
        print('WARNING: two builds under the same seed gave different weights,')
        print('         so this run is NOT reproducible and its gate verdict')
        print('         is a single unreproducible sample.')

    print('=' * 72)
    print('ASLA PRIVACY-ATTACK STUDY - PHASE 1 (replay validation)')
    print('=' * 72)
    print('loading data from ' + os.path.abspath(args.data_folder) + ' ...')
    client_data, client_names = asla.load_pjm_multiple_files(args.data_folder)

    # One extra series is requested so that a decoy control exists outside the
    # pass.  Preprocessing consumes no random numbers, so this cannot perturb
    # the run.
    usable, series = build_series_pool(client_data, client_names,
                                       args.clients + 1)
    victim_indices = usable[:args.clients]
    if not victim_indices:
        raise RuntimeError('no usable client series in '
                           + os.path.abspath(args.data_folder))
    decoy_index = pick_decoy(usable, victim_indices)

    victim_ids = None
    if args.victims is not None:
        victim_ids = set()
        for pos in args.victims:
            if pos < 0 or pos >= len(victim_indices):
                raise RuntimeError(
                    '--victims position %d is outside the selected-client '
                    'list (0..%d)' % (pos, len(victim_indices) - 1))
            victim_ids.add(victim_indices[pos])
    args.victim_ids = victim_ids

    print('selected clients (%d): %s'
          % (len(victim_indices),
             ', '.join(client_names[i] for i in victim_indices)))
    if decoy_index is not None:
        print('decoy control     : %s' % client_names[decoy_index])

    passes = []
    for cfg in configs:
        # Applied before any model of this configuration is built, so the
        # recorded pass - and therefore every replayed number - is reproducible
        # across processes and starts from the same documented initial weights.
        seed_for_build(args.build_seed)
        print('')
        print('-' * 72)
        print('config %s : dp_method=%s clip=%g sigma=%g' % cfg)
        print('-' * 72)
        clients, history, records = run_config_pass(
            series, victim_indices, client_names, cfg, args, layer)
        rounds_present = sorted({r['round'] for r in records})
        passes.append({'cfg': cfg, 'clients': clients, 'history': history,
                       'records': records, 'rounds': rounds_present})
        print('  recorded %d client update(s) over %d round(s)'
              % (len(records), len(rounds_present)))

    if args.eval_rounds is None:
        args.eval_rounds = set(r for p in passes for r in p['rounds'])
    return _evaluate_and_report(args, passes, series, victim_indices,
                                decoy_index, client_names, layer,
                                determinism_notes,
                                build_check=(build_ok, build_digest))

def _evaluate_and_report(args, passes, series, victim_indices, decoy_index,
                         client_names, layer, determinism_notes,
                         build_check=None):
    """Replay every pass, aggregate, gate, write the outputs, return results."""
    all_replays, identity_rows, blocks = [], [], []
    for entry in passes:
        name, dp_method, dp_clip, dp_sigma = entry['cfg']
        active = config_is_active(dp_method, dp_clip, dp_sigma)
        print('')
        print('replaying config %s (%s) ...'
              % (name, 'ACTIVE' if active else 'inactive'))
        replays = evaluate_replays(entry['records'], series, args.victim_ids,
                                   decoy_index, layer, args)
        all_replays += replays
        separation = summarize_separation(entry['records'], args.eval_rounds)
        # The DP summary is computed BEFORE the gate because condition 1 needs
        # it: for a noise-injecting arm the oracle residual is validated by
        # comparing it with ||noise||, not by bit-exactness (which such an arm
        # cannot achieve by construction).
        diag_summary = summarize_diag(entry['records'], replays)
        verdicts = evaluate_gate(replays, separation, {name: diag_summary})
        norms = [l2_norm(observed_delta(r)) for r in entry['records']
                 if r['round'] in args.eval_rounds
                 and (args.victim_ids is None
                      or r['client_id'] in args.victim_ids)]
        blocks.append({
            'name': name, 'dp_method': dp_method, 'dp_clip': dp_clip,
            'dp_sigma': dp_sigma, 'active': active,
            'history': entry['history'], 'records': entry['records'],
            'replays': replays, 'separation': separation,
            'delta_norms': norms,
            'verdict': verdicts[0] if verdicts else {},
            'diag_summary': diag_summary,
        })
        print('  replayed %d cell(s); gate: %s'
              % (len(replays),
                 (verdicts[0]['verdict'] if verdicts else 'no data')))
        if args.identity:
            identity_rows += run_identity(entry['records'], series,
                                          victim_indices, layer,
                                          args.identity_knowledge, args)

    knowledge_levels = ['A_full_state', 'B_realistic']
    if args.ablate:
        knowledge_levels += ['ablate_weights_only', 'ablate_optimizer_only']
    results = {
        'args': args, 'layer': layer, 'determinism_notes': determinism_notes,
        'client_names': client_names, 'victim_indices': victim_indices,
        'victim_ids': args.victim_ids, 'decoy_index': decoy_index,
        'decoy_name': (client_names[decoy_index]
                       if decoy_index is not None else None),
        'series_sizes': dict((i, (len(series[i][0]), len(series[i][2])))
                             for i in series),
        'eval_rounds': args.eval_rounds,
        'config_order': [b['name'] for b in blocks],
        'knowledge_levels': knowledge_levels,
        'configs': blocks, 'identity_rows': identity_rows, 'written': [],
        'build_check': build_check,
    }

    utility_rows = []
    for block in blocks:
        mape = block['history'].get('mape', [])
        mae = block['history'].get('mae', [])
        participants = block['history'].get('participants', [])
        utility_rows.append({
            'config': block['name'], 'dp_method': block['dp_method'],
            'dp_clip': block['dp_clip'], 'dp_sigma': block['dp_sigma'],
            'active': block['active'], 'rounds_completed': len(mape),
            'final_mape': mape[-1] if mape else None,
            'best_mape': min(mape) if mape else None,
            'final_mae': mae[-1] if mae else None,
            'participants_last': participants[-1] if participants else None,
        })

    written = []
    written.append(('replay metrics CSV', write_csv(
        args.replay_csv, REPLAY_CSV_HEADER, all_replays)))
    written.append(('utility CSV', write_csv(
        args.utility_csv, UTILITY_CSV_HEADER, utility_rows)))
    if identity_rows:
        written.append(('identity CSV', write_csv(
            args.identity_csv, IDENTITY_CSV_HEADER, identity_rows)))
    text_target = safe_output_path(args.out)
    written.append(('text report', text_target))
    results['written'] = written
    report_lines = render_report(results)
    write_text(text_target, report_lines)
    results['report_lines'] = report_lines

    print('')
    print('=' * 72)
    print('PHASE 1 GATE (review this before any ranking stage)')
    print('=' * 72)
    if build_check is not None:
        print('  precondition: build determinism %s (weights digest %s)'
              % ('OK' if build_check[0] else 'FAILED', build_check[1]))
    for block in blocks:
        verdict = block['verdict']
        if build_check is not None and not build_check[0]:
            print('  %-22s ABORT: the model build is not reproducible, so this '
                  'verdict is not' % block['name'])
            print('  %-22s        meaningful - fix the build seed first.' % '')
            continue
        if verdict and verdict.get('oracle_bit_exact'):
            exact_text = ('%.1f%%' % (100.0 * verdict['oracle_bit_exact_rate'])
                          if verdict.get('oracle_fully_exact') else '~exact')
        elif verdict and verdict.get('oracle_noise_explained'):
            exact_text = 'noise'
        elif verdict:
            exact_text = '%.1f%%' % (100.0 * verdict['oracle_bit_exact_rate'])
        else:
            exact_text = 'n/a'
        print('  %-22s oracle exact=%6s oracle cos=%s B cos=%s decoy cos=%s '
              'margin=%s sep=%s' % (
                  block['name'], exact_text,
                  ('%.6f' % verdict['oracle_mean_cosine']) if verdict else 'n/a',
                  ('%.6f' % verdict['realistic_mean_cosine'])
                  if verdict else 'n/a',
                  ('%.6f' % verdict['decoy_mean_cosine']) if verdict else 'n/a',
                  ('%+.4f' % verdict['realistic_margin']) if verdict else 'n/a',
                  ('%.4f' % verdict['separation_mean_cosine'])
                  if verdict else 'n/a'))
        print('      -> ' + verdict.get('verdict', 'no verdict'))
    print('')
    return results

# ---------------------------------------------------------------------------
# PART 12: COMMAND LINE
# ---------------------------------------------------------------------------

def parse_int_list(text):
    """Parse '1,2,3' into [1, 2, 3] (whitespace tolerated, empty -> None)."""
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(',') if p.strip()]
    return [int(p) for p in parts] if parts else None

def parse_str_list(text):
    """Parse 'none,clip_only_c2' into a list of names (empty -> None)."""
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(',') if p.strip()]
    return parts if parts else None

def build_parser():
    """The Phase 1 command line."""
    parser = argparse.ArgumentParser(
        description='ASLA privacy-attack study - ' + PHASE + '.',
        epilog='The 100-round, four-configuration attack (--mode full) is NOT '
               'implemented yet: the replay validation must be reviewed first.')
    parser.add_argument('--mode', default='validation',
                        choices=['validation', 'full'],
                        help="'validation' runs the Phase 1 replay study; "
                             "'full' is NOT implemented and exits non-zero")
    parser.add_argument('--data-folder', default='.',
                        help='folder holding the client CSV files (default: .)')
    parser.add_argument('--rounds', type=int, default=2,
                        help='communication rounds per configuration (Phase 1: '
                             'keep this small; every recorded update is held in '
                             'memory, so a long run should be restricted with '
                             '--eval-rounds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='run seed (default: 42, same as the stock main)')
    parser.add_argument('--build-seed', type=int, default=7,
                        help='seed for the attacker-side model construction '
                             '(default: 7)')
    parser.add_argument('--agg-layer', type=int, default=1,
                        help='selected layer index (default: 1 = Dense(50))')
    parser.add_argument('--quant-bits', type=int, default=32,
                        choices=[8, 16, 32],
                        help='quantization bits (default: 32 = lossless, so the '
                             'observed delta is exact)')
    parser.add_argument('--clients', type=int, default=10,
                        help='number of clients in the pass (default: 10)')
    parser.add_argument('--configs', default=None,
                        help='comma-separated subset of: '
                             + ', '.join(CONFIG_NAMES))
    parser.add_argument('--victims', default=None,
                        help='comma-separated 0-based positions in the '
                             'selected-client list (default: all clients)')
    parser.add_argument('--eval-rounds', default=None,
                        help='comma-separated round numbers to evaluate '
                             '(default: every completed round)')
    parser.add_argument('--no-seed-shim', action='store_true',
                        help='do NOT set the per-fit seed; demonstrates that '
                             'the stock pipeline is not reproducible, and '
                             'makes the gate report an expected failure')
    parser.add_argument('--ablate', action='store_true',
                        help='also run the two ablation knowledge levels '
                             '(weights-only / optimizer-only) to diagnose a '
                             'mismatch instead of merely reporting it')
    parser.add_argument('--identity', action='store_true',
                        help='also run the K x K identification stage '
                             '(Phase 2 scale: K replays per cell; off by '
                             'default so Phase 1 stays cheap)')
    parser.add_argument('--identity-knowledge', default='B_realistic',
                        choices=list(KNOWLEDGE_LEVELS),
                        help='knowledge level used by --identity')
    parser.add_argument('--out', default='privacy_attack_validation.txt',
                        help='text report (default: '
                             'privacy_attack_validation.txt)')
    parser.add_argument('--replay-csv',
                        default='privacy_attack_replay_metrics.csv',
                        help='per-cell replay metrics CSV')
    parser.add_argument('--utility-csv', default='privacy_attack_utility.csv',
                        help='per-configuration utility CSV')
    parser.add_argument('--identity-csv', default='privacy_attack_identity.csv',
                        help='identification CSV (only with --identity)')
    return parser

def main(argv=None):
    """Entry point.  Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.configs = parse_str_list(args.configs)
    args.victims = parse_int_list(args.victims)
    args.eval_rounds = (set(parse_int_list(args.eval_rounds))
                        if args.eval_rounds is not None else None)

    if args.mode == 'full':
        print('=' * 72)
        print('--mode full IS NOT IMPLEMENTED (' + PHASE + ').')
        print('=' * 72)
        print('The 100-round, four-configuration attack and the candidate')
        print('ranking stage must not run before the replay validation has')
        print('been reviewed.  Run the default mode first:')
        print('  python ' + SCRIPT_NAME)
        return 2

    try:
        results = run_validation(args)
    except (RuntimeError, ValueError) as exc:
        print('')
        print('ERROR: ' + str(exc))
        return 3

    print('')
    for line in results['report_lines']:
        print(line)
    print('Phase 1 complete: review the gate verdict above before any '
          'ranking stage.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

