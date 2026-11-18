"""
hybrid_lite.py - Hybrid Lite CNN-XGBoost intrusion detector for CAN-bus attacks.

Final model:  DCNN_Lite -> 64-d embedding -> int8 TFLite -> compiled XGBoost (100 trees, depth 6)

For each of the five datasets (DoS, Fuzzy, Gear, RPM, Replay) the script
  1. builds 29x29 CAN-ID bit frames and makes a shuffled 70/30 split (seed 42),
  2. trains DCNN_Lite and keeps its 64-d embedding layer,
  3. trains XGBoost on the training embeddings,
  4. exports the CNN to int8 TFLite,
  5. classifies every test frame with the real int8 model + the compiled C XGBoost,
  6. measures single-frame CPU latency against the 0.5 ms/frame budget,
  7. saves the result tables (CSV/PNG/JSON) and the model files.

The script was converted from a Google Colab notebook; the Colab-specific parts
(Drive mounting, pip) were replaced by command-line options.

Usage:
    python hybrid_lite.py --data-dir ./data --out-dir ./results

Needs: Python 3.10-3.12, numpy, pandas, matplotlib, tensorflow, xgboost and gcc (or clang) on the PATH.
A GPU speeds up training; latency is always measured on the CPU.
"""
import argparse, os, sys
_ap = argparse.ArgumentParser(description="Hybrid Lite CNN-XGBoost CAN-bus IDS")
_ap.add_argument("--data-dir", default="./data", help="folder with DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv, RPM_dataset.csv")
_ap.add_argument("--out-dir", default="./results", help="folder for tables, JSON and model files")
_ARGS = _ap.parse_args()
DATA_DIR = os.path.abspath(_ARGS.data_dir)
RESULT_DIR = os.path.abspath(_ARGS.out_dir)
import matplotlib
matplotlib.use("Agg")      # no display needed



# ######################################################################
# 0. Setup
# ######################################################################

import os, time, warnings, gc, json, math, ctypes, shutil, subprocess, pathlib, platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 110

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import xgboost as xgb
from xgboost import XGBClassifier

print(f'TensorFlow {tf.__version__} | XGBoost {xgb.__version__} | CPU cores: {os.cpu_count()} | {platform.processor() or platform.machine()}')



# ######################################################################
# 1. Configuration
# ######################################################################

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS = {
    'DoS':    DATA_DIR + '/DoS_dataset.csv',
    'Fuzzy':  DATA_DIR + '/Fuzzy_dataset.csv',
    'Gear':   DATA_DIR + '/gear_dataset.csv',
    'RPM':    DATA_DIR + '/RPM_dataset.csv',
    'Replay': DATA_DIR + '/Replay_dataset.csv',   # synthesised from DoS normal traffic
}

# ---- Replay-dataset synthesis constants (only used if the Replay CSV does not exist yet) ----
REPLAY_SOURCE_CSV    = DATA_DIR + '/DoS_dataset.csv'
REPLAY_N_INTRUSIONS  = 300       # number of replay bursts
REPLAY_CAPTURE_SIZE  = 204       # messages captured and re-injected per burst
REPLAY_INJECT_MS     = 1.0       # ms between replayed messages
REPLAY_NORMAL_ROWS   = 300_000   # normal rows kept
REPLAY_SEED          = 42

# ---- data / CNN training ----
SAMPLE_MAIN   = 150_000     # rows read per dataset
BATCH_SIZE    = 512
EPOCHS_DCNN   = 10          # EarlyStopping (patience 2) may stop earlier
LR_PEAK       = 1e-3
LR_MIN        = 1e-5
WINDOW        = 29
TEST_FRAC     = 0.30
EMBEDDING_DIM = 64
N_CALIB       = 256         # training frames used for int8 calibration

# ---- FINAL XGBoost configuration ----
XGB_N_TREES   = 100
XGB_MAX_DEPTH = 6
XGB_LR        = 0.1

# ---- latency benchmark ----
REALTIME_THRESHOLD_MS = 0.5
BENCH_REPS   = 1000         # single-frame calls per round
BENCH_ROUNDS = 9            # the MEDIAN round is reported
BENCH_WARMUP = 200

# ---- outputs ----
OUT_DIR = RESULT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ---- GPU / mixed precision (as in all earlier runs) ----
try:
    mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision (float16) enabled')
except Exception as e:
    print(f'Mixed precision unavailable: {e}')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f'GPU: {gpus[0].name}')
else:
    print('No GPU detected - CPU mode (CNN training will be slow)')



# ######################################################################
# 2. Data pipeline
# ######################################################################

# =============================================================================
# DATA PIPELINE
# =============================================================================
def load_and_build_frames(filepath, sample_rows):
    """CSV -> CAN-ID bits -> sliding-window frames. Returns X (N,29,29,1) float32, y (N,) int8."""
    cols = ['Timestamp','CAN_ID','DLC','D0','D1','D2','D3','D4','D5','D6','D7','Flag']
    df = pd.read_csv(filepath, header=None, names=cols, nrows=sample_rows, dtype=str)
    df['Flag'] = df['Flag'].str.strip()
    n_normal = int((df['Flag'] == 'R').sum())
    n_attack = int((df['Flag'] == 'T').sum())
    print(f"    rows={len(df):,}  normal={n_normal:,}  attack={n_attack:,}", end='  ')

    ids   = df['CAN_ID'].values
    flags = (df['Flag'] == 'T').astype(np.int8).values
    del df; gc.collect()

    int_ids = np.array([int(x.strip(), 16) for x in ids], dtype=np.int32)
    shifts  = np.arange(28, -1, -1, dtype=np.int32)
    bits    = ((int_ids[:, None] >> shifts[None, :]) & 1).astype(np.float32)
    del int_ids; gc.collect()

    n = len(bits) - WINDOW + 1
    frames = np.lib.stride_tricks.as_strided(
        bits, shape=(n, WINDOW, 29),
        strides=(bits.strides[0], bits.strides[0], bits.strides[1])).copy()
    frames = frames[:, :, :, np.newaxis]                                  # (N,29,29,1)

    flag_win = np.lib.stride_tricks.as_strided(
        flags, shape=(n, WINDOW), strides=(flags.strides[0], flags.strides[0]))
    y = flag_win.max(axis=1).astype(np.int8)
    del bits, flag_win; gc.collect()

    print(f"frames={frames.shape}  atk_ratio={y.mean():.2%}")
    return frames, y, n_normal, n_attack


def synthesize_replay_dataset(save_path):
    """
    Builds the Replay dataset from real normal CAN traffic (DoS source CSV):
    300 captured windows of 204 messages are re-injected at 1 ms spacing (Flag='T'),
    merged with 300k normal rows (Flag='R') and sorted by timestamp.
    Skipped if the file already exists.
    """
    if os.path.exists(save_path):
        print(f"    Replay dataset already exists at {save_path} - skipping synthesis.")
        return save_path

    print(f"    Synthesising replay dataset from {REPLAY_SOURCE_CSV} ...")
    cols = ['Timestamp','CAN_ID','DLC','D0','D1','D2','D3','D4','D5','D6','D7','Flag']
    src  = pd.read_csv(REPLAY_SOURCE_CSV, header=None, names=cols, dtype=str)
    src['Flag']      = src['Flag'].str.strip()
    src['Timestamp'] = src['Timestamp'].astype(float)
    normal           = src[src['Flag'] == 'R'].reset_index(drop=True)
    del src; gc.collect()
    print(f"      Source normal rows available: {len(normal):,}")

    max_start = len(normal) - REPLAY_CAPTURE_SIZE - 10
    seg_size  = max_start // REPLAY_N_INTRUSIONS
    rng       = np.random.default_rng(REPLAY_SEED)
    starts = [i * seg_size + int(rng.integers(0, max(1, seg_size // 2)))
              for i in range(REPLAY_N_INTRUSIONS)]
    starts = [min(s, max_start) for s in starts]

    attack_chunks = []
    for start in starts:
        captured  = normal.iloc[start : start + REPLAY_CAPTURE_SIZE].copy()
        after_idx = min(start + REPLAY_CAPTURE_SIZE + 50, len(normal) - 1)
        base_ts   = normal.iloc[after_idx]['Timestamp']
        new_ts    = base_ts + np.arange(len(captured)) * (REPLAY_INJECT_MS / 1000.0)
        captured  = captured.copy()
        captured['Timestamp'] = new_ts
        captured['Flag']      = 'T'
        attack_chunks.append(captured)
    replay_attacks = pd.concat(attack_chunks, ignore_index=True)

    n_normal_actual = min(REPLAY_NORMAL_ROWS, len(normal))
    normal_sample   = normal.sample(n=n_normal_actual, random_state=REPLAY_SEED)

    combined = pd.concat([normal_sample, replay_attacks], ignore_index=True)
    combined = combined.sort_values('Timestamp').reset_index(drop=True)
    combined['Timestamp'] = combined['Timestamp'].map(lambda x: f'{x:.6f}')
    combined.to_csv(save_path, header=False, index=False)

    n_atk = (combined['Flag'] == 'T').sum()
    n_nrm = (combined['Flag'] == 'R').sum()
    print(f"      Saved {len(combined):,} rows -> {save_path}")
    print(f"        Normal={n_nrm:,} ({n_nrm/len(combined):.1%})  Attack={n_atk:,} ({n_atk/len(combined):.1%})")
    del normal, replay_attacks, normal_sample, combined, attack_chunks; gc.collect()
    return save_path


def shuffle_split(X, y, seed=42):
    """Shuffle then 70/30 split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    cut  = int(len(X) * (1 - TEST_FRAC))
    return X[:cut], X[cut:], y[:cut], y[cut:]


def make_ds(X, y, batch, shuffle=False, cache=True):
    """tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.int32)))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(min(10_000, len(X)), reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)



# ######################################################################
# 3. DCNN_Lite feature extractor and metrics
# ######################################################################

# =============================================================================
# DCNN_Lite  (CNN feature extractor)
# =============================================================================
def _cbr(x, f, k, strides=1, padding='same'):
    x = layers.Conv2D(f, k, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu')(x)


def build_dcnn_lite(embedding_dim=64):
    inp = keras.Input(shape=(29, 29, 1))

    # Stem
    x = _cbr(inp, 16, 3)
    x = _cbr(x,   16, 3)
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
    x = _cbr(x,   32, 3)

    # One lightweight Inception-ResNet-style block (2 branches)
    ch = x.shape[-1]
    b1 = _cbr(x, 16, 1)
    b2 = _cbr(_cbr(x, 16, 1), 16, 3)
    m  = layers.Concatenate()([b1, b2])
    m  = layers.Conv2D(ch, 1, padding='same', use_bias=False)(m)
    m  = layers.BatchNormalization()(m)
    x  = layers.Lambda(lambda t: t[0] + 0.1 * t[1])([x, m])
    x  = layers.Activation('relu')(x)

    # Single strided conv
    x = _cbr(x, 64, 3, strides=2)

    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Lambda(lambda t: tf.cast(t, tf.float32))(x)
    emb = layers.Dense(embedding_dim, activation='relu', name='feature_embedding')(x)
    out = layers.Dense(2, activation='softmax', dtype='float32', name='pretrain_head')(emb)   # used for training only
    return keras.Model(inp, out, name='DCNN_Lite')


# =============================================================================
# Training utilities and metrics
# =============================================================================
def _free():
    keras.backend.clear_session()
    gc.collect()


def make_cosine_lr(steps_per_epoch, epochs):
    total_steps = steps_per_epoch * epochs
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR_PEAK,
        decay_steps=max(total_steps - steps_per_epoch, 1),
        alpha=LR_MIN / LR_PEAK,
        warmup_target=LR_PEAK,
        warmup_steps=steps_per_epoch,
    )


def _std_cbs(patience=2):
    return [keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                          restore_best_weights=True, verbose=0)]


def _metrics(y_true, y_pred):
    TP = int(((y_pred==1)&(y_true==1)).sum())
    TN = int(((y_pred==0)&(y_true==0)).sum())
    FP = int(((y_pred==1)&(y_true==0)).sum())
    FN = int(((y_pred==0)&(y_true==1)).sum())
    tot = TP+TN+FP+FN
    FNR = FN/(TP+FN)*100 if (TP+FN) else 0.
    ER  = (FP+FN)/tot*100 if tot    else 0.
    P   = TP/(TP+FP)      if (TP+FP) else 0.
    R   = TP/(TP+FN)      if (TP+FN) else 0.
    F1  = 2*P*R/(P+R)     if (P+R)  else 0.
    return dict(FNR=FNR, ER=ER, Precision=P, Recall=R, F1=F1, TP=TP, TN=TN, FP=FP, FN=FN)

_probe = build_dcnn_lite(EMBEDDING_DIM)
print(f'DCNN_Lite parameter count: {_probe.count_params():,}')
del _probe; _free()



# ######################################################################
# 4. Deployment helpers — compiled XGBoost, int8 TFLite, latency harness
# ######################################################################

# =============================================================================
# Compiled XGBoost: trained binary:logistic model -> C source (nested if/else on
# float32 thresholds) -> gcc -O3 shared library -> ctypes.
# Decision rule identical to XGBoost:  x[f] < threshold -> left child, else right;
# label = 1  <=>  base_margin + sum(leaf values) > 0.
# =============================================================================
class CompiledXGB:
    _counter = 0

    def __init__(self, clf, tag='xgb', workdir='/tmp/xgb_compiled', opt_flag='-O3'):
        CompiledXGB._counter += 1
        uid = f'{tag}_{CompiledXGB._counter}'
        os.makedirs(workdir, exist_ok=True)
        booster = clf.get_booster() if hasattr(clf, 'get_booster') else clf

        model_json = os.path.join(workdir, f'{uid}.json')
        booster.save_model(model_json)
        with open(model_json) as fh:
            model = json.load(fh)
        learner = model['learner']
        if learner['objective']['name'] != 'binary:logistic':
            raise NotImplementedError('CompiledXGB supports binary:logistic only.')
        bs = float(str(learner['learner_model_param']['base_score']).strip('[] '))
        self.base_margin = math.log(bs / (1.0 - bs))
        trees = learner['gradient_booster']['model']['trees']
        self.n_trees = len(trees)
        self.n_nodes = int(sum(len(t['left_children']) for t in trees))

        # ---- generate C source ----
        parts = ['#include <stddef.h>']
        parts.append(f'static float g_base = {self._f(self.base_margin)};')
        for i, t in enumerate(trees):
            parts.append(self._emit_tree(t, i))
        body = ['float predict_margin(const float* x) {', '  float s = g_base;']
        body += [f'  s += tree_{i}(x);' for i in range(len(trees))]
        body += ['  return s;', '}']
        parts.append('\n'.join(body))
        parts.append('int predict_label(const float* x) { return predict_margin(x) > 0.0f; }')
        parts.append('void predict_margin_batch(const float* X, int n, int d, float* out) {\n'
                     '  for (int i = 0; i < n; i++) out[i] = predict_margin(X + (size_t)i * (size_t)d);\n}')
        parts.append('void set_base(float b) { g_base = b; }')
        parts.append('float get_base(void) { return g_base; }')
        c_path  = os.path.join(workdir, f'{uid}.c')
        so_path = os.path.join(workdir, f'{uid}.so')
        with open(c_path, 'w') as fh:
            fh.write('\n'.join(parts) + '\n')

        # ---- compile + load ----
        cc = shutil.which('gcc') or shutil.which('cc') or shutil.which('clang')
        if cc is None:
            raise RuntimeError('No C compiler found (gcc/cc/clang). On Colab gcc is preinstalled.')
        subprocess.run([cc, opt_flag, '-shared', '-fPIC', '-o', so_path, c_path],
                       check=True, capture_output=True)
        self.json_path, self.c_path, self.so_path = model_json, c_path, so_path
        self.so_kb = os.path.getsize(so_path) / 1024.0
        self.c_kb  = os.path.getsize(c_path) / 1024.0
        self.lib = ctypes.CDLL(so_path)
        self.lib.predict_label.argtypes  = [ctypes.c_void_p]
        self.lib.predict_label.restype   = ctypes.c_int
        self.lib.predict_margin.argtypes = [ctypes.c_void_p]
        self.lib.predict_margin.restype  = ctypes.c_float
        self.lib.predict_margin_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.predict_margin_batch.restype  = None
        self.lib.set_base.argtypes = [ctypes.c_float]
        self.lib.set_base.restype  = None
        self.lib.get_base.argtypes = []
        self.lib.get_base.restype  = ctypes.c_float
        self.c_predict_label = self.lib.predict_label      # call as f(arr.ctypes.data)
        self.max_abs_margin_err = None

    @staticmethod
    def _f(v):
        return repr(float(np.float32(v))) + 'f'

    def _emit_tree(self, t, idx):
        left, right = t['left_children'], t['right_children']
        feat, cond  = t['split_indices'], t['split_conditions']
        lines = []
        def rec(n, d):
            pad = '  ' * d
            if left[n] == -1:                                   # leaf: split_conditions holds the leaf value
                lines.append(f'{pad}return {self._f(cond[n])};')
            else:
                lines.append(f'{pad}if (x[{int(feat[n])}] < {self._f(cond[n])}) {{')
                rec(left[n], d + 1)
                lines.append(f'{pad}}} else {{')
                rec(right[n], d + 1)
                lines.append(f'{pad}}}')
        rec(0, 1)
        return f'static inline float tree_{idx}(const float* x) {{\n' + '\n'.join(lines) + '\n}\n'

    def margins(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        self.lib.predict_margin_batch(X.ctypes.data, len(X), X.shape[1], out.ctypes.data)
        return out

    def predict(self, X):
        return (self.margins(X) > 0).astype(np.int8)

    def calibrate(self, booster, X, n=2000):
        """Align the base margin with XGBoost's own margins on n rows; returns (offset, max |margin error|)."""
        Xs  = np.ascontiguousarray(X[:n], dtype=np.float32)
        ref = booster.predict(xgb.DMatrix(Xs), output_margin=True)
        mine = self.margins(Xs)
        off = float(np.median(ref - mine))
        if abs(off) > 1e-4:
            self.lib.set_base(self.lib.get_base() + off)
            self.base_margin += off
            mine = self.margins(Xs)
        self.max_abs_margin_err = float(np.max(np.abs(ref - mine)))
        return off, self.max_abs_margin_err


# =============================================================================
# int8 TFLite export, real-int8 embeddings, deployed single-frame pipeline
# =============================================================================
def convert_to_int8_tflite(feature_extractor, X_calib_source, int8_path, n_calib=256):
    """Full-integer post-training quantisation, calibrated on the first n_calib training frames."""
    calib = np.asarray(X_calib_source[:n_calib], dtype=np.float32)
    def representative_data_gen():
        for frame in calib:
            yield [frame[np.newaxis, ...]]                       # (1,29,29,1)
    conv = tf.lite.TFLiteConverter.from_keras_model(feature_extractor)
    conv.optimizations             = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset    = representative_data_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type      = tf.float32
    conv.inference_output_type     = tf.float32
    blob = conv.convert()
    pathlib.Path(int8_path).write_bytes(blob)
    return int8_path


def tflite_embed_all(tflite_path, X):
    """Run the REAL int8 TFLite model frame by frame over X (N,29,29,1) -> (N,64) float32 embeddings."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_det = interp.get_output_details()[0]
    out_idx = out_det['index']
    dim     = int(out_det['shape'][-1])
    Xf      = np.asarray(X, dtype=np.float32)
    feats   = np.empty((len(Xf), dim), dtype=np.float32)
    for i in range(len(Xf)):
        interp.set_tensor(inp_idx, Xf[i:i+1])
        interp.invoke()
        feats[i] = interp.get_tensor(out_idx)[0]
    return feats


def make_deployed_pipeline(tflite_path, evaluator):
    """Single-frame pipeline: int8 TFLite CNN (default threads) -> compiled XGBoost -> label (0/1)."""
    interp = tf.lite.Interpreter(model_path=tflite_path)       # default threads
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    invoke  = interp.invoke
    c_label = evaluator.c_predict_label

    def _predict(x_single):
        interp.set_tensor(inp_idx, x_single)
        invoke()
        emb = interp.get_tensor(out_idx)                       # (1,64) float32
        return c_label(emb.ctypes.data)
    return _predict


# =============================================================================
# Noise-robust latency harness: BENCH_ROUNDS rounds x BENCH_REPS single-frame calls,
# garbage collection paused, perf_counter_ns clock; the MEDIAN round is reported.
# =============================================================================
def robust_latency(fns, x, reps=None, rounds=None, warmup=None):
    reps   = reps   or BENCH_REPS
    rounds = rounds or BENCH_ROUNDS
    warmup = warmup or BENCH_WARMUP
    samples = {k: [] for k in fns}
    gc.collect(); gc.disable()
    try:
        for _ in range(warmup):
            for f in fns.values():
                f(x)
        for _ in range(rounds):
            for k, f in fns.items():
                t0 = time.perf_counter_ns()
                for _ in range(reps):
                    f(x)
                samples[k].append((time.perf_counter_ns() - t0) / reps / 1e6)   # ms per call
    finally:
        gc.enable()
    out = {}
    for k, s in samples.items():
        s = np.asarray(s)
        out[k] = dict(median=float(np.median(s)), p25=float(np.percentile(s, 25)),
                      p75=float(np.percentile(s, 75)), min=float(s.min()),
                      max=float(s.max()), rounds=int(len(s)))
    return out


# =============================================================================
# Table helpers: print + CSV + PNG
# =============================================================================
def _fmt_cell(v):
    if isinstance(v, (float, np.floating)):
        return '-' if np.isnan(v) else f'{v:,.4f}'
    return str(v)

def show_table(df, title, fname, highlight_rows=None):
    df = df.reset_index(drop=True)
    print('\n' + title)
    print(df.to_string(index=False))
    df.to_csv(f'{OUT_DIR}/{fname}.csv', index=False)

    cell_text = [[_fmt_cell(v) for v in row] for row in df.values]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.55 * len(df.columns)), 0.42 * (len(df) + 2.2)))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=[str(c) for c in df.columns], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.45)
    tbl.auto_set_column_width(list(range(len(df.columns))))
    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor('#dfe7f2'); tbl[(0, j)].set_text_props(fontweight='bold')
    for r in (highlight_rows or []):
        for j in range(len(df.columns)):
            tbl[(r + 1, j)].set_facecolor('#d9f2d9')
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    fig.savefig(f'{OUT_DIR}/{fname}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {fname}.csv / {fname}.png')

print('Helpers ready.')



# ######################################################################
# 5. Train and evaluate the final model on all five datasets
# ######################################################################

# =============================================================================
# TRAIN + EVALUATE (all five datasets)
# =============================================================================
print('Ensuring replay dataset exists...')
synthesize_replay_dataset(DATASETS['Replay'])

results       = {}    # per-dataset metrics of the FINAL pipeline (int8 CNN + compiled XGBoost 100x6)
final_eval    = {}    # per-dataset CompiledXGB objects
tflite_paths  = {}    # per-dataset int8 TFLite files
dos_x1 = dos_X_small = dos_y_small = None

for name, path in DATASETS.items():
    print(f'\n[{name}] loading + building frames...')
    X, y, _, _ = load_and_build_frames(path, SAMPLE_MAIN)
    Xtr, Xte, ytr, yte = shuffle_split(X, y, seed=42)
    del X, y; gc.collect()

    # ---- 1) DCNN_Lite trained as a classifier ----
    print(f'[{name}] training DCNN_Lite...')
    _free()
    ds_tr = make_ds(Xtr, ytr, BATCH_SIZE, shuffle=True,  cache=True)
    ds_te = make_ds(Xte, yte, BATCH_SIZE, shuffle=False, cache=True)
    steps = len(ds_tr)
    cnn = build_dcnn_lite(EMBEDDING_DIM)
    cnn.compile(optimizer=keras.optimizers.Adam(make_cosine_lr(steps, EPOCHS_DCNN)),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    hist = cnn.fit(ds_tr, validation_data=ds_te, epochs=EPOCHS_DCNN, callbacks=_std_cbs(), verbose=2)
    cnn_train_sec = time.time() - t0

    # ---- 2) drop the softmax head -> 64-d feature extractor ----
    extractor = keras.Model(inputs=cnn.input, outputs=cnn.get_layer('feature_embedding').output,
                            name='DCNN_Lite_FeatureExtractor')
    feat_tr = extractor.predict(Xtr, batch_size=BATCH_SIZE, verbose=0)

    # ---- 3) XGBoost 100 trees x depth 6 on the training embeddings ----
    clf = XGBClassifier(n_estimators=XGB_N_TREES, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LR,
                        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                        n_jobs=-1, random_state=42)
    t1 = time.time()
    clf.fit(feat_tr, ytr)
    xgb_train_sec = time.time() - t1

    # ---- 4) int8 TFLite export ----
    tpath = convert_to_int8_tflite(extractor, Xtr, f'/tmp/final_hybrid_int8_{name}.tflite', n_calib=N_CALIB)
    tflite_paths[name] = tpath
    tflite_kb = os.path.getsize(tpath) / 1024.0

    # ---- 5) REAL int8 embeddings for every test frame ----
    t2 = time.time()
    feat_te_i8 = tflite_embed_all(tpath, Xte)
    print(f'[{name}] int8 embedding pass over {len(Xte):,} test frames: {time.time()-t2:.0f}s')

    # ---- 6) compiled XGBoost: build, calibrate, evaluate ----
    ev = CompiledXGB(clf, tag=f'final_{name}_{XGB_N_TREES}x{XGB_MAX_DEPTH}')
    _off, _err = ev.calibrate(clf.get_booster(), feat_te_i8)
    yhat  = ev.predict(feat_te_i8)
    agree = float((yhat == clf.predict(feat_te_i8)).mean())
    m = _metrics(yte, yhat)
    m.update(cnn_params=int(cnn.count_params()), cnn_epochs_run=len(hist.history['loss']),
             cnn_train_sec=cnn_train_sec, xgb_train_sec=xgb_train_sec,
             tflite_kb=tflite_kb, xgb_nodes=ev.n_nodes, xgb_so_kb=ev.so_kb,
             compiled_vs_xgb_agreement=agree, max_abs_margin_err=_err, n_test=len(yte))
    results[name]    = m
    final_eval[name] = ev

    print(f'[{name}] FNR={m["FNR"]:.4f}%  ER={m["ER"]:.4f}%  Precision={m["Precision"]:.4f}  '
          f'Recall={m["Recall"]:.4f}  F1={m["F1"]:.4f}')
    print(f'[{name}] compiled C vs xgboost.predict label agreement: {agree*100:.4f}%   '
          f'(trees={ev.n_trees}, nodes={ev.n_nodes:,}, TFLite={tflite_kb:.1f} KB)')

    if name == 'DoS':                                   # kept for the latency benchmark + consistency check
        dos_x1      = Xte[:1].astype(np.float32).copy()
        dos_X_small = Xte[:2000].copy()
        dos_y_small = yte[:2000].copy()

    del cnn, extractor, clf, ds_tr, ds_te, Xtr, Xte, ytr, yte, feat_tr, feat_te_i8, yhat
    _free()

mean_f1  = float(np.mean([r['F1']  for r in results.values()]))
mean_fnr = float(np.mean([r['FNR'] for r in results.values()]))
mean_er  = float(np.mean([r['ER']  for r in results.values()]))
print(f'\nFINAL MODEL - mean over 5 datasets:  F1 = {mean_f1:.4f}   FNR = {mean_fnr:.4f}%   ER = {mean_er:.4f}%')



# ######################################################################
# 6. Deployed-pipeline consistency check
# ######################################################################

# =============================================================================
# The single-frame deployed pipeline (int8 TFLite -> compiled XGBoost) must produce the
# same labels as the batch evaluation used for Table 1.  Checked on 2,000 DoS test frames.
# =============================================================================
ev_dos   = final_eval['DoS']
deployed = make_deployed_pipeline(tflite_paths['DoS'], ev_dos)

lab_pipe  = np.array([int(deployed(dos_X_small[i:i+1])) for i in range(len(dos_X_small))])
lab_batch = ev_dos.predict(tflite_embed_all(tflite_paths['DoS'], dos_X_small))
m_chk = _metrics(dos_y_small, lab_pipe)
print(f'Single-frame pipeline == batch evaluation labels on {len(lab_pipe):,} DoS frames: '
      f'{bool((lab_pipe == lab_batch).all())}')
print(f'Sanity metrics on this 2,000-frame subset: F1 = {m_chk["F1"]:.4f}   FNR = {m_chk["FNR"]:.3f}%  '
      f'(subset only - the full-test-set results are in Table 1)')



# ######################################################################
# 7. Latency
# ######################################################################

# =============================================================================
# LATENCY of the final pipeline
# =============================================================================
print(f'Timing final pipeline ({BENCH_ROUNDS} rounds x {BENCH_REPS} calls)...')
lat = robust_latency({'final': deployed}, dos_x1)['final']

lat_ms  = lat['median']
iqr_ms  = lat['p75'] - lat['p25']
meets   = lat_ms <= REALTIME_THRESHOLD_MS
print(f'Median latency : {lat_ms:.4f} ms/frame   (IQR {lat["p25"]:.4f}-{lat["p75"]:.4f} ms, '
      f'min {lat["min"]:.4f}, max {lat["max"]:.4f})')
print(f'vs. 0.5 ms budget: {lat_ms/REALTIME_THRESHOLD_MS:.2f}x   ->   '
      f'{"MEETS the real-time target" if meets else "ABOVE the real-time target"}')



# ######################################################################
# 8. Final results tables
# ######################################################################

# =============================================================================
# TABLE 1 - detection performance per dataset
# =============================================================================
rows = []
for ds, r in results.items():
    rows.append(dict(Dataset=ds, FNR_pct=r['FNR'], ER_pct=r['ER'],
                     Precision=r['Precision'], Recall=r['Recall'], F1=r['F1']))
df_perf = pd.DataFrame(rows)
mean_row = {c: df_perf[c].mean() for c in df_perf.columns if c != 'Dataset'}
df_perf = pd.concat([df_perf, pd.DataFrame([dict(Dataset='MEAN', **mean_row)])], ignore_index=True)
show_table(df_perf,
           f'Table 1 - Detection performance (int8 TFLite CNN + compiled XGBoost {XGB_N_TREES}x{XGB_MAX_DEPTH}, real int8 embeddings)',
           'table1_detection_performance', highlight_rows=[len(df_perf) - 1])

# =============================================================================
# TABLE 2 - latency and model footprint
# =============================================================================
ev_ = final_eval['DoS']
df_lat = pd.DataFrame([dict(
    Model=f'Hybrid Lite: int8 TFLite + compiled XGB {XGB_N_TREES}x{XGB_MAX_DEPTH}',
    CPU_ms_median=lat_ms, IQR_ms=iqr_ms, Min_ms=lat['min'],
    x_over_0_5ms=lat_ms / REALTIME_THRESHOLD_MS,
    Meets_0_5ms='YES' if meets else 'No',
    TFLite_KB=results['DoS']['tflite_kb'],
    XGB_nodes=float(ev_.n_nodes),
    XGB_compiled_KB_x86=ev_.so_kb)])
show_table(df_lat, 'Table 2 - Single-frame CPU latency and footprint of the final model (DoS model)',
           'table2_latency', highlight_rows=[0])

# =============================================================================
# Machine-readable summary
# =============================================================================
summary = dict(
    architecture='DCNN_Lite -> 64-d embedding -> int8 TFLite (default threads) -> compiled XGBoost',
    xgboost=dict(n_estimators=XGB_N_TREES, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LR,
                 subsample=0.8, colsample_bytree=0.8, nodes=int(ev_.n_nodes)),
    latency_ms=dict(median=lat_ms, p25=lat['p25'], p75=lat['p75'], min=lat['min'], max=lat['max'],
                    rounds=BENCH_ROUNDS, reps_per_round=BENCH_REPS, budget=REALTIME_THRESHOLD_MS,
                    meets_budget=bool(meets)),
    mean=dict(F1=mean_f1, FNR_pct=mean_fnr, ER_pct=mean_er),
    per_dataset={d: dict(FNR_pct=r['FNR'], ER_pct=r['ER'], Precision=r['Precision'], Recall=r['Recall'],
                         F1=r['F1'], compiled_vs_xgb_agreement=r['compiled_vs_xgb_agreement'],
                         cnn_epochs_run=r['cnn_epochs_run'], n_test=r['n_test'])
                 for d, r in results.items()},
    tflite_kb={d: r['tflite_kb'] for d, r in results.items()},
    hardware=dict(cpu_cores=os.cpu_count(), processor=platform.processor() or platform.machine()))
with open(f'{OUT_DIR}/final_results.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

SEP = '=' * 70
print('\n' + SEP)
print(f'  FINAL MODEL: DCNN_Lite -> int8 TFLite -> compiled XGBoost {XGB_N_TREES}x{XGB_MAX_DEPTH}')
print(SEP)
print(f'  Mean F1  = {mean_f1:.4f}    Mean FNR = {mean_fnr:.4f}%    Mean ER = {mean_er:.4f}%')
for d, r in results.items():
    print(f'    {d:7s} F1={r["F1"]:.4f}   FNR={r["FNR"]:.4f}%   ER={r["ER"]:.4f}%')
print(f'  Latency  = {lat_ms:.4f} ms/frame  (IQR {iqr_ms:.4f})   {lat_ms/REALTIME_THRESHOLD_MS:.2f}x of the 0.5 ms budget   '
      f'{"MEETS" if meets else "ABOVE"} target')
print(SEP)



# ######################################################################
# 9. Save model artifacts (for Objective 3)
# ######################################################################

art_dir = f'{OUT_DIR}/model_artifacts'
os.makedirs(art_dir, exist_ok=True)
for d in DATASETS:
    shutil.copy(tflite_paths[d],        f'{art_dir}/{d}_dcnn_lite_int8.tflite')
    shutil.copy(final_eval[d].json_path, f'{art_dir}/{d}_xgb_{XGB_N_TREES}x{XGB_MAX_DEPTH}.json')
    shutil.copy(final_eval[d].c_path,    f'{art_dir}/{d}_xgb_{XGB_N_TREES}x{XGB_MAX_DEPTH}_compiled.c')
print('Saved to', art_dir)
print(sorted(os.listdir(art_dir)))
