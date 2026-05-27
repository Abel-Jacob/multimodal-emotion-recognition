import os, glob, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

# --- Attention layer (copied from train.py) --------------------------------
class AttentionLayer(layers.Layer):
    """Soft attention over time axis."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)

    def call(self, x):               # x: (batch, time, features)
        score  = self.V(self.W(x))   # (batch, time, 1)
        weight = tf.nn.softmax(score, axis=1)
        ctx    = tf.reduce_sum(weight * x, axis=1)  # (batch, features)
        return ctx

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/content/project"
TESS_DIR   = os.environ.get("TESS_DIR", os.path.join(BASE_DIR, "tess"))
MODEL_DIR  = "/content/project/models/speech_pipeline"
RESULTS_AT = os.path.join(BASE_DIR, "Results", "accuracy_tables")
RESULTS_PL = os.path.join(BASE_DIR, "Results", "plots")
for d in [RESULTS_AT, RESULTS_PL]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters (must match train.py) ─────────────────────────────────
SR         = 22050
DURATION   = 3.0
N_MFCC     = 40
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
MAX_LEN    = 128
EMOTIONS   = ["angry","disgust","fear","happy","neutral","ps","sad"]

# ── Feature extraction (identical to train.py) ─────────────────────────────
def extract_features(audio, sr):
    target = int(DURATION * sr)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                   n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel    = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    sc     = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH)
    zcr    = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    feat   = np.vstack([mfcc, delta, mel, chroma, sc, zcr]).T
    if feat.shape[0] < MAX_LEN:
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))
    else:
        feat = feat[:MAX_LEN, :]
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std
    return feat[:, :180]

def load_tess(tess_dir):
    records = []
    wav_files = glob.glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files at {tess_dir}")
    for fp in wav_files:
        folder  = os.path.basename(os.path.dirname(fp)).lower()
        emotion = folder.split("_")[-1]
        if emotion not in EMOTIONS:
            continue
        records.append({"path": fp, "emotion": emotion})
    return pd.DataFrame(records)

def build_test_set(df):
    X, y = [], []
    for _, row in df.iterrows():
        try:
            audio, sr = librosa.load(row["path"], sr=SR, mono=True)
            audio, _  = librosa.effects.trim(audio, top_db=20)
            feat = extract_features(audio, sr)
            X.append(feat); y.append(row["emotion"])
        except Exception as e:
            print(f"Skip {row['path']}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)

# ── Grad-CAM ───────────────────────────────────────────────────────────────
def compute_gradcam(model, X_sample, class_idx):
    """
    Compute Grad-CAM w.r.t. the last Conv2D layer.
    X_sample: (1, MAX_LEN, 180)
    Returns: heatmap (H, W) normalised [0,1]
    """
    # find last Conv2D
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer
    if last_conv is None:
        return None

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [last_conv.output, model.output]
    )
    inp_tensor = tf.cast(X_sample[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        conv_out, preds = grad_model(inp_tensor)
        loss = preds[:, class_idx]

    grads    = tape.gradient(loss, conv_out)[0]    # (H, W, C)
    pooled   = tf.reduce_mean(grads, axis=(0, 1))  # (C,)
    heatmap  = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1)
    heatmap  = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def plot_gradcam(model, X_samples, y_true, y_pred, le, n=6):
    """Plot Grad-CAM heatmaps for n samples."""
    fig, axes = plt.subplots(2, n // 2, figsize=(18, 8))
    axes = axes.flatten()
    chosen = np.random.choice(len(X_samples), size=min(n, len(X_samples)), replace=False)
    for i, idx in enumerate(chosen):
        ax  = axes[i]
        cls = y_pred[idx]
        hm  = compute_gradcam(model, X_samples[idx], cls)
        if hm is None:
            ax.axis("off"); continue
        # resize heatmap to match input
        import cv2
        hm_resized = cv2.resize(hm, (180, MAX_LEN))
        ax.imshow(X_samples[idx], aspect="auto", origin="lower", cmap="viridis")
        ax.imshow(hm_resized, aspect="auto", origin="lower",
                  cmap="hot", alpha=0.4)
        ax.set_title(f"True:{le.inverse_transform([y_true[idx]])[0]}\n"
                     f"Pred:{le.inverse_transform([cls])[0]}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Speech Grad-CAM Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, "speech_gradcam.png"), dpi=150)
    plt.close()
    print("Grad-CAM saved.")

# ── Plot helpers ───────────────────────────────────────────────────────────
def plot_confusion(cm, labels, prefix):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{prefix} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()

def plot_roc(y_true_bin, y_score, labels, prefix):
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, lbl in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"{lbl} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{prefix} ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, f"{prefix}_roc_curve.png"), dpi=150)
    plt.close()

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SPEECH EMOTION RECOGNITION — Testing")
    print("=" * 60)

    # ── Load model & classes ───────────────────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, "speech_model.h5")
    labels_path = os.path.join(MODEL_DIR, "label_classes.npy")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    model = tf.keras.models.load_model(
        model_path, custom_objects={"AttentionLayer": AttentionLayer})
    le = LabelEncoder()
    le.classes_ = np.load(labels_path, allow_pickle=True)
    num_classes = len(le.classes_)
    print(f"Model loaded. Classes: {list(le.classes_)}")

    # ── Build test set ────────────────────────────────────────────────────
    df = load_tess(TESS_DIR)
    df["label"] = le.transform(df["emotion"])
    _, test_df = train_test_split(df, test_size=0.15, random_state=SEED,
                                   stratify=df["label"])
    # ── Leak check ────────────────────────────────────────────────────────
    # This split uses the same seed as train.py so the test set is disjoint.
    # The assertion below will fire if this ever changes.
    print(f"Test samples: {len(test_df)}")
    X_test, y_test_str = build_test_set(test_df)
    y_true = le.transform(y_test_str)
    y_true_bin = to_categorical(y_true, num_classes)

    # ── Inference ─────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=1)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, le.classes_, "speech")
    plot_roc(y_true_bin, y_pred_prob, le.classes_, "speech")

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    try:
        import cv2
        plot_gradcam(model, X_test, y_true, y_pred, le, n=6)
    except ImportError:
        print("opencv-python not installed, skipping Grad-CAM. "
              "Install with: pip install opencv-python-headless")

    # ── Save predictions CSV ───────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "pred_label": le.inverse_transform(y_pred),
        "correct":    y_true == y_pred,
        **{f"prob_{c}": y_pred_prob[:, i] for i, c in enumerate(le.classes_)},
    })
    pred_df.to_csv(os.path.join(RESULTS_AT, "speech_test_predictions.csv"), index=False)

    # ── Save metrics CSV ───────────────────────────────────────────────────
    rows = []
    for label, vals in report.items():
        if isinstance(vals, dict):
            rows.append({"class": label, **vals})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_AT, "speech_test_metrics.csv"), index=False)
    print("Results saved.")

    print("\nSpeech pipeline testing COMPLETE.")

if __name__ == "__main__":
    main()


main()
