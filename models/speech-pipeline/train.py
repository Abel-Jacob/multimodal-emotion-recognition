"""
Speech Emotion Recognition — CNN + BiLSTM + Attention
Dataset: TESS (Toronto Emotional Speech Set)
Self-contained: preprocessing, augmentation, feature extraction,
model creation, training, evaluation, plotting, saving.
"""

# ── Colab: install deps if needed ──────────────────────────────────────────
import subprocess, sys
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import librosa
except ImportError:
    _install("librosa")
try:
    import soundfile
except ImportError:
    _install("soundfile")

# ── Imports ────────────────────────────────────────────────────────────────
import os, glob, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.utils import to_categorical

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ── GPU config ─────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
print(f"GPUs available: {len(gpus)}")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/content/project"
TESS_DIR   = os.environ.get("TESS_DIR", os.path.join(BASE_DIR, "tess"))
MODEL_DIR  = "/content/project/models/speech_pipeline"
RESULTS_AT = os.path.join(BASE_DIR, "Results", "accuracy_tables")
RESULTS_PL = os.path.join(BASE_DIR, "Results", "plots")
for d in [RESULTS_AT, RESULTS_PL, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
SR         = 22050
DURATION   = 3.0        # seconds per clip
N_MFCC     = 40
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
MAX_LEN    = 128        # time frames
BATCH_SIZE = 32
EPOCHS     = 60
LR         = 1e-3
EMOTIONS   = ["angry","disgust","fear","happy","neutral","ps","sad"]

# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(audio, sr):
    """
    Returns array shape (MAX_LEN, 180):
      40 MFCC + 40 delta MFCC + 128 mel + 12 chroma + 7 spectral_contrast + 1 ZCR + ??? → 228
    We keep first 180 dims for fixed size.
    """
    # ── pad / trim ─────────────────────────────────────────────────────────
    target = int(DURATION * sr)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    # ── MFCC (40) ──────────────────────────────────────────────────────────
    mfcc  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)          # (40, T)
    delta = librosa.feature.delta(mfcc)                                         # (40, T)

    # ── Mel spectrogram (128) ───────────────────────────────────────────────
    mel   = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH) # (128, T)
    mel   = librosa.power_to_db(mel, ref=np.max)

    # ── Chroma (12) ────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)                 # (12, T)

    # ── Spectral contrast (7) ──────────────────────────────────────────────
    sc     = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH)           # (7, T)

    # ── ZCR (1) ────────────────────────────────────────────────────────────
    zcr    = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)   # (1, T)

    # ── Stack → (228, T) → transpose → (T, 228) ────────────────────────────
    feat = np.vstack([mfcc, delta, mel, chroma, sc, zcr])  # (228, T)
    feat = feat.T                                           # (T, 228)

    # ── Pad/trim time axis ─────────────────────────────────────────────────
    if feat.shape[0] < MAX_LEN:
        feat = np.pad(feat, ((0, MAX_LEN - feat.shape[0]), (0, 0)))
    else:
        feat = feat[:MAX_LEN, :]

    # ── Normalise per feature ──────────────────────────────────────────────
    mean = feat.mean(axis=0, keepdims=True)
    std  = feat.std(axis=0, keepdims=True) + 1e-8
    feat = (feat - mean) / std

    return feat[:, :180]   # fixed 180 features

# ── Augmentation ───────────────────────────────────────────────────────────
def augment(audio, sr):
    """Apply one of 4 augmentations randomly."""
    choice = random.randint(0, 3)
    if choice == 0:                       # noise injection
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
    elif choice == 1:                     # pitch shift
        n_steps = random.uniform(-2, 2)
        audio   = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    elif choice == 2:                     # time stretch
        rate  = random.uniform(0.85, 1.15)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    elif choice == 3:                     # amplitude scaling
        audio = audio * random.uniform(0.7, 1.3)
    return audio

# ── Dataset loading ────────────────────────────────────────────────────────
def load_tess(tess_dir):
    """Walk TESS folder structure, load audio + labels."""
    records = []
    # TESS layout: <tess_dir>/<FolderName>/<actor>_<word>_<emotion>.wav
    # Folder names contain emotion in their name, e.g. "OAF_angry"
    wav_files = glob.glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)
    if len(wav_files) == 0:
        raise FileNotFoundError(
            f"No .wav files found under {tess_dir}. "
            "Set TESS_DIR env variable or place dataset at project/tess/"
        )
    for fp in wav_files:
        folder = os.path.basename(os.path.dirname(fp)).lower()
        # emotion is last segment after last underscore in folder name
        emotion = folder.split("_")[-1]
        if emotion not in EMOTIONS:
            continue
        records.append({"path": fp, "emotion": emotion})
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} files | emotions: {df['emotion'].value_counts().to_dict()}")
    return df

# ── Build feature matrix ───────────────────────────────────────────────────
def build_dataset(df, augment_train=True, aug_ratio=0.5):
    X, y = [], []
    for _, row in df.iterrows():
        try:
            audio, sr = librosa.load(row["path"], sr=SR, mono=True)
            # trim leading/trailing silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            feat = extract_features(audio, sr)
            X.append(feat)
            y.append(row["emotion"])
            # augmentation for training data
            if augment_train and random.random() < aug_ratio:
                aug_audio = augment(audio, sr)
                aug_feat  = extract_features(aug_audio, sr)
                X.append(aug_feat)
                y.append(row["emotion"])
        except Exception as e:
            print(f"Skipping {row['path']}: {e}")
    return np.array(X, dtype=np.float32), np.array(y)

# ── Attention layer ────────────────────────────────────────────────────────
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

# ── Model definition ───────────────────────────────────────────────────────
def build_model(input_shape, num_classes):
    """CNN + BiLSTM + Attention classifier."""
    inp = layers.Input(shape=input_shape, name="mfcc_input")

    # ── CNN block ──────────────────────────────────────────────────────────
    x = layers.Reshape((*input_shape, 1))(inp)          # add channel dim
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,1))(x)
    x = layers.Dropout(0.25)(x)

    # ── Merge spatial dims into features ───────────────────────────────────
    # shape after pool3: (batch, T//8, F//4, 128) → reshape to (batch, T', feat)
    t_dim  = x.shape[1]
    f_dim  = x.shape[2]
    ch_dim = x.shape[3]
    x = layers.Reshape((t_dim, f_dim * ch_dim))(x)

    # ── BiLSTM ─────────────────────────────────────────────────────────────
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                          dropout=0.3, recurrent_dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True,
                                          dropout=0.3, recurrent_dropout=0.2))(x)

    # ── Attention ──────────────────────────────────────────────────────────
    x = AttentionLayer(units=64)(x)

    # ── Classifier head ────────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = models.Model(inp, out, name="CNN_BiLSTM_Attention")
    return model

# ── Plot helpers ───────────────────────────────────────────────────────────
def plot_history(history, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="train acc")
    axes[0].plot(history.history["val_accuracy"], label="val acc")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history.history["loss"],     label="train loss")
    axes[1].plot(history.history["val_loss"], label="val loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, f"{prefix}_training_curves.png"), dpi=150)
    plt.close()

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
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{lbl} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{prefix} ROC Curve"); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, f"{prefix}_roc_curve.png"), dpi=150)
    plt.close()

def save_metrics_csv(report_dict, prefix):
    rows = []
    for label, vals in report_dict.items():
        if isinstance(vals, dict):
            rows.append({"class": label, **vals})
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_AT, f"{prefix}_metrics.csv")
    df.to_csv(path, index=False)
    print(f"Metrics saved → {path}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SPEECH EMOTION RECOGNITION — Training")
    print("=" * 60)

    # ── Load dataset ───────────────────────────────────────────────────────
    df = load_tess(TESS_DIR)

    # ── Encode labels ──────────────────────────────────────────────────────
    le          = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    # ── Train / val / test split ───────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED,
                                          stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=SEED,
                                          stratify=train_df["label"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Feature extraction ─────────────────────────────────────────────────
    print("Extracting features (train) …")
    X_train, y_train_str = build_dataset(train_df, augment_train=True)
    print("Extracting features (val) …")
    X_val,   y_val_str   = build_dataset(val_df,   augment_train=False)
    print("Extracting features (test) …")
    X_test,  y_test_str  = build_dataset(test_df,  augment_train=False)

    y_train = to_categorical(le.transform(y_train_str), num_classes)
    y_val   = to_categorical(le.transform(y_val_str),   num_classes)
    y_test  = to_categorical(le.transform(y_test_str),  num_classes)

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    # ── Class weights ──────────────────────────────────────────────────────
    cw = compute_class_weight("balanced", classes=np.arange(num_classes),
                               y=np.argmax(y_train, axis=1))
    class_weight = dict(enumerate(cw))

    # ── Build model ────────────────────────────────────────────────────────
    model = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # ── Callbacks ──────────────────────────────────────────────────────────
    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=6, min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "speech_model_best.h5"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # ── Training ───────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=1,
    )

    # ── Save final model ───────────────────────────────────────────────────
    model.save(os.path.join(MODEL_DIR, "speech_model.h5"))
    np.save(os.path.join(MODEL_DIR, "label_classes.npy"), le.classes_)
    print("Model saved.")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_history(history, "speech")

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, le.classes_, "speech")
    plot_roc(y_test, y_pred_prob, le.classes_, "speech")
    save_metrics_csv(report, "speech")

    # ── Save test predictions ──────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "pred_label": le.inverse_transform(y_pred),
        "correct":    y_true == y_pred,
    })
    pred_df.to_csv(os.path.join(RESULTS_AT, "speech_predictions.csv"), index=False)
    print("Predictions saved.")

    print("\nSpeech pipeline training COMPLETE.")

if __name__ == "__main__":
    main()


main()
