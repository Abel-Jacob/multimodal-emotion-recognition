"""
Multimodal Fusion Emotion Recognition
Speech (CNN+BiLSTM+Attention) + Text (DistilBERT) → Concatenation Fusion
"""

import subprocess, sys
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import librosa
except ImportError:
    _install("librosa")
try:
    import transformers
except ImportError:
    _install("transformers")

import os, glob, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import torch
from transformers import (DistilBertTokenizerFast, DistilBertModel)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
torch.manual_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPUs: {len(gpus)} | PyTorch device: {TORCH_DEVICE}")

BASE_DIR      = "/content/project"
TESS_ROOT     = "/content/tess"
MODEL_DIR     = "/content/project/models/fusion_pipeline"
RESULTS_AT    = os.path.join(BASE_DIR, "Results", "accuracy_tables")
RESULTS_PL    = os.path.join(BASE_DIR, "Results", "plots")
for d in [RESULTS_AT, RESULTS_PL, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

SR         = 22050
DURATION   = 3.0
N_MFCC     = 40
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
MAX_LEN_SP = 128
MAX_LEN_TX = 32
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
EMOTIONS   = ["angry","disgust","fear","happy","neutral","ps","sad"]
DISTILBERT  = "distilbert-base-uncased"

def extract_speech_features(audio, sr):
    target = int(DURATION * sr)
    audio = np.pad(audio, (0, max(0, target - len(audio))))[:target]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH), ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    sc = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    feat = np.vstack([mfcc, delta, mel, chroma, sc, zcr]).T
    feat = np.pad(feat, ((0, max(0, MAX_LEN_SP - feat.shape[0])), (0, 0)))[:MAX_LEN_SP, :180]
    return ((feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)).astype(np.float32)

def load_tess(root_dir):
    records = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                fp = os.path.join(root, f)
                # Extract emotion from filename: e.g. YAF_back_angry.wav
                fname = os.path.splitext(f.lower())[0]
                parts = fname.split("_")
                emotion = parts[-1]
                if emotion in EMOTIONS:
                    word = " ".join(parts[1:-1]) if len(parts) >= 3 else "word"
                    records.append({"path": fp, "text": word, "emotion": emotion})
    if not records:
        raise FileNotFoundError(f"No valid .wav files found in {root_dir}.")
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} samples"); return df

class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)
    def call(self, x):
        score = self.V(self.W(x))
        weight = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weight * x, axis=1)

def build_speech_embedding_model(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Reshape((*input_shape, 1))(inp)
    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2,1))(x)
    x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    emb = AttentionLayer(units=64)(x)
    out = layers.Dense(128, activation="relu")(emb)
    return models.Model(inp, out)

def get_text_embeddings(texts, tokenizer, bert_model):
    bert_model.eval(); embs = []
    for i in range(0, len(texts), 64):
        batch = texts[i:i+64]
        enc = tokenizer(batch, max_length=MAX_LEN_TX, padding="max_length", truncation=True, return_tensors="pt").to(TORCH_DEVICE)
        with torch.no_grad(): out = bert_model(**enc)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)

def build_fusion_model(sp_dim, tx_dim, num_classes):
    sp_in = layers.Input(shape=(sp_dim,))
    tx_in = layers.Input(shape=(tx_dim,))
    tx_p = layers.Dense(128, activation="relu")(tx_in)
    fused = layers.Concatenate()([sp_in, tx_p])
    x = layers.Dense(256, activation="relu")(fused)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model([sp_in, tx_in], out)

def main():
    df = load_tess(TESS_ROOT)
    le = LabelEncoder(); df["label"] = le.fit_transform(df["emotion"])
    num_classes = len(le.classes_)
    np.save(os.path.join(MODEL_DIR, "fusion_label_classes.npy"), le.classes_)
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"])

    def get_sp_data(df_split):
        X, y = [], []
        for _, r in df_split.iterrows():
            try:
                a, s = librosa.load(r["path"], sr=SR)
                X.append(extract_speech_features(a, s)); y.append(r["label"])
            except: continue
        return np.array(X), np.array(y)

    print("Processing speech features...")
    X_sp_train, y_train = get_sp_data(train_df)
    X_sp_val, y_val = get_sp_data(val_df)

    print("Processing text embeddings...")
    tok = DistilBertTokenizerFast.from_pretrained(DISTILBERT)
    bert = DistilBertModel.from_pretrained(DISTILBERT).to(TORCH_DEVICE)
    E_tx_train = get_text_embeddings(list(train_df["text"]), tok, bert)
    E_tx_val = get_text_embeddings(list(val_df["text"]), tok, bert)

    sp_embedder = build_speech_embedding_model(X_sp_train.shape[1:])
    fusion_model = build_fusion_model(128, 768, num_classes)

    raw_sp = layers.Input(shape=X_sp_train.shape[1:])
    raw_tx = layers.Input(shape=(768,))
    comb_out = fusion_model([sp_embedder(raw_sp), raw_tx])
    model = models.Model([raw_sp, raw_tx], comb_out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Starting training...")
    model.fit([X_sp_train, E_tx_train], y_train, validation_data=([X_sp_val, E_tx_val], y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save(os.path.join(MODEL_DIR, "fusion_model.h5"))
    print("Fusion training complete.")

if __name__ == "__main__":
    main();
