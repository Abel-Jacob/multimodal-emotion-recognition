"""
Multimodal Fusion — Testing
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import torch
from transformers import DistilBertTokenizerFast, DistilBertModel

import tensorflow as tf


# ==========================================
# SETTINGS
# ==========================================
SEED = 42

TESS_DIR = "/content/tess"
MODEL_DIR = "/content/project/models/fusion_pipeline"

TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Torch device:", TORCH_DEVICE)


# ==========================================
# CUSTOM ATTENTION LAYER
# ==========================================
class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)

        self.W = tf.keras.layers.Dense(
            units,
            activation="tanh"
        )

        self.V = tf.keras.layers.Dense(1)

    def call(self, x):

        score = self.V(self.W(x))

        weight = tf.nn.softmax(
            score,
            axis=1
        )

        return tf.reduce_sum(
            weight * x,
            axis=1
        )


# ==========================================
# FEATURE EXTRACTION
# MUST MATCH TRAINING EXACTLY
# ==========================================
def extract_speech_features(audio, sr):

    target = int(3.0 * sr)

    audio = np.pad(
        audio,
        (0, max(0, target - len(audio)))
    )[:target]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    delta = librosa.feature.delta(mfcc)

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        ),
        ref=np.max
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )

    sc = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        hop_length=512
    )

    feat = np.vstack([
        mfcc,
        delta,
        mel,
        chroma,
        sc,
        zcr
    ]).T

    feat = np.pad(
        feat,
        ((0, max(0, 128 - feat.shape[0])), (0, 0))
    )[:128, :180]

    feat = (
        (feat - feat.mean(axis=0)) /
        (feat.std(axis=0) + 1e-8)
    )

    return feat.astype(np.float32)


# ==========================================
# MAIN
# ==========================================
def main():

    # ======================================
    # LOAD MODEL
    # ======================================
    model_path = os.path.join(
        MODEL_DIR,
        "fusion_model.h5"
    )

    if not os.path.exists(model_path):
        print("Fusion model not found.")
        return

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AttentionLayer": AttentionLayer
        },
        compile=False
    )

    print("Fusion model loaded.")

    # ======================================
    # LOAD LABELS
    # ======================================
    le = LabelEncoder()

    le.classes_ = np.load(
        os.path.join(
            MODEL_DIR,
            "fusion_label_classes.npy"
        ),
        allow_pickle=True
    )

    print("Classes:", le.classes_)

    # ======================================
    # FIND WAV FILES
    # ======================================
    wav_files = []

    for root, _, files in os.walk(TESS_DIR):

        for f in files:

            if f.lower().endswith(".wav"):

                wav_files.append(
                    os.path.join(root, f)
                )

    print("WAV files found:", len(wav_files))

    if len(wav_files) == 0:
        print("No WAV files found.")
        return

    # ======================================
    # BUILD DATAFRAME
    # ======================================
    records = []

    for fp in wav_files:

        folder = os.path.basename(
            os.path.dirname(fp)
        ).lower()

        emotion = (
            folder.split("_")[-1]
            if "_" in folder
            else folder
        )

        if emotion in le.classes_:

            records.append({
                "path": fp,
                "text": "word",
                "emotion": emotion
            })

    df = pd.DataFrame(records)

    print("Dataset size:", len(df))

    # ======================================
    # TEST SPLIT
    # ======================================
    _, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["emotion"]
    )

    print("Test samples:", len(test_df))

    # ======================================
    # LOAD DISTILBERT
    # ======================================
    tok = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    bert = DistilBertModel.from_pretrained(
        "distilbert-base-uncased"
    ).to(TORCH_DEVICE)

    # ======================================
    # PREPARE SPEECH FEATURES
    # ======================================
    X_sp = []
    y_true = []

    print("Extracting speech features...")

    for _, r in test_df.iterrows():

        audio, sr = librosa.load(
            r["path"],
            sr=22050
        )

        feat = extract_speech_features(
            audio,
            sr
        )

        X_sp.append(feat)

        y_true.append(
            le.transform(
                [r["emotion"]]
            )[0]
        )

    X_sp = np.array(X_sp)

    print("Speech shape:", X_sp.shape)

    # ======================================
    # TEXT EMBEDDINGS
    # ======================================
    X_tx = []

    print("Generating text embeddings...")

    for i in range(0, len(test_df), 64):

        batch_size = len(
            test_df[i:i+64]
        )

        batch = ["word"] * batch_size

        enc = tok(
            batch,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(TORCH_DEVICE)

        with torch.no_grad():

            out = bert(**enc)

        X_tx.append(
            out.last_hidden_state[:, 0, :]
            .cpu()
            .numpy()
        )

    X_tx = np.vstack(X_tx)

    print("Text shape:", X_tx.shape)

    # ======================================
    # PREDICTION
    # ======================================
    print("Running prediction...")

    y_pred = model.predict(
        [X_sp, X_tx]
    ).argmax(axis=1)

    # ======================================
    # RESULTS
    # ======================================
    acc = accuracy_score(
        y_true,
        y_pred
    )

    print("\nAccuracy:", acc)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(le.classes_)),
            target_names=le.classes_,
            zero_division=0
        )
    )


# ==========================================
# RUN
# ==========================================
main()
