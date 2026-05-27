"""
Text Emotion Recognition — Test / Inference
Loads saved DistilBERT model, runs inference, generates:
- Confusion matrix, classification report, ROC curve
- Attention visualisation (token weights)
- Prediction CSV
"""

import os, glob, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification)
from tensorflow.keras.utils import to_categorical

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR   = "/content/project"
TESS_DIR   = os.environ.get("TESS_DIR", os.path.join(BASE_DIR, "tess"))
MODEL_DIR  = "/content/project/models/text_pipeline"
RESULTS_AT = os.path.join(BASE_DIR, "Results", "accuracy_tables")
RESULTS_PL = os.path.join(BASE_DIR, "Results", "plots")
for d in [RESULTS_AT, RESULTS_PL]:
    os.makedirs(d, exist_ok=True)

MAX_LEN  = 32
BATCH_SIZE = 64
EMOTIONS   = ["angry","disgust","fear","happy","neutral","ps","sad"]

# ── TESS text loading ──────────────────────────────────────────────────────
def load_tess_text(tess_dir):
    records = []
    wav_files = glob.glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files at {tess_dir}")
    for fp in wav_files:
        fname   = os.path.splitext(os.path.basename(fp))[0].lower()
        folder  = os.path.basename(os.path.dirname(fp)).lower()
        emotion = folder.split("_")[-1]
        if emotion not in EMOTIONS:
            continue
        parts = fname.split("_")
        word  = " ".join(parts[1:-1]) if len(parts) >= 3 else parts[0]
        records.append({"text": word, "emotion": emotion})
    return pd.DataFrame(records)

class EmotionTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts=texts; self.labels=labels
        self.tokenizer=tokenizer; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Attention visualisation ────────────────────────────────────────────────
def visualize_attention(model, tokenizer, texts, true_labels, pred_labels, le, n=6):
    """Plot token attention weights from last layer's first head."""
    model.eval()
    fig, axes = plt.subplots(2, n//2, figsize=(20, 8))
    axes = axes.flatten()
    chosen = random.sample(range(len(texts)), min(n, len(texts)))
    for i, idx in enumerate(chosen):
        ax = axes[i]
        enc = tokenizer(texts[idx], max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_attentions=True)
        # last transformer layer, first head, CLS token attention
        attn = out.attentions[-1][0, 0, 0, :].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        # filter PAD
        valid = [(t, a) for t, a in zip(tokens, attn) if t not in ["[PAD]"]][:12]
        toks, weights = zip(*valid) if valid else ([], [])
        ax.bar(range(len(toks)), weights, color="steelblue")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=7)
        tl = le.inverse_transform([true_labels[idx]])[0]
        pl = le.inverse_transform([pred_labels[idx]])[0]
        ax.set_title(f"True:{tl} | Pred:{pl}", fontsize=8)
        ax.set_ylabel("Attention")
    plt.suptitle("DistilBERT Attention Weights (CLS token, last layer)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, "text_attention_viz.png"), dpi=150)
    plt.close()
    print("Attention visualisation saved.")

# ── Confusion / ROC ────────────────────────────────────────────────────────
def plot_confusion(cm, labels, prefix):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
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
    print("TEXT EMOTION RECOGNITION — Testing")
    print("=" * 60)

    model_path  = os.path.join(MODEL_DIR, "text_model")
    labels_path = os.path.join(MODEL_DIR, "text_label_classes.npy")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    le = LabelEncoder()
    le.classes_ = np.load(labels_path, allow_pickle=True)
    num_classes  = len(le.classes_)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(
        model_path, num_labels=num_classes).to(DEVICE)
    print(f"Model loaded. Classes: {list(le.classes_)}")

    df = load_tess_text(TESS_DIR)
    df["label"] = le.transform(df["emotion"])
    _, test_df  = train_test_split(df, test_size=0.15, random_state=SEED,
                                    stratify=df["label"])
    # Same seed as train → disjoint test set guaranteed
    print(f"Test samples: {len(test_df)}")

    test_ds  = EmotionTextDataset(list(test_df["text"]), list(test_df["label"]),
                                   tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────
    model.eval()
    all_logits = []
    y_true     = []
    with torch.no_grad():
        for batch in test_loader:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            out   = model(input_ids=ids, attention_mask=mask)
            all_logits.append(out.logits.cpu().numpy())
            y_true.extend(batch["label"].numpy())

    logits     = np.concatenate(all_logits)
    y_pred_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.array(y_true)
    y_true_bin  = to_categorical(y_true, num_classes)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, le.classes_, "text")
    plot_roc(y_true_bin, y_pred_prob, le.classes_, "text")

    # ── Attention visualisation ────────────────────────────────────────────
    try:
        visualize_attention(model, tokenizer,
                            list(test_df["text"]), y_true, y_pred, le, n=6)
    except Exception as e:
        print(f"Attention viz skipped: {e}")

    # ── Save metrics ──────────────────────────────────────────────────────
    rows = [{"class": lbl, **vals}
            for lbl, vals in report.items() if isinstance(vals, dict)]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_AT, "text_test_metrics.csv"), index=False)

    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "pred_label": le.inverse_transform(y_pred),
        "correct":    y_true == y_pred,
        **{f"prob_{c}": y_pred_prob[:, i] for i, c in enumerate(le.classes_)},
    })
    pred_df.to_csv(os.path.join(RESULTS_AT, "text_test_predictions.csv"), index=False)
    print("Results saved.")

    print("\nText pipeline testing COMPLETE.")

if __name__ == "__main__":
    main()


main()
