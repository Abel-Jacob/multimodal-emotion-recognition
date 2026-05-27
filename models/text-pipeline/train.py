"""
Text Emotion Recognition — DistilBERT Fine-tuning
Dataset: TESS transcript labels extracted from filenames
Self-contained: preprocessing, tokenisation, model, training,
evaluation, plotting, saving.
"""

import subprocess, sys
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import transformers
except ImportError:
    _install("transformers")
try:
    import torch
except ImportError:
    _install("torch")

import os, glob, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/content/project"
TESS_DIR   = os.environ.get("TESS_DIR", os.path.join(BASE_DIR, "tess"))
MODEL_DIR  = "/content/project/models/text_pipeline"
RESULTS_AT = os.path.join(BASE_DIR, "Results", "accuracy_tables")
RESULTS_PL = os.path.join(BASE_DIR, "Results", "plots")
for d in [RESULTS_AT, RESULTS_PL, os.path.join(MODEL_DIR, "text_model")]:
    os.makedirs(d, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────
MAX_LEN    = 32
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 2e-5
WARMUP_P   = 0.1
EMOTIONS   = ["angry","disgust","fear","happy","neutral","ps","sad"]
MODEL_NAME = "distilbert-base-uncased"

# ── TESS text extraction ───────────────────────────────────────────────────
def load_tess_text(tess_dir):
    """
    TESS filenames: <Actor>_<word>_<emotion>.wav
    We extract the spoken word(s) as the 'text transcript'.
    """
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
        # parse spoken word: e.g. "OAF_back_angry" → word = "back"
        parts = fname.split("_")
        if len(parts) >= 3:
            word = " ".join(parts[1:-1])
        else:
            word = parts[0]
        records.append({"text": word, "emotion": emotion})
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} text samples")
    return df

# ── PyTorch Dataset ────────────────────────────────────────────────────────
class EmotionTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length   = self.max_len,
            padding      = "max_length",
            truncation   = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Plot helpers ───────────────────────────────────────────────────────────
def plot_history(train_accs, val_accs, train_losses, val_losses, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_accs, label="train acc")
    axes[0].plot(val_accs,   label="val acc")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(train_losses, label="train loss")
    axes[1].plot(val_losses,   label="val loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PL, f"{prefix}_training_curves.png"), dpi=150)
    plt.close()

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

# ── Training epoch ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        preds = out.logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * len(labels)
        total         += len(labels)
    return total_loss / total, total_correct / total

def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)
            out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            preds = out.logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_loss    += out.loss.item() * len(labels)
            total         += len(labels)
            all_logits.append(out.logits.cpu().numpy())
    logits = np.concatenate(all_logits)
    return total_loss / total, total_correct / total, logits

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("TEXT EMOTION RECOGNITION — Training (DistilBERT)")
    print("=" * 60)

    df = load_tess_text(TESS_DIR)

    le          = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED,
                                          stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=SEED,
                                          stratify=train_df["label"])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    np.save(os.path.join(MODEL_DIR, "text_label_classes.npy"), le.classes_)

    # ── Tokeniser ─────────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # ── Datasets / Loaders ────────────────────────────────────────────────
    train_ds = EmotionTextDataset(list(train_df["text"]), list(train_df["label"]),
                                   tokenizer, MAX_LEN)
    val_ds   = EmotionTextDataset(list(val_df["text"]),   list(val_df["label"]),
                                   tokenizer, MAX_LEN)
    test_ds  = EmotionTextDataset(list(test_df["text"]),  list(test_df["label"]),
                                   tokenizer, MAX_LEN)

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes)
    model.to(DEVICE)

    # ── Optimiser / Scheduler ─────────────────────────────────────────────
    no_decay     = ["bias", "LayerNorm.weight"]
    opt_grouped  = [
        {"params": [p for n,p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n,p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],     "weight_decay": 0.0},
    ]
    optimizer    = AdamW(opt_grouped, lr=LR)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_P * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=(DEVICE.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    patience, patience_ctr = 8, 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler,
                                       scaler, DEVICE)
        vl_loss, vl_acc, _ = eval_epoch(model, val_loader, DEVICE)
        train_accs.append(tr_acc); val_accs.append(vl_acc)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"Val loss {vl_loss:.4f} acc {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_pretrained(os.path.join(MODEL_DIR, "text_model"))
            tokenizer.save_pretrained(os.path.join(MODEL_DIR, "text_model"))
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best Val Accuracy: {best_val_acc:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_history(train_accs, val_accs, train_losses, val_losses, "text")

    # ── Test evaluation ───────────────────────────────────────────────────
    best_model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(MODEL_DIR, "text_model"), num_labels=num_classes).to(DEVICE)
    _, te_acc, logits = eval_epoch(best_model, test_loader, DEVICE)

    y_true      = np.array(list(test_df["label"]))
    y_pred_prob = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred      = np.argmax(y_pred_prob, axis=1)
    from tensorflow.keras.utils import to_categorical as to_cat
    y_true_bin  = to_cat(y_true, num_classes)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, le.classes_, "text")
    plot_roc(y_true_bin, y_pred_prob, le.classes_, "text")

    # ── Save metrics ──────────────────────────────────────────────────────
    rows = [{"class": lbl, **vals}
            for lbl, vals in report.items() if isinstance(vals, dict)]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_AT, "text_metrics.csv"), index=False)

    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "pred_label": le.inverse_transform(y_pred),
        "correct":    y_true == y_pred,
    })
    pred_df.to_csv(os.path.join(RESULTS_AT, "text_predictions.csv"), index=False)
    print("Results saved.")

    print("\nText pipeline training COMPLETE.")

if __name__ == "__main__":
    main()


main()
