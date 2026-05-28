# Multimodal Emotion Recognition using Speech and Text

## Overview

This project presents a **Multimodal Emotion Recognition System** developed using the **TESS (Toronto Emotional Speech Set)** dataset. The system combines both **speech-based** and **text-based** emotional understanding to classify emotions more effectively.

The project explores:

* Temporal acoustic modelling
* Transformer-based contextual embeddings
* Multimodal representation learning

Three independent pipelines were implemented:

* Speech Emotion Recognition
* Text Emotion Recognition
* Multimodal Fusion

---

# System Architecture

## Speech Pipeline

The speech pipeline uses:

* CNN layers for acoustic feature extraction
* BiLSTM layers for temporal sequence modelling
* Attention mechanism for focusing on emotionally relevant regions

### Extracted Features

* MFCC
* Delta MFCC
* Mel Spectrogram
* Chroma Features
* Spectral Contrast
* Zero Crossing Rate

The speech model demonstrated extremely strong standalone performance due to the rich emotional information present in speech signals.

---

## Text Pipeline

The text pipeline uses **DistilBERT** contextual embeddings for textual emotion understanding.

Since the TESS dataset contains repetitive and constrained textual content, the text-only model struggled to achieve strong standalone performance. However, contextual embeddings still contributed useful complementary information during multimodal fusion.

---

## Fusion Pipeline

The fusion pipeline combines:

* Speech embeddings
* DistilBERT contextual embeddings

These representations are fused for joint emotion classification, improving robustness and reducing ambiguity in emotionally similar classes.

---

# Dataset

## Toronto Emotional Speech Set (TESS)

### Emotion Classes

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

### Dataset Information

* 5600 audio samples
* WAV format
* Emotion-labelled speech recordings

---

# Experimental Results

| Pipeline            | Accuracy            | Observation                            |
| ------------------- | ------------------- | -------------------------------------- |
| Speech Pipeline | **99.58%**          | Strong temporal acoustic modelling     |
| Text Pipeline    | **16.67%**          | Limited contextual diversity in TESS   |
| Fusion Pipeline  | Improved robustness | Combined speech + text representations |

The speech pipeline achieved the strongest standalone performance due to highly discriminative acoustic emotional features. The text pipeline struggled because the dataset contains repetitive textual structures with limited semantic variation.

The multimodal fusion model improved robustness by combining complementary acoustic and contextual representations.

---

# Speech Pipeline Analysis

## Training Performance

<p align="center">
  <img src="results/plots/speech_training_curves.png" width="470"/>
</p>

The speech training curves demonstrate stable convergence with minimal overfitting. Training and validation accuracy improved consistently throughout training, indicating strong temporal feature learning and good generalization capability.

---

## Confusion Matrix & ROC Curve

<p align="center">
  <img src="results/plots/speech_confusion_matrix.png" width="390"/>
  <img src="results/plots/speech_roc_curve.png" width="390"/>
</p>

The confusion matrix demonstrates strong class separability across most emotions.

### Key Observations

* Angry and Happy emotions achieved very high precision.
* Most emotional categories were classified with minimal overlap.
* Difficult emotional pairs included:

  * Fear ↔ Sad
  * Neutral ↔ Pleasant Surprise

The ROC curves further confirm excellent discrimination capability across emotional classes.

The high AUC performance validates the effectiveness of:

* CNN feature extraction
* BiLSTM temporal modelling
* Attention-based learning

---

## Speech Classification Metrics

| Metric             | Value                      |
| ------------------ | -------------------------- |
| Accuracy           | 99.58%                     |
| Strongest Emotions | Angry, Happy               |
| Hardest Emotions   | Fear, Neutral              |
| Main Strength      | Temporal acoustic learning |

### CSV Reports

* [Speech Metrics](results/accuracy-tables/speech_metrics.csv)
* [Speech Predictions](results/accuracy-tables/speech_predictions.csv)

---

## Grad-CAM Visualization

<p align="center">
  <img src="results/plots/speech_gradcam.png" width="560"/>
</p>

Grad-CAM heatmaps highlight emotionally relevant speech regions used during classification.

The model focused strongly on:

* pitch transitions
* energy shifts
* emotionally informative spectral regions

This improves interpretability and confirms meaningful emotional representation learning.

---

# Text Pipeline Analysis

## Training Performance

<p align="center">
  <img src="results/plots/text_training_curves.png" width="470"/>
</p>

The text model showed significantly weaker learning performance compared to the speech model.

The primary limitation was the repetitive textual structure of the TESS dataset, which restricted DistilBERT’s ability to learn discriminative contextual semantics.

---

## Confusion Matrix & ROC Curve

<p align="center">
  <img src="results/plots/text_confusion_matrix.png" width="390"/>
  <img src="results/plots/text_roc_curve.png" width="390"/>
</p>

The confusion matrix demonstrates substantial overlap between emotional classes.

### Observations

* Strong prediction bias toward dominant classes
* Reduced contextual separability
* Limited semantic diversity

The ROC curves further confirm weaker discrimination capability compared to the speech pipeline.

Despite weaker standalone accuracy, contextual embeddings still contributed useful complementary information during multimodal fusion.

---

## Text Classification Metrics

| Metric           | Value                              |
| ---------------- | ---------------------------------- |
| Accuracy         | 16.67%                             |
| Main Limitation  | Low textual diversity              |
| Strongest Aspect | Contextual embeddings              |
| Weakness         | Poor standalone emotion separation |

### CSV Reports

* [Text Metrics](results/accuracy-tables/text_metrics.csv)
* [Text Predictions](results/accuracy-tables/text_test_predictions.csv)

---

# Fusion Pipeline Analysis

The multimodal fusion model combines speech and contextual embeddings for joint emotion classification.

Fusion improved overall robustness by leveraging complementary modalities simultaneously.

### Fusion Benefits

* Reduced ambiguity between emotionally similar classes
* Improved representation consistency
* Combined temporal + contextual information

---

## Fusion Metrics

| Aspect          | Observation                   |
| --------------- | ----------------------------- |
| Fusion Strategy | Early feature fusion          |
| Modalities Used | Speech + Text                 |
| Main Benefit    | Improved robustness           |
| Key Strength    | Complementary representations |

### CSV Reports

* [Fusion Metrics](results/accuracy-tables/fusion_metrics.csv)
* [Fusion Predictions](results/accuracy-tables/fusion_predictions.csv)

---

# Error Analysis

| Emotion Pair                | Reason                                     |
| --------------------------- | ------------------------------------------ |
| Fear ↔ Sad                  | Similar acoustic energy and pitch patterns |
| Neutral ↔ Pleasant Surprise | Subtle emotional intensity differences     |
| Sad ↔ Neutral               | Low spectral variation                     |

Most classification errors occurred in lower-intensity emotional categories with overlapping temporal characteristics.

---

# Repository Structure

```text id="jbyrbt"
.
├── models/
│   ├── fusion-pipeline/
│   ├── speech-pipeline/
│   └── text-pipeline/
│
├── results/
│   ├── accuracy-tables/
│   └── plots/
│
├── multimodal.ipynb
├── README.md
└── requirements.txt
```

---

# Running the Project

## Install Dependencies

```bash id="a1t59g"
pip install -r requirements.txt
```

---

## Speech Pipeline

```bash id="e3nztc"
python models/speech-pipeline/train.py
python models/speech-pipeline/test.py
```

---

## Text Pipeline

```bash id="ag12wo"
python models/text-pipeline/train.py
python models/text-pipeline/test.py
```

---

## Fusion Pipeline

```bash id="pq8v1q"
python models/fusion-pipeline/train.py
python models/fusion-pipeline/test.py
```

---

# 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* HuggingFace Transformers
* Librosa
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

---

# License

This project is intended for academic and educational purposes.
