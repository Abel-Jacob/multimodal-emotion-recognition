# Multimodal Emotion Recognition using Speech and Text

## Overview

This project implements a **Multimodal Emotion Recognition System** that combines both **speech-based** and **text-based** emotional understanding for improved emotion classification performance.

The system consists of three independent pipelines:

* **Speech Emotion Recognition Pipeline**
* **Text Emotion Recognition Pipeline**
* **Multimodal Fusion Pipeline**

The final fusion model combines acoustic and contextual information to improve classification robustness and overall accuracy.

---

# Features

## Speech Emotion Recognition

* Audio preprocessing using Librosa
* Feature extraction:

  * MFCC
  * Delta MFCC
  * Mel Spectrogram
  * Chroma Features
  * Spectral Contrast
  * Zero Crossing Rate
* CNN + BiLSTM + Attention architecture

## Text Emotion Recognition

* DistilBERT contextual embeddings
* Transformer-based language representation
* Dense classification layers

## Multimodal Fusion

* Fusion of speech and text embeddings
* Joint multimodal learning
* Improved emotion separability

---

# Dataset

## TESS Dataset

Toronto Emotional Speech Set (TESS)

### Emotions Used

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

### Dataset Characteristics

* 5600 speech samples
* WAV audio format
* Professionally acted emotional speech

---

# Project Structure

```bash
project/
│
├── notebooks/
│   ├── speech_pipeline.ipynb
│   ├── text_pipeline.ipynb
│   └── fusion_pipeline.ipynb
│
├── train/
│   ├── speech_train.py
│   ├── text_train.py
│   └── fusion_train.py
│
├── test/
│   ├── speech_test.py
│   ├── text_test.py
│   └── fusion_test.py
│
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── Results/
│   ├── plots/
│   └── accuracy_tables/
│
├── report/
│
└── README.md
```

---

# Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Librosa
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Google Colab

---

# Model Architectures

## 1. Speech Pipeline

* CNN layers for spatial acoustic feature extraction
* BiLSTM layers for temporal modelling
* Attention mechanism for important sequence weighting

## 2. Text Pipeline

* DistilBERT transformer embeddings
* Dense neural classifier

## 3. Fusion Pipeline

* Concatenation of speech and text embeddings
* Dense fusion network for multimodal learning

---

# Experimental Setup

Three separate experiments were conducted:

| Experiment        | Description                                    |
| ----------------- | ---------------------------------------------- |
| Speech-only       | Emotion classification using acoustic features |
| Text-only         | Emotion classification using text embeddings   |
| Multimodal Fusion | Combined speech and text classification        |

---

# Results

| Model        | Accuracy |
| ------------ | -------- |
| Speech Model | 97%      |
| Text Model   | 85%      |
| Fusion Model | 100%     |

The multimodal fusion approach achieved the highest performance by combining complementary information from both modalities.

---

# Analysis

## Key Observations

* Happy and angry emotions were easiest to classify due to distinct acoustic patterns.
* Fear and neutral emotions showed higher confusion because of overlapping emotional characteristics.
* Fusion improved robustness when one modality alone was ambiguous.

## Error Analysis

Common confusion cases:

* Fear ↔ Sad
* Neutral ↔ Pleasant Surprise
* Sad ↔ Neutral

---

# Visualizations

The project includes:

* Confusion matrices
* Accuracy comparison plots
* ROC curves
* Emotion cluster visualization (PCA / t-SNE)

---

# Future Improvements

* Real-time emotion recognition
* Video modality integration
* Larger multilingual datasets
* Transformer-based multimodal fusion
* Real-world conversational emotion recognition

---

# How to Run

## Install Dependencies

```bash
pip install tensorflow torch transformers librosa scikit-learn matplotlib seaborn pandas
```

## Run Speech Pipeline

```bash
python speech_train.py
```

## Run Text Pipeline

```bash
python text_train.py
```

## Run Fusion Pipeline

```bash
python fusion_train.py
```

---

# Authors

Abel Jacob
