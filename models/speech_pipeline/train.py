"""
SPEECH EMOTION RECOGNITION PIPELINE - TRAINING
===============================================
CNN + BiLSTM + Attention for speech emotion classification

This script:
- Generates synthetic speech audio data
- Extracts acoustic features (MFCC, spectrograms, etc)
- Applies augmentation
- Trains CNN+BiLSTM model with attention
- Saves model and embeddings for fusion
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import soundfile as sf
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'num_classes': 7,
    'emotions': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
    'sr': 22050,
    'duration': 3,
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'learning_rate': 1e-3
}

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('models/speech_pipeline', exist_ok=True)
os.makedirs('Results/plots', exist_ok=True)
os.makedirs('Results/accuracy_tables', exist_ok=True)

print("="*70)
print(" SPEECH EMOTION RECOGNITION - TRAINING")
print("="*70)

# ============================================================================
# 2. SYNTHETIC AUDIO GENERATION
# ============================================================================

def generate_emotion_audio(emotion_idx, sr, duration):
    """
    Generate synthetic audio with emotion characteristics
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Emotion-specific parameters
    emotion_params = {
        0: {'pitch': 200, 'variation': 80, 'speed': 1.2, 'energy': 1.0},   # anger
        1: {'pitch': 150, 'variation': 30, 'speed': 0.8, 'energy': 0.6},   # disgust
        2: {'pitch': 220, 'variation': 100, 'speed': 1.3, 'energy': 0.8},  # fear
        3: {'pitch': 180, 'variation': 60, 'speed': 1.1, 'energy': 0.9},   # happiness
        4: {'pitch': 160, 'variation': 20, 'speed': 1.0, 'energy': 0.5},   # neutral
        5: {'pitch': 120, 'variation': 40, 'speed': 0.7, 'energy': 0.4},   # sadness
        6: {'pitch': 240, 'variation': 120, 'speed': 1.2, 'energy': 0.85}, # surprise
    }
    
    params = emotion_params[emotion_idx]
    
    # Generate base frequency modulation
    freq = params['pitch'] + params['variation'] * np.sin(2 * np.pi * t * params['speed'] / duration)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    audio = np.sin(phase) * params['energy']
    
    # Add harmonic components
    audio += 0.3 * np.sin(2 * phase) * params['energy']
    audio += 0.15 * np.sin(3 * phase) * params['energy']
    
    # Add speech-like modulation
    envelope = np.sin(2 * np.pi * t * 2 / duration) ** 2
    audio *= envelope
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio.astype(np.float32)

def generate_speech_dataset(num_samples=1400):
    """
    Generate synthetic speech emotion dataset
    """
    print("\nGenerating synthetic speech dataset...")
    
    samples_per_emotion = num_samples // CONFIG['num_classes']
    audios = []
    labels = []
    
    for emotion_idx in range(CONFIG['num_classes']):
        for _ in range(samples_per_emotion):
            audio = generate_emotion_audio(
                emotion_idx, CONFIG['sr'], CONFIG['duration']
            )
            audios.append(audio)
            labels.append(emotion_idx)
    
    print(f"✓ Generated {num_samples} audio samples")
    print(f"  - Duration per sample: {CONFIG['duration']}s")
    print(f"  - Sample rate: {CONFIG['sr']} Hz")
    
    return np.array(audios), np.array(labels)

# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

def extract_audio_features(audio, sr):
    """
    Extract MFCC + Spectrogram features from audio
    """
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=CONFIG['n_mfcc'],
                                n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'])
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=CONFIG['n_fft'],
                                              hop_length=CONFIG['hop_length'])
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=CONFIG['n_fft'],
                                         hop_length=CONFIG['hop_length'])
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=CONFIG['n_fft'],
                                                      hop_length=CONFIG['hop_length'])
    spec_contrast_mean = np.mean(spec_contrast, axis=1)
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=CONFIG['hop_length'])
    zcr_mean = np.mean(zcr)
    
    # Combine spectral features
    combined_spec = np.vstack([mel_spec[:40], np.zeros((8, mel_spec.shape[1]))])
    
    return combined_spec.astype(np.float32), np.concatenate([
        mfcc_mean, chroma_mean, spec_contrast_mean, [zcr_mean]
    ]).astype(np.float32)

def create_spectrogram_dataset(audios):
    """
    Extract spectrograms for all audio samples
    """
    print("\nExtracting spectrograms...")
    
    spectrograms = []
    for i, audio in enumerate(audios):
        spec, _ = extract_audio_features(audio, CONFIG['sr'])
        spectrograms.append(spec)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(audios)} samples")
    
    spectrograms = np.array(spectrograms)
    
    print(f"✓ Spectrograms shape: {spectrograms.shape}")
    
    return spectrograms

# ============================================================================
# 4. DATA AUGMENTATION
# ============================================================================

def augment_audio(audio):
    """
    Apply data augmentation to audio
    """
    # Pitch shifting
    if np.random.rand() > 0.5:
        n_steps = np.random.randint(-2, 3)
        audio = librosa.effects.pitch_shift(audio, sr=CONFIG['sr'], n_steps=n_steps)
    
    # Time stretching
    if np.random.rand() > 0.5:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)
    
    # Noise injection
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.005, len(audio))
        audio = audio + noise
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio.astype(np.float32)

def apply_augmentation(audios):
    """
    Apply augmentation to dataset
    """
    print("\nApplying data augmentation...")
    
    augmented_audios = []
    for audio in audios:
        augmented = augment_audio(audio)
        augmented_audios.append(augmented)
    
    print(f"✓ Augmented {len(audios)} samples")
    
    return np.array(augmented_audios)

# ============================================================================
# 5. TRAIN/VAL/TEST SPLIT
# ============================================================================

def create_train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Split data into train, validation, and test sets
    """
    print("\nCreating train/validation/test split...")
    
    num_samples = len(y)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    
    X_train = X[:num_train]
    y_train = y[:num_train]
    
    X_val = X[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]
    
    X_test = X[num_train + num_val:]
    y_test = y[num_train + num_val:]
    
    print(f"✓ Train set: {len(y_train)} samples ({train_ratio*100:.0f}%)")
    print(f"✓ Val set: {len(y_val)} samples ({val_ratio*100:.0f}%)")
    print(f"✓ Test set: {len(y_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ============================================================================
# 6. BUILD SPEECH MODEL
# ============================================================================

def build_speech_model(input_shape, num_classes):
    """
    Build CNN + BiLSTM + Attention model for speech
    
    Architecture:
    - Input: Spectrogram (48, T)
    - 3 CNN blocks for feature extraction
    - BiLSTM for temporal modeling
    - Attention mechanism
    - Dense classification layers
    """
    print("\nBuilding speech model...")
    
    model = models.Sequential()
    
    # Reshape input
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
    
    # CNN Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # CNN Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Reshape for LSTM
    model.add(layers.Reshape((model.output_shape[1], -1)))
    
    # BiLSTM
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Dropout(0.3))
    
    # Attention
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.RepeatVector(1))
    
    # Output
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("✓ Model compiled")
    print(model.summary())
    
    return model

# ============================================================================
# 7. TRAINING
# ============================================================================

def train_speech_model(model, train_data, val_data):
    """
    Train speech model
    """
    print("\n" + "="*70)
    print(" TRAINING SPEECH MODEL")
    print("="*70)
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        'models/speech_pipeline/speech_model.h5',
        monitor='val_accuracy', save_best_only=True, mode='max'
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print("\n✓ Training completed")
    
    return history

# ============================================================================
# 8. EVALUATION & VISUALIZATION
# ============================================================================

def evaluate_and_visualize(model, test_data, emotions):
    """
    Evaluate model and generate visualizations
    """
    print("\n" + "="*70)
    print(" EVALUATION & VISUALIZATION")
    print("="*70)
    
    X_test, y_test = test_data
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Speech Model - Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/plots/speech_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speech_confusion_matrix.png")
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    
    for i, emotion in enumerate(emotions):
        y_test_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{emotion} (AUC={roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Speech Model - ROC Curves', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/plots/speech_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speech_roc_curves.png")
    plt.close()
    
    return accuracy, precision, recall, f1, y_pred, y_pred_probs

def plot_training_history(history):
    """
    Plot training curves
    """
    print("\nGenerating training plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/plots/speech_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speech_training_history.png")
    plt.close()

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

def save_results(accuracy, precision, recall, f1, y_test, y_pred, emotions):
    """
    Save results to CSV
    """
    print("\nSaving results...")
    
    # Metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('Results/accuracy_tables/speech_metrics.csv', index=False)
    print("✓ Saved: speech_metrics.csv")
    
    # Per-class metrics
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(emotions))
    )
    
    per_class_df = pd.DataFrame({
        'Emotion': emotions,
        'Precision': precision_pc,
        'Recall': recall_pc,
        'F1-Score': f1_pc,
        'Support': support_pc
    })
    per_class_df.to_csv('Results/accuracy_tables/speech_per_class_metrics.csv', index=False)
    print("✓ Saved: speech_per_class_metrics.csv")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    cm_df.to_csv('Results/accuracy_tables/speech_confusion_matrix.csv')
    print("✓ Saved: speech_confusion_matrix.csv")

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # 1. Generate synthetic dataset
    audios, labels = generate_speech_dataset(num_samples=1400)
    
    # 2. Extract spectrograms
    X = create_spectrogram_dataset(audios)
    
    # 3. Normalize
    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_norm = scaler.fit_transform(X_flat)
    X = X_norm.reshape(X.shape)
    
    # 4. Split data
    train_data, val_data, test_data = create_train_val_test_split(X, labels)
    
    # 5. Build model
    model = build_speech_model(X[0].shape, CONFIG['num_classes'])
    
    # 6. Train
    history = train_speech_model(model, train_data, val_data)
    
    # 7. Evaluate
    accuracy, precision, recall, f1, y_pred, y_pred_probs = evaluate_and_visualize(
        model, test_data, CONFIG['emotions']
    )
    
    # 8. Visualize
    plot_training_history(history)
    
    # 9. Save results
    X_test, y_test = test_data
    save_results(accuracy, precision, recall, f1, y_test, y_pred, CONFIG['emotions'])
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print(" Model saved to: models/speech_pipeline/speech_model.h5")
    print(" Results saved to: Results/ directory")
    print("="*70 + "\n")
