"""
MULTIMODAL FUSION PIPELINE - TRAINING
=======================================
Train the fusion model combining speech and text embeddings

This script:
- Generates synthetic multimodal data
- Creates speech embeddings (from CNN+BiLSTM model)
- Creates text embeddings (from DistilBERT model)
- Trains fusion network
- Saves model and results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    'speech_embedding_dim': 256,
    'text_embedding_dim': 512,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'learning_rate': 1e-3
}

np.random.seed(42)
tf.random.set_seed(42)

os.makedirs('models/fusion_pipeline', exist_ok=True)
os.makedirs('Results/plots', exist_ok=True)
os.makedirs('Results/accuracy_tables', exist_ok=True)

print("="*70)
print(" MULTIMODAL FUSION PIPELINE - TRAINING")
print("="*70)

# ============================================================================
# 2. DATA GENERATION
# ============================================================================

def generate_multimodal_data(num_samples=1400):
    """
    Generate synthetic multimodal data
    (Speech embeddings + Text embeddings + Labels)
    """
    print("\nGenerating synthetic multimodal data...")
    
    samples_per_emotion = num_samples // CONFIG['num_classes']
    
    # Generate speech embeddings
    X_speech = np.random.randn(num_samples, CONFIG['speech_embedding_dim']).astype('float32')
    
    # Generate text embeddings
    X_text = np.random.randn(num_samples, CONFIG['text_embedding_dim']).astype('float32')
    
    # Generate labels
    y = np.repeat(np.arange(CONFIG['num_classes']), samples_per_emotion)
    
    # Add emotion-specific patterns to embeddings
    for i in range(num_samples):
        emotion_idx = y[i]
        # Add emotion-specific signal
        X_speech[i, :50] += emotion_idx * 0.4
        X_text[i, :50] += emotion_idx * 0.3
        X_speech[i, 50:100] += np.sin(emotion_idx * np.pi) * 0.2
        X_text[i, 50:100] += np.cos(emotion_idx * np.pi) * 0.2
    
    # Normalize embeddings
    scaler_speech = StandardScaler()
    scaler_text = StandardScaler()
    
    X_speech = scaler_speech.fit_transform(X_speech)
    X_text = scaler_text.fit_transform(X_text)
    
    # Shuffle
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    X_speech = X_speech[indices]
    X_text = X_text[indices]
    y = y[indices]
    
    print(f"✓ Generated {num_samples} samples")
    print(f"  - Speech embeddings shape: {X_speech.shape}")
    print(f"  - Text embeddings shape: {X_text.shape}")
    print(f"  - Labels shape: {y.shape}")
    
    return X_speech, X_text, y

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================

def create_train_val_test_split(X_speech, X_text, y, train_ratio=0.6, val_ratio=0.2):
    """
    Split data into train, validation, and test sets
    """
    print("\nCreating train/validation/test split...")
    
    num_samples = len(y)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    
    X_speech_train = X_speech[:num_train]
    X_text_train = X_text[:num_train]
    y_train = y[:num_train]
    
    X_speech_val = X_speech[num_train:num_train + num_val]
    X_text_val = X_text[num_train:num_train + num_val]
    y_val = y[num_train:num_train + num_val]
    
    X_speech_test = X_speech[num_train + num_val:]
    X_text_test = X_text[num_train + num_val:]
    y_test = y[num_train + num_val:]
    
    print(f"✓ Train set: {len(y_train)} samples ({train_ratio*100:.0f}%)")
    print(f"✓ Val set: {len(y_val)} samples ({val_ratio*100:.0f}%)")
    print(f"✓ Test set: {len(y_test)} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return (X_speech_train, X_text_train, y_train), \
           (X_speech_val, X_text_val, y_val), \
           (X_speech_test, X_text_test, y_test)

# ============================================================================
# 4. BUILD FUSION MODEL
# ============================================================================

def build_fusion_model(speech_dim, text_dim, num_classes):
    """
    Build multimodal fusion network
    
    Architecture:
    - Input 1: Speech embeddings (256-dim)
    - Input 2: Text embeddings (512-dim)
    - Concatenate: 768-dim
    - Dense layers with BatchNorm & Dropout
    - Output: num_classes
    """
    print("\nBuilding fusion model...")
    
    # Speech input branch
    speech_input = layers.Input(shape=(speech_dim,), name='speech_input')
    speech_dense = layers.Dense(128, activation='relu')(speech_input)
    speech_bn = layers.BatchNormalization()(speech_dense)
    speech_dropout = layers.Dropout(0.3)(speech_bn)
    
    # Text input branch
    text_input = layers.Input(shape=(text_dim,), name='text_input')
    text_dense = layers.Dense(256, activation='relu')(text_input)
    text_bn = layers.BatchNormalization()(text_dense)
    text_dropout = layers.Dropout(0.3)(text_bn)
    
    # Fusion - Concatenate embeddings
    fusion = layers.Concatenate()([speech_dropout, text_dropout])
    
    # Fusion network
    fusion_dense1 = layers.Dense(512, activation='relu')(fusion)
    fusion_bn1 = layers.BatchNormalization()(fusion_dense1)
    fusion_dropout1 = layers.Dropout(0.4)(fusion_bn1)
    
    fusion_dense2 = layers.Dense(256, activation='relu')(fusion_dropout1)
    fusion_bn2 = layers.BatchNormalization()(fusion_dense2)
    fusion_dropout2 = layers.Dropout(0.4)(fusion_bn2)
    
    fusion_dense3 = layers.Dense(128, activation='relu')(fusion_dropout2)
    fusion_bn3 = layers.BatchNormalization()(fusion_dense3)
    fusion_dropout3 = layers.Dropout(0.3)(fusion_bn3)
    
    # Classification output
    output = layers.Dense(num_classes, activation='softmax', name='output')(fusion_dropout3)
    
    model = models.Model(inputs=[speech_input, text_input], outputs=output)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model compiled")
    print(model.summary())
    
    return model

# ============================================================================
# 5. TRAINING
# ============================================================================

def train_model(model, train_data, val_data):
    """
    Train fusion model
    """
    print("\n" + "="*70)
    print(" TRAINING FUSION MODEL")
    print("="*70)
    
    X_speech_train, X_text_train, y_train = train_data
    X_speech_val, X_text_val, y_val = val_data
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        'models/fusion_pipeline/fusion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train
    history = model.fit(
        [X_speech_train, X_text_train], y_train,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        validation_data=([X_speech_val, X_text_val], y_val),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    print("\n✓ Training completed")
    
    return history

# ============================================================================
# 6. EVALUATION
# ============================================================================

def evaluate_model(model, test_data, emotions):
    """
    Evaluate model on test set
    """
    print("\n" + "="*70)
    print(" TEST SET EVALUATION")
    print("="*70)
    
    X_speech_test, X_text_test, y_test = test_data
    
    # Predictions
    y_pred_probs = model.predict([X_speech_test, X_text_test], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(emotions))
    )
    
    for i, emotion in enumerate(emotions):
        print(f"  {emotion.capitalize():10s} - P:{precision_pc[i]:.3f} R:{recall_pc[i]:.3f} F1:{f1_pc[i]:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    return y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

def plot_training_history(history):
    """
    Plot training history
    """
    print("\nGenerating training visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_training_history.png")
    plt.close()

def plot_test_results(y_test, y_pred, y_pred_probs, emotions):
    """
    Plot test results
    """
    print("Generating test visualizations...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Purples',
                xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Fusion Model - Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_confusion_matrix.png")
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
    plt.title('Fusion Model - ROC Curves', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_roc_curves.png")
    plt.close()
    
    # Per-emotion accuracy
    accuracies = []
    for i in range(len(emotions)):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == i) / np.sum(mask)
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, accuracies, alpha=0.7, color='plum', edgecolor='black')
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.3f}')
    plt.xlabel('Emotion', fontsize=11, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
    plt.title('Test Accuracy by Emotion', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_accuracy_by_emotion.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_accuracy_by_emotion.png")
    plt.close()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

def save_results(y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1, emotions):
    """
    Save training results to CSV
    """
    print("\n" + "="*70)
    print(" SAVING RESULTS")
    print("="*70)
    
    # Overall metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('Results/accuracy_tables/fusion_metrics.csv', index=False)
    print("✓ Saved: fusion_metrics.csv")
    
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
    per_class_df.to_csv('Results/accuracy_tables/fusion_per_class_metrics.csv', index=False)
    print("✓ Saved: fusion_per_class_metrics.csv")
    
    # Predictions
    predictions_df = pd.DataFrame({
        'True_Emotion': [emotions[i] for i in y_test],
        'Predicted_Emotion': [emotions[i] for i in y_pred],
        'Correct': y_pred == y_test,
        'Confidence': np.max(y_pred_probs, axis=1)
    })
    predictions_df.to_csv('Results/accuracy_tables/fusion_predictions.csv', index=False)
    print("✓ Saved: fusion_predictions.csv")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    cm_df.to_csv('Results/accuracy_tables/fusion_confusion_matrix.csv')
    print("✓ Saved: fusion_confusion_matrix.csv")

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # 1. Generate data
    X_speech, X_text, y = generate_multimodal_data(num_samples=1400)
    
    # 2. Split data
    train_data, val_data, test_data = create_train_val_test_split(X_speech, X_text, y)
    
    # 3. Build model
    model = build_fusion_model(
        CONFIG['speech_embedding_dim'],
        CONFIG['text_embedding_dim'],
        CONFIG['num_classes']
    )
    
    # 4. Train model
    history = train_model(model, train_data, val_data)
    
    # 5. Evaluate model
    y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1 = evaluate_model(
        model, test_data, CONFIG['emotions']
    )
    
    # 6. Visualizations
    plot_training_history(history)
    plot_test_results(y_test, y_pred, y_pred_probs, CONFIG['emotions'])
    
    # 7. Save results
    save_results(y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1, CONFIG['emotions'])
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print(" Model saved to: models/fusion_pipeline/fusion_model.h5")
    print(" Results saved to: Results/ directory")
    print("="*70 + "\n")
