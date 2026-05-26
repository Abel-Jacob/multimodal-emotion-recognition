"""
MULTIMODAL FUSION PIPELINE - TESTING
=====================================
Test the fusion model with embeddings from both modalities

This script:
- Loads trained fusion model
- Generates test embeddings
- Runs inference
- Generates comprehensive evaluation
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    'num_classes': 7,
    'emotions': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
    'speech_embedding_dim': 256,
    'text_embedding_dim': 512
}

np.random.seed(42)

os.makedirs('Results/plots', exist_ok=True)
os.makedirs('Results/accuracy_tables', exist_ok=True)

# ============================================================================
# 2. LOAD MODEL
# ============================================================================

def load_model():
    """
    Load trained fusion model
    """
    print("Loading trained fusion model...")
    
    try:
        model = keras.models.load_model('models/fusion_pipeline/fusion_model.h5')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Model not found! Please run training first:")
        print("  python models/fusion_pipeline/train.py")
        exit(1)
    
    return model

# ============================================================================
# 3. GENERATE TEST DATA
# ============================================================================

def generate_test_embeddings(num_samples=400):
    """
    Generate test embeddings
    (In production, these would come from real models)
    """
    print("Generating test embeddings...")
    
    samples_per_emotion = num_samples // CONFIG['num_classes']
    
    # Generate speech embeddings
    X_speech = np.random.randn(num_samples, CONFIG['speech_embedding_dim']).astype('float32')
    
    # Generate text embeddings
    X_text = np.random.randn(num_samples, CONFIG['text_embedding_dim']).astype('float32')
    
    # Generate labels
    y = np.repeat(np.arange(CONFIG['num_classes']), samples_per_emotion)
    
    # Add emotion-specific patterns
    for i in range(num_samples):
        emotion_idx = y[i]
        X_speech[i, :50] += emotion_idx * 0.3
        X_text[i, :50] += emotion_idx * 0.2
    
    # Normalize
    scaler_speech = StandardScaler()
    scaler_text = StandardScaler()
    
    X_speech = scaler_speech.fit_transform(X_speech)
    X_text = scaler_text.fit_transform(X_text)
    
    return X_speech, X_text, y

# ============================================================================
# 4. INFERENCE
# ============================================================================

def get_predictions(model, X_speech, X_text):
    """
    Get model predictions
    """
    print("Running inference...")
    
    y_pred_probs = model.predict([X_speech, X_text], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    return y_pred, y_pred_probs

# ============================================================================
# 5. EVALUATION
# ============================================================================

def compute_metrics(y_test, y_pred, y_pred_probs, emotions):
    """
    Compute evaluation metrics
    """
    print("\n" + "="*60)
    print("TEST SET EVALUATION METRICS")
    print("="*60)
    
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
        print(f"  {emotion.capitalize():10s} - P:{precision_pc[i]:.3f} R:{recall_pc[i]:.3f} F1:{f1_pc[i]:.3f} (n={support_pc[i]})")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    return accuracy, precision, recall, f1

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

def visualize_results(y_test, y_pred, y_pred_probs, emotions):
    """
    Generate test visualizations
    """
    print("\n" + "="*60)
    print("GENERATING TEST VISUALIZATIONS")
    print("="*60)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Purples',
                xticklabels=emotions, yticklabels=emotions,
                cbar_kws={'label': 'Normalized Value'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Fusion Model - Test Confusion Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_confusion.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_confusion.png")
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    
    for i, emotion in enumerate(emotions):
        y_test_binary = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{emotion} (AUC={roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('Fusion Model - Test ROC Curves', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_roc.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_roc.png")
    plt.close()
    
    # 3. Per-Emotion Accuracy
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
    plt.title('Fusion Model - Test Accuracy by Emotion', fontsize=12, fontweight='bold')
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
    
    # 4. Confidence Distribution
    max_probs = np.max(y_pred_probs, axis=1)
    correct = y_pred == y_test
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct], bins=30, alpha=0.6, label='Correct', color='green')
    plt.hist(max_probs[~correct], bins=30, alpha=0.6, label='Incorrect', color='red')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Confidence Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([max_probs[correct], max_probs[~correct]],
                labels=['Correct', 'Incorrect'])
    plt.ylabel('Confidence', fontsize=11)
    plt.title('Confidence by Correctness', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_confidence.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_confidence.png")
    plt.close()
    
    # 5. Error Analysis - Emotion Confusion
    confusion_data = np.zeros((len(emotions), len(emotions)))
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            mask = y_test == i
            if np.sum(mask) > 0:
                confusion_data[i, j] = np.sum(y_pred[mask] == j) / np.sum(mask)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_data, annot=np.round(confusion_data, 2), fmt='g',
                xticklabels=emotions, yticklabels=emotions, cmap='YlOrRd',
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Fusion Model - Emotion Confusion Proportions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_emotion_confusion.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_emotion_confusion.png")
    plt.close()
    
    # 6. Error Types
    errors = y_pred != y_test
    error_count = np.sum(errors)
    correct_count = np.sum(~errors)
    
    plt.figure(figsize=(8, 6))
    sizes = [correct_count, error_count]
    labels = [f'Correct\n({correct_count})', f'Incorrect\n({error_count})']
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0.1)
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            explode=explode, textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Fusion Model - Test Predictions Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Results/plots/fusion_test_predictions_pie.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fusion_test_predictions_pie.png")
    plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

def save_results(y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1, emotions):
    """
    Save all results to CSV
    """
    print("\n" + "="*60)
    print("SAVING TEST RESULTS")
    print("="*60)
    
    # Overall metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('Results/accuracy_tables/fusion_test_metrics.csv', index=False)
    print("✓ Saved: fusion_test_metrics.csv")
    
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
    per_class_df.to_csv('Results/accuracy_tables/fusion_test_per_class.csv', index=False)
    print("✓ Saved: fusion_test_per_class.csv")
    
    # Predictions
    predictions_df = pd.DataFrame({
        'True_Emotion': [emotions[i] for i in y_test],
        'Predicted_Emotion': [emotions[i] for i in y_pred],
        'Correct': y_pred == y_test,
        'Confidence': np.max(y_pred_probs, axis=1)
    })
    predictions_df.to_csv('Results/accuracy_tables/fusion_test_predictions.csv', index=False)
    print("✓ Saved: fusion_test_predictions.csv")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=emotions, columns=emotions)
    cm_df.to_csv('Results/accuracy_tables/fusion_test_confusion_matrix.csv')
    print("✓ Saved: fusion_test_confusion_matrix.csv")
    
    # Error analysis
    errors = y_pred != y_test
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        errors_data = []
        for idx in error_indices[:100]:
            errors_data.append({
                'True_Emotion': emotions[y_test[idx]],
                'Predicted_Emotion': emotions[y_pred[idx]],
                'Confidence': np.max(y_pred_probs[idx]),
                'Second_Best_Confidence': np.sort(y_pred_probs[idx])[-2]
            })
        
        errors_df = pd.DataFrame(errors_data)
        errors_df.to_csv('Results/accuracy_tables/fusion_test_errors.csv', index=False)
        print("✓ Saved: fusion_test_errors.csv")
    
    # Summary report
    summary_text = f"""
FUSION MODEL - TEST SUMMARY REPORT
===================================

Overall Performance:
  Accuracy:  {accuracy:.4f}
  Precision: {precision:.4f}
  Recall:    {recall:.4f}
  F1-Score:  {f1:.4f}

Total Test Samples: {len(y_test)}
Correct Predictions: {np.sum(y_pred == y_test)}
Incorrect Predictions: {np.sum(y_pred != y_test)}
Error Rate: {100 * np.sum(y_pred != y_test) / len(y_test):.2f}%

Per-Class Performance:
"""
    
    for i, emotion in enumerate(emotions):
        summary_text += f"  {emotion.capitalize():10s}: P={precision_pc[i]:.3f}, R={recall_pc[i]:.3f}, F1={f1_pc[i]:.3f}\n"
    
    with open('Results/accuracy_tables/fusion_test_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("✓ Saved: fusion_test_summary.txt")

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" MULTIMODAL FUSION PIPELINE - TESTING")
    print("="*70)
    
    # 1. Load model
    model = load_model()
    
    # 2. Generate test data
    X_speech_test, X_text_test, y_test = generate_test_embeddings(num_samples=400)
    
    # 3. Get predictions
    y_pred, y_pred_probs = get_predictions(model, X_speech_test, X_text_test)
    
    # 4. Compute metrics
    accuracy, precision, recall, f1 = compute_metrics(y_test, y_pred, y_pred_probs, CONFIG['emotions'])
    
    # 5. Visualize results
    visualize_results(y_test, y_pred, y_pred_probs, CONFIG['emotions'])
    
    # 6. Save results
    save_results(y_test, y_pred, y_pred_probs, accuracy, precision, recall, f1, CONFIG['emotions'])
    
    print("\n" + "="*70)
    print(" TESTING COMPLETE!")
    print(" Results saved to Results/ directory")
    print("="*70 + "\n")
