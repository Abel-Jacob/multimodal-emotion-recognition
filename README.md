You are an expert academic reviewer, research mentor, and technical writing assistant.

Your task is to create a complete final-year project report for a project titled:

“Multimodal Emotion Recognition using Speech and Text”

The report must be professionally written in formal academic style and formatted like a university project report.

The report should NOT sound AI-generated.
Avoid repetitive wording, exaggerated claims, or robotic transitions.
Write naturally, technically, and professionally.

Use clear section headings, proper paragraph flow, and concise explanations.

The project implementation includes:

1. Speech Emotion Recognition Pipeline

* CNN + BiLSTM + Attention architecture
* Audio preprocessing using Librosa
* Features:

  * MFCC
  * Delta MFCC
  * Mel Spectrogram
  * Chroma Features
  * Spectral Contrast
  * Zero Crossing Rate

2. Text Emotion Recognition Pipeline

* DistilBERT-based contextual embeddings
* Dense classification layers

3. Multimodal Fusion Pipeline

* Fusion of speech and text embeddings
* Dense layers for final classification
* Multimodal learning approach

Dataset used:

* TESS (Toronto Emotional Speech Set)

Emotions:

* angry
* disgust
* fear
* happy
* neutral
* pleasant surprise
* sad

Implementation environment:

* Python
* TensorFlow / Keras
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* Google Colab

The report MUST include the following sections in order:

1. Title Page
2. Certificate / Declaration Page
3. Acknowledgement
4. Abstract
5. Table of Contents
6. Introduction
7. Problem Statement
8. Objectives
9. Literature Survey
10. Dataset Description
11. System Architecture
12. Methodology
13. Implementation Details
14. Experiments
15. Analysis
16. Results
17. Challenges Faced
18. Limitations
19. Future Scope
20. Conclusion
21. References

IMPORTANT REQUIREMENTS FROM PROJECT PDF:

A. Architecture Decisions
For EACH block explain:

* What architecture was used
* Why that architecture was selected
* Advantages of the selected architecture

Include detailed explanation for:

* Temporal Modelling block
* Contextual Modelling block
* Fusion block

B. Experiments Section
Compare:

* Speech-only model
* Text-only model
* Multimodal Fusion model

Include:

* accuracy comparison table
* performance discussion
* observations

C. Analysis Section
The report MUST discuss:

1. Which emotions are easiest to classify and why

2. Which emotions are hardest to classify and why

3. When multimodal fusion helps most

4. Error analysis with 3–5 failure cases

5. Visualization and separability analysis of learned emotion representations from:

* Temporal Modelling block
* Contextual Modelling block
* Fusion block

Discuss:

* clustering behavior
* overlap between emotions
* improvement in separability after fusion

The report should also include:

* confusion matrix discussion
* model comparison discussion
* interpretation of results
* practical observations
* academic-style analysis

Use realistic academic language.

DO NOT:

* use bullet spam everywhere
* overuse “furthermore”, “moreover”, etc.
* sound like marketing
* make unrealistic claims
* overstate accuracy

Include tables wherever appropriate.

Include placeholders where figures/plots/confusion matrices should be inserted.

Examples:
[Insert Speech Confusion Matrix]
[Insert Fusion Accuracy Comparison Graph]
[Insert t-SNE Visualization]

Write the report in a clean university report format suitable for direct PDF export.

The final output should be detailed, professional, and approximately 25–40 pages worth of content.
