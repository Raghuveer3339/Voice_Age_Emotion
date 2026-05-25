# 🎙️ Task 3 — Age & Emotion Detection through Voice

> **Internship Task** | Machine Learning | Custom CNN + Pitch Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/Task3_Voice_Age_Emotion/blob/main/notebooks/Task3_VoiceIQ.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-green)](https://zenodo.org/record/1188976)

---

## 📌 Problem Statement

Build a machine learning model that:
- Accepts a **voice note** as input
- **Rejects female voices** with the message *"Upload male voice"*
- For **male voices**: estimates the speaker's **age**
- If age **> 60** (Senior Citizen): additionally detects **emotion**
- Includes a **Graphical User Interface (GUI)**

---

## 🧠 Approach

### Two-Layer Gender Detection (Robust & Reliable)

| Layer | Method | Condition |
|-------|--------|-----------|
| **1 — Physics** | Pitch (F0) analysis | F0 < 160 Hz → Male &nbsp;\|&nbsp; F0 > 200 Hz → Female |
| **2 — CNN Model** | Custom Conv1D network | F0 in 160–200 Hz (ambiguous zone) |

Male vocal cords physically vibrate at **85–160 Hz** on average. This physics-based first layer ensures even a poorly uploaded recording is handled correctly. The CNN handles edge cases in the ambiguous zone.

### Full Pipeline

```
Voice Input (.wav / .mp3 / .ogg)
       │
       ▼
Feature Extraction
(MFCCs + Delta + Mel Spectrogram + Chroma + ZCR + RMS → 157 features × 128 timesteps)
       │
       ▼
Pitch Analysis (librosa pyin)
       │
   ┌───┴────────────────────────┐
F0 < 160 Hz              F0 > 200 Hz         160–200 Hz
   │                          │                    │
MALE ✅                  FEMALE ⛔           CNN Model
                                                   │
                                          Male / Female decision
       │
       ▼ (Male only)
Age Estimation (CNN + BiLSTM)
       │
   ┌───┴──────────────────┐
Age ≤ 60                Age > 60
   │                        │
Show Age Only          👴 Senior Citizen
                            │
                            ▼
                    Emotion Detection (CNN)
                    8 classes: neutral, calm, happy,
                    sad, angry, fearful, disgust, surprised
```

---

## 🏗️ Model Architectures

### 1. Gender Model — Custom Conv1D CNN
```
Input (128, 157)
→ Conv1D(32) + BN + ReLU + MaxPool + Dropout
→ Conv1D(64) + BN + ReLU + MaxPool + Dropout
→ Conv1D(128) + BN + ReLU + GlobalAvgPool
→ Dense(64) + Dropout
→ Dense(1, sigmoid)   ← Binary: male/female
```

### 2. Age Model — CNN + Bidirectional LSTM
```
Input (128, 157)
→ Conv1D(64) + Conv1D(128) + Conv1D(256) + MaxPooling
→ BiLSTM(128)
→ Dense(128) → Dense(64) → Dense(1, linear)   ← Regression: age in years
```

### 3. Emotion Model — 4-Block CNN
```
Input (128, 157)
→ Conv1D(64) → Conv1D(128) → Conv1D(256) → Conv1D(512)
→ GlobalAvgPool
→ Dense(256) → Dense(128) → Dense(8, softmax)   ← 8 emotion classes
```

---

## 📊 Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 13 female) — perfectly **50/50 gender balanced**
- ~1,440 audio files of emotional speech
- 8 emotion labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Download: [zenodo.org/record/1188976](https://zenodo.org/record/1188976)

> **Note on Age Labels:** RAVDESS does not include real age labels. Age is simulated from actor groups for training purposes. For production use, the Common Voice dataset (which includes age annotations) would be more appropriate.

---

## 🔬 Feature Engineering

Each audio file is converted to a **157-dimensional feature vector** over **128 timesteps**:

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| MFCCs | 40 | Mel-frequency cepstral coefficients |
| Delta MFCCs | 40 | First-order derivatives of MFCCs |
| Mel Spectrogram | 64 | Log-scaled mel filterbank energies |
| Chroma | 12 | Pitch class energy distribution |
| ZCR | 1 | Zero-crossing rate |
| RMS Energy | 1 | Root mean square energy |
| **Total** | **157** | Per timestep |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
Click the **Open in Colab** badge above, then:
1. `Runtime → Change runtime type → T4 GPU`
2. Run all cells top to bottom
3. Everything downloads and trains automatically

### Option 2: Local
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/Task3_Voice_Age_Emotion.git
cd Task3_Voice_Age_Emotion

# Install dependencies
pip install -r requirements.txt

# Download RAVDESS manually from https://zenodo.org/record/1188976
# Extract to ./RAVDESS/

# Open notebook
jupyter notebook notebooks/Task3_VoiceIQ.ipynb
```

---

## 📁 Project Structure

```
Task3_Voice_Age_Emotion/
│
├── notebooks/
│   └── Task3_VoiceIQ.ipynb      # Main Colab notebook (all-in-one)
│
├── src/
│   ├── features.py               # Feature extraction utilities
│   ├── models.py                 # CNN model architectures
│   └── inference.py              # Inference pipeline
│
├── models/                       # Saved model weights (after training)
│   ├── gender_model.h5
│   ├── age_model.h5
│   ├── emotion_model.h5
│   ├── scaler.pkl
│   ├── gender_encoder.pkl
│   └── emotion_encoder.pkl
│
├── assets/
│   └── pipeline.png              # Architecture diagram
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📦 Requirements

```
tensorflow>=2.12.0
librosa>=0.10.0
numpy>=1.23.0
scikit-learn>=1.2.0
ipywidgets>=8.0.0
soundfile>=0.12.0
matplotlib>=3.7.0
```

---

## 🖥️ GUI Features

Built with **ipywidgets** inside Google Colab:
- 📂 Upload any audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`)
- 🎵 Audio playback widget
- 📊 Gender badge + Age display + Pitch reading + Confidence score
- 🎭 Emotion probability bars (for senior citizens only)
- ⛔ Clear rejection message for female voices

---

## 👤 Author

**Raghuveer** | Internship Project — Task 3  
Built with Python · TensorFlow · librosa · ipywidgets
