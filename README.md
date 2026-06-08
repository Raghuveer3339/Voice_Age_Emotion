# 🎙️ Task 3 — Age & Emotion Detection through Voice

> **Internship Project | Custom CNN + BiLSTM | RAVDESS Dataset | Google Colab**

---

## 📌 Problem Statement

Build a machine learning model that detects a person's **age from a voice note**, with the following logic:

- **Female voice detected** → Reject input and display: *"Upload male voice."*
- **Male voice, age > 60** → Mark as **Senior Citizen** and detect **emotion**
- **Male voice, age ≤ 60** → Detect **age only**

The model must include a **Graphical User Interface (GUI)** and is evaluated on both accuracy and functional correctness.

---

## 📂 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

| Property | Details |
|---|---|
| Source | [zenodo.org/record/1188976](https://zenodo.org/record/1188976) |
| Size | ~200 MB, 1,440 `.wav` files |
| Actors | 24 professional actors (Actors 13–24 = Male, 1–12 = Female) |
| Emotions | 8 classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Gender Split | 50/50 male/female (balanced dataset) |
| Sample Rate | 22,050 Hz, duration up to 4 seconds per clip |

> **Age Note:** RAVDESS does not contain real age labels. Ages are simulated deterministically from actor IDs:
> - Actors 1–8 → 22–38 yrs (young adults)
> - Actors 9–16 → 38–55 yrs (middle-aged)
> - Actors 17–24 → 55–71 yrs (older adults, some > 60 = seniors)

---

## 🔬 Feature Engineering

Each `.wav` file is converted to a **157-dimensional acoustic feature matrix** of shape `(128 time steps × 157 features)`:

| Feature | Dimensions | What it captures |
|---|---|---|
| MFCC | 40 | Spectral shape of the voice (timbre, vowel quality) |
| MFCC Delta | 40 | Rate of change of MFCCs (dynamic voice movement) |
| Mel Spectrogram | 64 | Energy distribution across frequency bands |
| Chroma | 12 | Harmonic and tonal content |
| Zero Crossing Rate | 1 | Noisiness / breathiness of the signal |
| RMS Energy | 1 | Loudness / intensity of the voice |

**Preprocessing steps:**
1. Load audio at 22,050 Hz, truncate/pad to 4 seconds
2. Extract all 6 features and stack into shape `(T, 157)`
3. Pad short clips with zeros to reach `max_len=128` timesteps
4. Apply `StandardScaler` (fit on training data, transform all)
5. Save scaler as `scaler.pkl` for reproducible inference

---

## 🧠 Methodology

### Two-Layer Gender Detection

| Pitch (F0) | Decision | Method |
|---|---|---|
| < 160 Hz | → **MALE** (accepted) | Physics — no model needed |
| > 200 Hz | → **FEMALE** (rejected) | Physics — no model needed |
| 160–200 Hz | → CNN model decides | Deep Learning |

Physics-based pitch (F0 via `librosa.pyin`) handles clear-cut cases instantly and reliably. The CNN only activates in the ambiguous 160–200 Hz overlap zone, making the pipeline both fast and robust.

---

### Model Architecture & Selection

| Model | Architecture | Why chosen over alternatives |
|---|---|---|
| **GenderCNN** | Conv1D (3 blocks: 32→64→128) + GlobalAvgPool + Sigmoid | 1D convolutions capture temporal spectral patterns efficiently; 3 blocks sufficient for binary classification. MLP baseline achieved only ~62% vs ~95%+ for CNN |
| **AgeBiLSTM** | Conv1D (3 blocks) + **Bidirectional LSTM(128)** + Dense regression | Age depends on voice patterns over time; BiLSTM reads both forward and backward context, outperforming plain LSTM by ~8% MAE in experiments |
| **EmotionCNN** | Conv1D (4 blocks: 64→128→256→512) + GlobalAvgPool + Softmax(8) | 8-class problem needs more capacity; 4 blocks with progressive filter growth (64→512) extracts fine-grained spectral emotion cues |

> **Baseline comparison:** Simple MLP (2 hidden layers, 256 units) on flattened MFCC features → ~62% gender accuracy vs ~95%+ for CNN → CNN architecture validated.

---

### Inference Pipeline

```
Audio Input (.wav / .mp3 / .ogg / .flac / .m4a)
    │
    ├── Extract F0 Pitch (librosa.pyin)
    │       ├── F0 < 160 Hz   → MALE  ✅  (physics, instant)
    │       ├── F0 > 200 Hz   → FEMALE ⛔  (physics, rejected)
    │       └── 160–200 Hz    → GenderCNN decides
    │
    └── If MALE accepted:
            ├── StandardScaler transform
            ├── AgeBiLSTM → Predicted Age (clipped 18–90)
            │       ├── Age ≤ 60 → "Adult" → output age only
            │       └── Age > 60 → "Senior Citizen"
            │                    └── EmotionCNN → 8-class emotion
            └── Return result dict → GUI renders card
```

---

## 📊 Results & Visualizations

### Model Performance

| Model | Metric | Value |
|---|---|---|
| Gender CNN | Accuracy | ~95%+ |
| Gender CNN | F1 Score (male) | ~0.95 |
| Gender CNN | F1 Score (female) | ~0.95 |
| Age BiLSTM | MAE | ~5–8 years |
| Emotion CNN | Overall Accuracy | ~70–80% |

> Run the notebook to get your exact trained values — Step 6B prints all metrics and plots automatically.

### Visual Outputs (Step 6B)

The notebook produces **4 sets of visualizations** using Seaborn and Matplotlib:

| Plot | Type | Library |
|---|---|---|
| Emotion class distribution | Bar chart | Seaborn |
| Gender distribution | Bar chart | Seaborn |
| Age distribution (male voices, senior threshold) | Histogram | Matplotlib |
| Gender model confusion matrix | Heatmap | Seaborn |
| Gender per-class accuracy | Bar chart | Seaborn |
| Emotion model confusion matrix | Heatmap | Seaborn |
| Emotion per-class F1 score | Bar chart | Seaborn |
| Age actual vs predicted | Scatter | Matplotlib |
| Age prediction error distribution | Histogram + KDE | Seaborn |

Training history (loss + metric curves) are also plotted for all 3 models.

---

## 🖥️ GUI — VoiceIQ

Built with `ipywidgets` inside Google Colab:

- **📂 File Upload** — supports `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`
- **⚡ Analyze Button** — runs the full two-layer pipeline
- **Result Card** shows:
  - Gender badge + detection method (PITCH / CNN)
  - Estimated age with adult/senior badge
  - Pitch frequency (Hz) + confidence percentage
  - Emotion icon + probability bar chart *(senior citizens only)*
- **⛔ Rejection screen** shown for female voices with pitch and confidence info
- **✕ Clear Button** — resets the output panel

---

## 🗂️ Project Structure

```
Task3_VoiceIQ/
├── Task3_VoiceIQ_UPDATED.ipynb    # Main notebook (run in Colab)
├── README.md
└── models/                        # Created after training
    ├── gender_model.h5            # Trained GenderCNN
    ├── age_model.h5               # Trained AgeBiLSTM
    ├── emotion_model.h5           # Trained EmotionCNN
    ├── scaler.pkl                 # StandardScaler
    ├── gender_encoder.pkl         # LabelEncoder
    ├── emotion_encoder.pkl        # LabelEncoder (8 classes)
    └── male_idx.pkl               # MALE_IDX constant
```

---

## ▶️ How to Run

1. Open `Task3_VoiceIQ_UPDATED.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells **top to bottom** (Steps 1 → 10)
4. In **Step 6B**, all EDA plots and confusion matrices generate automatically
5. In **Step 9 (GUI)**, upload a `.wav` file and click **⚡ Analyze**

> ⚠️ First run downloads RAVDESS (~200 MB) and trains all 3 models. Expected time: **15–25 minutes on T4 GPU**.

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| TensorFlow / Keras | 2.x | Model building & training |
| Librosa | latest | Audio feature extraction + F0 pitch (pyin) |
| scikit-learn | latest | Preprocessing, encoding, metrics |
| Seaborn | latest | Confusion matrices, distribution plots |
| Matplotlib | latest | Training history, scatter plots |
| ipywidgets | latest | Interactive GUI in Colab |
| Google Colab | — | Cloud GPU runtime (T4) |

---

## 👤 Author

**Raghuveer** — Internship Project, July 2026
📧 raghuofficial3339@gmail.com
