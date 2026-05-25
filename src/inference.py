"""
Inference pipeline for Task 3: Age & Emotion Detection through Voice.

Two-layer gender detection:
  Layer 1 — Physics (pitch F0): handles clear male/female cases
  Layer 2 — CNN model: handles ambiguous pitch zone (160-200 Hz)
"""

import pickle
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from features import extract_features, get_pitch

# Gender detection thresholds (Hz)
PITCH_MALE_MAX   = 160.0   # F0 below this → definitely male
PITCH_FEMALE_MIN = 200.0   # F0 above this → definitely female
# F0 between 160-200 Hz → use CNN model

EMOTION_ICONS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry':   '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}


class VoiceAnalyzer:
    """Loads trained models and runs the full inference pipeline."""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir    = model_dir
        self.gender_model = None
        self.age_model    = None
        self.emotion_model= None
        self.scaler       = None
        self.gender_enc   = None
        self.emotion_enc  = None
        self.male_idx     = None
        self._loaded      = False

    def load(self):
        """Load all models and encoders from disk."""
        d = self.model_dir
        self.gender_model  = tf.keras.models.load_model(f'{d}/gender_model.h5')
        self.age_model     = tf.keras.models.load_model(f'{d}/age_model.h5')
        self.emotion_model = tf.keras.models.load_model(f'{d}/emotion_model.h5')

        with open(f'{d}/scaler.pkl',         'rb') as f: self.scaler      = pickle.load(f)
        with open(f'{d}/gender_encoder.pkl', 'rb') as f: self.gender_enc  = pickle.load(f)
        with open(f'{d}/emotion_encoder.pkl','rb') as f: self.emotion_enc = pickle.load(f)
        with open(f'{d}/male_idx.pkl',       'rb') as f: self.male_idx    = pickle.load(f)

        self._loaded = True
        print(f"✅ Models loaded. Gender classes: {list(self.gender_enc.classes_)}")

    def predict(self, file_path: str, verbose: bool = True) -> dict:
        """
        Run full pipeline on an audio file.

        Returns:
            dict with keys:
                rejected, message, gender, pitch_hz, gender_confidence,
                detection_method, age, is_senior, emotion, emotion_probs
        """
        if not self._loaded:
            raise RuntimeError("Call .load() first.")

        # ── Feature extraction ──────────────────────────────────────────
        feat = extract_features(file_path)
        if feat is None:
            return {'rejected': True, 'message': '⛔ Audio too short or unreadable.',
                    'pitch_hz': 0, 'gender_confidence': 0}

        T, F        = feat.shape
        feat_scaled = self.scaler.transform(feat.reshape(1, -1)).reshape(1, T, F)

        # ── Pitch (physics-based, reliable) ────────────────────────────
        pitch_hz = get_pitch(file_path)

        # ── Layer 1: physics gender decision ───────────────────────────
        if pitch_hz > 0 and pitch_hz < PITCH_MALE_MAX:
            pred_gender = 'male'
            gender_conf = min(99.0, 60.0 + (PITCH_MALE_MAX - pitch_hz) / PITCH_MALE_MAX * 40.0)
            method      = 'pitch'

        elif pitch_hz > PITCH_FEMALE_MIN:
            pred_gender = 'female'
            gender_conf = min(99.0, 60.0 + (pitch_hz - PITCH_FEMALE_MIN) / PITCH_FEMALE_MIN * 40.0)
            method      = 'pitch'

        else:
            # ── Layer 2: CNN model for ambiguous zone ──────────────────
            raw_prob     = float(self.gender_model.predict(feat_scaled, verbose=0)[0][0])
            is_male_prob = raw_prob if self.male_idx == 1 else (1.0 - raw_prob)
            pred_gender  = 'male' if is_male_prob >= 0.5 else 'female'
            gender_conf  = (is_male_prob if pred_gender == 'male'
                            else 1.0 - is_male_prob) * 100.0
            method       = 'cnn'

        # ── Reject female ────────────────────────────────────────────────
        if pred_gender == 'female':
            result = {
                'rejected': True,
                'message': '⛔  Female voice detected. Please upload a MALE voice.',
                'gender': 'female',
                'pitch_hz': round(pitch_hz, 1),
                'gender_confidence': round(gender_conf, 1),
                'detection_method': method,
                'age': None, 'is_senior': None,
                'emotion': None, 'emotion_probs': {}
            }

        else:
            # ── Age estimation ───────────────────────────────────────────
            age_raw  = float(self.age_model.predict(feat_scaled, verbose=0)[0][0])
            age_pred = float(np.clip(age_raw, 18, 90))
            is_senior= bool(age_pred > 60)

            # ── Emotion (senior citizens only) ───────────────────────────
            emotion, emotion_probs = None, {}
            if is_senior:
                em_probs = self.emotion_model.predict(feat_scaled, verbose=0)[0]
                em_idx   = int(np.argmax(em_probs))
                emotion  = self.emotion_enc.classes_[em_idx]
                emotion_probs = {
                    c: round(float(p) * 100, 1)
                    for c, p in zip(self.emotion_enc.classes_, em_probs)
                }

            result = {
                'rejected': False, 'message': None,
                'gender': 'male',
                'pitch_hz': round(pitch_hz, 1),
                'gender_confidence': round(gender_conf, 1),
                'detection_method': method,
                'age': round(age_pred, 1),
                'is_senior': is_senior,
                'emotion': emotion,
                'emotion_probs': emotion_probs
            }

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, r: dict):
        print("\n" + "═" * 55)
        if r['rejected']:
            print(f"  ⛔  REJECTED — Female Voice")
            print(f"  Pitch         : {r['pitch_hz']} Hz")
            print(f"  Method        : {r.get('detection_method','').upper()}")
            print(f"  Confidence    : {r['gender_confidence']:.1f}%")
        else:
            sr_tag = "👴 SENIOR CITIZEN" if r['is_senior'] else "👨 ADULT"
            print(f"  ✅  MALE Voice Accepted")
            print(f"  Pitch         : {r['pitch_hz']} Hz  ({r.get('detection_method','').upper()})")
            print(f"  Confidence    : {r['gender_confidence']:.1f}%")
            print(f"  Age           : ~{r['age']} yrs  →  {sr_tag}")
            if r['is_senior'] and r['emotion']:
                icon = EMOTION_ICONS.get(r['emotion'], '🎭')
                print(f"  Emotion       : {icon}  {r['emotion'].capitalize()}")
                print("  Probabilities :")
                for em, p in sorted(r['emotion_probs'].items(), key=lambda x: -x[1])[:5]:
                    bar = '█' * int(p / 5)
                    print(f"    {em:<12} {bar:<20} {p:.1f}%")
        print("═" * 55)


# ── CLI usage ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        sys.exit(1)

    analyzer = VoiceAnalyzer(model_dir='../models')
    analyzer.load()
    analyzer.predict(sys.argv[1])
