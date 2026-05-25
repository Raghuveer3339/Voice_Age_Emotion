"""
Feature extraction utilities for voice analysis.
Used by the training notebook and inference pipeline.
"""

import numpy as np
import librosa


MAX_LEN = 128  # timesteps per audio clip

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm',    '03': 'happy',    '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust',  '08': 'surprised'
}


def extract_features(path: str, max_len: int = MAX_LEN):
    """
    Extract 157-dimensional acoustic feature matrix from an audio file.

    Features per timestep:
        - 40 MFCCs
        - 40 Delta MFCCs
        - 64 Mel Spectrogram bands (log scale)
        - 12 Chroma features
        - 1  Zero Crossing Rate
        - 1  RMS Energy
        Total: 157

    Returns:
        np.ndarray of shape (max_len, 157) or None if file is unreadable.
    """
    try:
        y, sr = librosa.load(path, sr=22050, duration=4.0)
        if len(y) < sr * 0.5:
            return None  # skip clips shorter than 0.5s

        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta  = librosa.feature.delta(mfcc)
        mel    = librosa.power_to_db(
                     librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64),
                     ref=np.max)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
        zcr    = librosa.feature.zero_crossing_rate(y)
        rms    = librosa.feature.rms(y=y)

        feat = np.vstack([mfcc, delta, mel, chroma, zcr, rms]).T  # (T, 157)

        if feat.shape[0] < max_len:
            pad  = np.zeros((max_len - feat.shape[0], feat.shape[1]))
            feat = np.vstack([feat, pad])
        else:
            feat = feat[:max_len]

        return feat.astype(np.float32)

    except Exception:
        return None


def get_pitch(path: str) -> float:
    """
    Estimate mean fundamental frequency (F0) of a voice recording using pyin.

    Male voices:   typically  85–160 Hz
    Female voices: typically 165–255 Hz
    Overlap zone:            160–200 Hz  (use CNN model here)

    Returns:
        float: mean F0 in Hz, or 0.0 if detection fails.
    """
    try:
        y, sr = librosa.load(path, sr=22050)
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        if voiced is not None:
            f0_voiced = f0[voiced]
        else:
            f0_voiced = f0
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        return float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    except Exception:
        return 0.0


def simulate_age(actor_id: int) -> float:
    """
    Simulate age from RAVDESS actor ID (since RAVDESS has no real age labels).
    Actors 17-24 may produce age > 60 so the emotion model gets training signal.
    """
    np.random.seed(actor_id * 13)
    if actor_id <= 8:
        return float(np.random.randint(22, 38))
    elif actor_id <= 16:
        return float(np.random.randint(38, 55))
    else:
        return float(np.random.randint(55, 72))
