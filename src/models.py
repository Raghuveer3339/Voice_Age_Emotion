"""
Custom CNN model architectures for voice-based age & emotion detection.
All models are built from scratch using Keras (no pre-trained weights).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_gender_model(input_shape):
    """
    Custom Conv1D CNN for binary gender classification (male/female).

    Architecture:
        3 Conv1D blocks (32 → 64 → 128 filters)
        GlobalAveragePooling → Dense(64) → Dense(1, sigmoid)

    Training note:
        Sigmoid output = P(class index 1).
        LabelEncoder sorts alphabetically: female=0, male=1.
        So output > 0.5 → male.

    Args:
        input_shape: tuple (timesteps, n_features)
    Returns:
        Compiled Keras model
    """
    inp = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv1D(32, 5, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv1D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv1D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    m = models.Model(inp, out, name='GenderCNN')
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return m


def build_age_model(input_shape):
    """
    Custom CNN + Bidirectional LSTM for age regression.
    Trained on male voices only.

    Architecture:
        3 Conv1D blocks → BiLSTM(128) → Dense(128) → Dense(64) → Dense(1)

    Args:
        input_shape: tuple (timesteps, n_features)
    Returns:
        Compiled Keras model (regression, loss=MSE)
    """
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x   = layers.Bidirectional(layers.LSTM(128))(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dense(64,  activation='relu')(x)
    out = layers.Dense(1,   activation='linear')(x)

    m = models.Model(inp, out, name='AgeBiLSTM')
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m


def build_emotion_model(input_shape, n_classes: int = 8):
    """
    Custom 4-block Conv1D CNN for 8-class emotion classification.
    Only runs on senior citizen (age > 60) male voices.

    Architecture:
        4 Conv1D blocks (64 → 128 → 256 → 512 filters)
        GlobalAvgPool → Dense(256) → Dense(128) → Dense(n_classes, softmax)

    Args:
        input_shape: tuple (timesteps, n_features)
        n_classes:   number of emotion classes (8 for RAVDESS)
    Returns:
        Compiled Keras model (categorical, loss=sparse_categorical_crossentropy)
    """
    inp = layers.Input(shape=input_shape)

    x = inp
    for filters in [64, 128, 256, 512]:
        x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.25)(x)

    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(256, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    x   = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)

    m = models.Model(inp, out, name='EmotionCNN')
    m.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m
