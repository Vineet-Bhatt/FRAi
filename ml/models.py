from django.db import models
import tensorflow as tf
from tensorflow.keras import layers, models

def build_classifier(input_length, num_classes):
    inp = layers.Input(shape=(input_length,1))
    x = layers.Conv1D(32, 11, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out, name="fra_classifier")
    return model

def build_autoencoder(input_length, latent_dim=64):
    inp = layers.Input(shape=(input_length,1))
    x = layers.Conv1D(32,11,activation='relu',padding='same')(inp)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16,7,activation='relu',padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    z = layers.Dense(latent_dim, activation='relu')(x)

    # decoder
    x = layers.Dense((input_length//4)*16, activation='relu')(z)
    x = layers.Reshape((input_length//4, 16))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32,7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(1,11, activation='linear', padding='same')(x)
    model = models.Model(inp, x, name='fra_autoencoder')
    return model
