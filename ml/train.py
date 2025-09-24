import numpy as np
import tensorflow as tf
from ml.models import build_classifier, build_autoencoder
from sklearn.model_selection import train_test_split
import os

# Assume you have X (num_samples, input_length) and y (labels or None)
def load_dataset_npz(path):
    data = np.load(path)  # X: (N,L) and y: (N,) optional
    X = data['X']
    y = data['y'] if 'y' in data else None
    return X, y

def train_classifier(X, y, input_length, num_classes, save_dir):
    X = X[..., np.newaxis].astype('float32')
    y_cat = tf.keras.utils.to_categorical(y, num_classes)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.15, random_state=42)
    model = build_classifier(input_length, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'classifier_best.h5'), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100, batch_size=32, callbacks=callbacks)
    model.save(os.path.join(save_dir, 'classifier_savedmodel'))  # SavedModel format
    return model

def train_autoencoder(X, input_length, save_dir):
    X = X[..., np.newaxis].astype('float32')
    X_train, X_val = train_test_split(X, test_size=0.15, random_state=42)
    model = build_autoencoder(input_length)
    model.compile(optimizer='adam', loss='mse')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(save_dir,'autoencoder_best.h5'), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=200, batch_size=32, callbacks=callbacks)
    model.save(os.path.join(save_dir, 'autoencoder_savedmodel'))
    return model

if __name__ == "__main__":
    # Example usage: python ml/train.py data.npz models/
    import sys
    data_path = sys.argv[1]
    save_dir = sys.argv[2]
    os.makedirs(save_dir, exist_ok=True)
    X, y = load_dataset_npz(data_path)
    input_length = X.shape[1]
    if y is not None:
        num_classes = len(np.unique(y))
        train_classifier(X, y, input_length, num_classes, save_dir)
    # train autoencoder on healthy subset or whole dataset if using unsupervised:
    train_autoencoder(X, input_length, save_dir)
