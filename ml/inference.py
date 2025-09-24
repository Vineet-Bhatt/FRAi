import tensorflow as tf
import numpy as np

def load_model(path):
    return tf.keras.models.load_model(path)

def predict_classifier(model, x):  # x: (L,)
    x_in = x[np.newaxis, ..., np.newaxis].astype('float32')
    probs = model.predict(x_in)[0]
    pred = np.argmax(probs)
    return pred, probs

def saliency_map(model, x, class_index=None):
    # returns gradient magnitude per input frequency
    x_t = tf.convert_to_tensor(x[np.newaxis, ..., np.newaxis], dtype=tf.float32)
    x_t = tf.Variable(x_t)
    with tf.GradientTape() as tape:
        preds = model(x_t)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        score = preds[0, class_index]
    grads = tape.gradient(score, x_t)[0, :, 0].numpy()
    return np.abs(grads)
