import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

from ann import load_model, predict_classes
from segments import extract_edge_segment_features, compute_edges


# -------------------------------------------------------
# Preprocessing: convert arbitrary image -> MNIST format
# -------------------------------------------------------

def preprocess_image(path, invert=True):
    """
    Load any PNG/JPG, convert to MNIST-style 28x28 grayscale.
    invert=True is necessary if white-on-black images are used.
    """
    img = Image.open(path).convert("L")
    img = img.resize((28, 28))

    img = np.array(img).astype("float32") / 255.0
    if invert:
        img = 1.0 - img   # MNIST digits are white on black

    return img


# -------------------------------------------------------
# ANN PREDICTION
# -------------------------------------------------------

def predict_ann(img28):
    """
    img28: 2D numpy array (28x28) normalized 0–1
    """
    feats = extract_edge_segment_features(img28, grid_rows=4, grid_cols=4)
    feats = feats.reshape(1, -1)  # (1, 16) for ANN input

    weights, biases = load_model("custom_ann_model.npz")
    pred = predict_classes(feats, weights, biases)[0]
    return int(pred)


# -------------------------------------------------------
# TF CNN PREDICTION
# -------------------------------------------------------

def predict_cnn(img28):
    model = tf.keras.models.load_model("tf_cnn_model.keras")

    x = img28.reshape(1, 28, 28, 1)
    probs = model.predict(x)
    return int(np.argmax(probs))


# -------------------------------------------------------
# Full side-by-side prediction
# -------------------------------------------------------

def predict_digit(path):
    img = preprocess_image(path)

    ann_pred = predict_ann(img)
    cnn_pred = predict_cnn(img)

    print("\n===== PREDICTION RESULTS =====")
    print(f"ANN model prediction: {ann_pred}")
    print(f"TensorFlow CNN prediction: {cnn_pred}")
    return ann_pred, cnn_pred


if __name__ == "__main__":
    # Example usage:
    # python predict.py
    test_path = "example_digit.png"
    predict_digit(test_path)
