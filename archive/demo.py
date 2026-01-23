import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from segments import extract_edge_segment_features, segment_edge_image, compute_edges
from ann import train_network, predict_proba, predict_classes

def to_one_hot(y, num_classes):
    N = len(y)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    for i, label in enumerate(y):
        onehot[i, label] = 1.0
    return onehot

if __name__ == "__main__":
    # 1. Load handwritten digits (8x8 images)
    digits = load_digits()
    images = digits.images      # (N, 8, 8)
    labels = digits.target      # (N,)

    print("Loaded digits dataset:", images.shape, "labels:", labels.shape)

    # 2. Extract edge-segment features for each image
    #    (here we use a 4x4 grid → 16 features)
    grid_rows, grid_cols = 4, 4
    X_features = []
    for img in images:
        feats = extract_edge_segment_features(img, grid_rows, grid_cols, threshold=0.2)
        X_features.append(feats)
    X_features = np.array(X_features)
    print("Feature shape:", X_features.shape)  # (N, 16)

    # 3. Train/test split
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_features, labels, test_size=0.2, random_state=42
    )

    num_classes = len(np.unique(labels))
    y_train = to_one_hot(y_train_raw, num_classes)
    y_test  = to_one_hot(y_test_raw,  num_classes)

    # 4. Define network
    hidden_layers = [32, 16]   # you can tune these
    num_outputs   = num_classes
    learning_rate = 0.5
    bias_value    = 0.1
    max_epochs    = 2000
    error_thresh  = 0.01

    # 5. Train
    weights, biases, log = train_network(
        X_train, y_train,
        hidden_layers=hidden_layers,
        num_outputs=num_outputs,
        learning_rate=learning_rate,
        bias_value=bias_value,
        max_epochs=max_epochs,
        error_threshold=error_thresh
    )

    print("\nTraining log (sample):")
    for line in log[:10]:
        print(line)
    if len(log) > 10:
        print("...")

    # 6. Evaluate
    train_preds = predict_classes(X_train, weights, biases)
    test_preds  = predict_classes(X_test,  weights, biases)

    train_acc = np.mean(train_preds == y_train_raw)
    test_acc  = np.mean(test_preds  == y_test_raw)

    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    # Example outputs
    print("\nSample predictions (test set):")
    for i in range(10):
        print(f"True: {y_test_raw[i]}, Pred: {test_preds[i]}")
