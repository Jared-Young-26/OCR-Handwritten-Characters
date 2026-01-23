import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

'''
60,000 training samples, 10,000 test samples, 70,000 total samples
Each sample is 28 x 28 x 1 of 8 bit integers 0 - 255
'''

def print_samples(sample):
    """Used to see how data is structured"""
    RED = "\033[31m"
    RESET = "\033[0m"
    assert len(sample) == 28
    for row in sample:
        assert len(row) == 28
        for num in row:
            s = f"{num:03d}"
            if num != 0:
                print(f"{RED}{s}{RESET} ", end = '')
            else:
                print(s, end = ' ')
        print()

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = np.expand_dims(x_train, -1) # shape (N, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)  # shape (N, 28, 28, 1)

val_split = 0.1
val_count = int(len(x_train) * val_split)
x_val, y_val = x_train[:val_count], y_train[:val_count]
x_train, y_train = x_train[val_count:], y_train[val_count:]

# The actual model
model = models.Sequential([
    # Input 28 x 28 x 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), # 32 3 x 3 convolutional filters -> 26 x 26 x 32
    layers.MaxPooling2D((2,2)), # -> 13 x 13 x 32
    layers.Conv2D(64, (3,3), activation='relu'), # 64 3 x 3 convolutional filters -> 11 x 11 x 64
    layers.MaxPooling2D((2,2)), # -> 5 x 5 x 64
    layers.Flatten(), # Combine all 64 5 x 5 layers into 1 "array"
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Try to avoid overfitting
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_val, y_val)
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")
