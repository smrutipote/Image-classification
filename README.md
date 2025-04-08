# Image Classification Model with Deep Learning

## Overview
This project demonstrates how to build an image classification model using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained on the Fashion MNIST dataset, a popular alternative to the classic MNIST dataset but with images of clothing items.

---

## Dataset: Fashion MNIST
Fashion MNIST consists of 70,000 grayscale images (28x28 pixels) in 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

- **60,000** images for training
- **10,000** images for testing

---

## Setup & Dependencies
```bash
pip install numpy matplotlib tensorflow scikit-learn plotly
```

---

## 1. Load and Explore the Data
```python
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training shape:", x_train.shape)
print("Test shape:", x_test.shape)
```

---

## 2. Visualize Sample Images
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig = make_subplots(rows=2, cols=5, subplot_titles=[class_names[y_train[i]] for i in range(10)])

for i in range(10):
    row = i // 5 + 1
    col = i % 5 + 1
    fig.add_trace(go.Heatmap(z=x_train[i], colorscale='gray'), row=row, col=col)
    fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)

fig.update_layout(height=600, width=1000, title_text="Sample Images")
fig.show()
```

---

## 3. Preprocess the Data
```python
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
```

---

## 4. Build the CNN Model
```python
from tensorflow.keras import layers

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()
```

---

## 5. Train the Model with Callbacks
```python
early_stop = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)
```

---

## 6. Evaluate the Model
```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
```

---

## 7. Classification Report
```python
import numpy as np
from sklearn.metrics import classification_report

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes, target_names=class_names))
```

---

## Summary
This project demonstrated how to:
- Load and explore image data
- Preprocess grayscale images for CNN input
- Build and train a convolutional neural network
- Use callbacks for smarter training
- Evaluate model performance with accuracy and a classification report

Achieved a baseline accuracy of ~88% using a simple CNN architecture. This model can be improved further with data augmentation, deeper networks, or transfer learning.

---

## Author
**Smruti Pote** — AI enthusiast building smart, sustainable solutions at the intersection of technology and everyday life.
