# CNN Image Classification

This project classifies wether the animal inside an image is a cat or a dog using Convolutional Neural Network (CNN). 

## Dataset
[Dataset](https://drive.google.com/file/d/1ck6VYKZEI9lVq9mLHjvClCy0AOX61_d2/view?usp=sharing) is divided into two part:
- `dataset/training_set` – used for training and validation.
- `dataset/test_set` – used for evaluating model.

I resized the images into 128x128 px and splitted training_set with ratio of 80:20 for training and validation.

## Data Augmentation
Model is trained with augmented images to improve generalization:
- `RandomFlip (horizontal)`
- `RandomRotation (0.2)`
- `RandomZoom (0.2)`

## Model Architecture
```python
model = tf.keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, 5, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
```

## Training
Model is trained using:
- Loss function: `binary_crossentropy`
- Optimizer: `Adam`
- Metrics: `accuracy`

## Model Evaluation
Result of model evaluation:
- Test Accuracy: 0.8600
- Test Loss: 0.3161
