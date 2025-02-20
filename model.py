from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Paths to dataset
dataset_path =  # Replace with your dataset path
train_dir = os.path.join(dataset_path, path to the folder)  # TRAIN folder path
test_dir = os.path.join(dataset_path, path to the folder)    # TEST folder path

# Image dimensions
IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"  # Binary classification for O and R
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load pre-trained model
pretrained_model_path =  # Replace with the path to your pre-trained model
try:
    base_model = load_model(pretrained_model_path)
    print("Pre-trained model loaded successfully.")
except Exception as e:
    print(f"Error loading pre-trained model: {e}")
    exit()

# Modify the model for binary classification
base_model.pop()  # Remove the last layer
for layer in base_model.layers[:-1]:
    layer.trainable = False  # Freeze all layers except the final ones

# Add new layers for binary classification
model = Sequential(base_model.layers)
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))  # Output for binary classification

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Define a checkpoint to save the best model
checkpoint = ModelCheckpoint(
    "model_checkpoint.h5",  # Save as fine-tuned model
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,  # Adjust based on your computational resources
    callbacks=[checkpoint],
    verbose=1
)

# Save the final model
model.save("model_final.h5")
print("Fine-tuned model saved as 'model_final.h5'.")

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
