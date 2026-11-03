
import tensorflow as tf
import json

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)

# Load dataset
data = tf.keras.utils.image_dataset_from_directory(
    "Train",
    image_size=(64, 64),
    batch_size=2
)

class_names = data.class_names

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(data, epochs=50)

# Save without optimizer state
model.save("CAT.keras")

# Save classes
with open("classes.json", "w") as f:
    json.dump(class_names, f)

print("Model saved successfully!")
