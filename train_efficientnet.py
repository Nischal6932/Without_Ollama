import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

dataset_path = "/Users/nischalmittal/Desktop/PlantVillage"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation"
)

# Explicit input layer
inputs = tf.keras.Input(shape=(224,224,3))

base_model = EfficientNetB0(
    weights=None,
    include_top=False,
    input_tensor=inputs
)

base_model.trainable = True

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(train_data.num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=5)

model.save("plant_disease_efficientnet.keras")