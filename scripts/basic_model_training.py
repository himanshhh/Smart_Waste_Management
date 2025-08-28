from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

model = Sequential([
    InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

#compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#train the model
epochs = 10
start_time = time.time()
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    class_weight=class_weight_dict
)
print("Training time:", int(time.time() - start_time), "seconds")

