from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
import time


#get number of classes
num_classes = len(train_gen.class_indices)
print("Number of classes after dropping 'Other':", num_classes)

#build improved model
model_improved = Sequential([
    InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    #Conv Block 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001),
           kernel_initializer='glorot_uniform', padding='valid'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.5),

    #Conv Block 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001),
           kernel_initializer='glorot_uniform', padding='valid'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001),
           kernel_initializer='glorot_uniform', padding='valid'),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001),
           kernel_initializer='glorot_uniform', padding='valid'),
    MaxPooling2D(pool_size=(2, 2), padding='valid'),
    Dropout(0.5),

    #dense block
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Multi-class classification
])

#compile the model
model_improved.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model_improved.summary()

#train the improved model
epochs = 10
start_time = time.time()

print("Training time:", int(time.time() - start_time), "seconds")

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

#callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    LearningRateScheduler(lr_schedule)
]

#train
history = model_improved.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
)