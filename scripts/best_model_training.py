from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedKFold

# BEST MODEL
# IMG_HEIGHT = 150
# IMG_WIDTH = 150
# BATCH_SIZE = 32
# EPOCHS = 20
# DROPOUT_RATE = 0.3

#final model with tuned dropout
model_final = Sequential([
    InputLayer(input_shape=(150, 150, 3)),

    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model_final.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

model_final.summary()

def lr_schedule(epoch):
    return 0.001 * (0.1 ** int(epoch / 10))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('final_best_model.h5', save_best_only=True, monitor='val_loss'),
    LearningRateScheduler(lr_schedule)
]

history = model_final.fit(
    train_gen,  # from your earlier train flow
    validation_data=val_gen,  # from your valid folder
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=callbacks
)