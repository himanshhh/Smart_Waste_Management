from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3


# BEST MODEL
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20
DROPOUT_RATE = 0.3

base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model_inception = Model(inputs=base_model.input, outputs=output)


#unfreeze some top layers of InceptionV3
fine_tune_at = 249  # Typically the last few inception blocks
base_model.trainable = True

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

#Re-compile with lower learning rate
model_inception.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

fine_tune_epochs = 5
total_epochs = EPOCHS + fine_tune_epochs

#fine-tune training
fine_tune_history = model_inception.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        ModelCheckpoint('best_inception_finetuned.h5', save_best_only=True)
    ],
    class_weight=class_weight_dict
)