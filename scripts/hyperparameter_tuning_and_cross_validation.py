from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedKFold

#hyperparameters to tune
batch_sizes = [32, 64]
dropout_rates = [0.3, 0.5]
epochs_list = [20, 30]

IMG_HEIGHT = 150
IMG_WIDTH = 150

#read annotation CSV
train_dir = "/content/trash_data/train"
train_df = pd.read_csv(os.path.join(train_dir, '_annotations.csv'))
train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(train_dir, x))
train_df['class'] = train_df['class'].astype(str)

#encode labels for StratifiedKFold
y = train_df['class'].astype('category').cat.codes.to_numpy()

#K-Fold Cross-validation setup
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_val_acc = 0
best_model_path = "best_model_tuned.h5"

#loop
for batch_size in batch_sizes:
    for dropout_rate in dropout_rates:
        for num_epochs in epochs_list:
            fold = 1
            print(f"\n=== Trying batch_size={batch_size}, dropout_rate={dropout_rate}, epochs={num_epochs} ===")
            val_scores = []

            for train_idx, val_idx in kf.split(train_df, y):
                train_data = train_df.iloc[train_idx].copy()
                val_data = train_df.iloc[val_idx].copy()

                # Data Generators
                train_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, horizontal_flip=True)
                val_gen = ImageDataGenerator(rescale=1./255)


                train_flow = train_gen.flow_from_dataframe(
                    dataframe=train_data,
                    x_col='filename',
                    y_col='class',
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=batch_size,
                    class_mode='categorical'
                )
                val_flow = val_gen.flow_from_dataframe(
                    dataframe=val_data,
                    x_col='filename',
                    y_col='class',
                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                    batch_size=batch_size,
                    class_mode='categorical'
                )

                num_classes = len(train_flow.class_indices)

                # Build model
                def build_model():
                    model = Sequential([
                        InputLayer(input_shape=(150, 150, 3)),

                        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                        MaxPooling2D((2, 2)),
                        Dropout(dropout_rate),

                        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                        MaxPooling2D((2, 2)),
                        Dropout(dropout_rate),

                        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                        MaxPooling2D((2, 2)),
                        Dropout(dropout_rate),

                        Flatten(),
                        Dense(512, activation='relu'),
                        Dropout(dropout_rate),
                        Dense(num_classes, activation='softmax')
                    ])
                    model.compile(
                        optimizer=Adam(learning_rate=0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    return model

                model = build_model()

                # Class weights
                y_train_codes = train_data['class'].astype('category').cat.codes.to_numpy()
                weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train_codes),
                    y=y_train_codes
                )
                weight_dict = dict(enumerate(weights))

                # Callbacks
                def lr_schedule(epoch):
                    return 0.001 * (0.1 ** int(epoch / 10))

                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
                    LearningRateScheduler(lr_schedule)
                ]

                # Train model
                history = model.fit(
                    train_flow,
                    validation_data=val_flow,
                    epochs=num_epochs,
                    class_weight=weight_dict,
                    callbacks=callbacks,
                    verbose=0
                )

                val_acc = max(history.history['val_accuracy'])
                print(f"Fold {fold} - Val Accuracy: {val_acc:.4f}")
                val_scores.append(val_acc)
                fold += 1

            avg_val_acc = np.mean(val_scores)
            print(f"Average Val Accuracy: {avg_val_acc:.4f}")

            # Save best model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                model.save(best_model_path)
                best_params = (batch_size, dropout_rate, num_epochs)

print(f"\n Best Model Saved: {best_model_path}")
print(f"Best Params -> Batch Size: {best_params[0]}, Dropout: {best_params[1]}, Epochs: {best_params[2]}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")