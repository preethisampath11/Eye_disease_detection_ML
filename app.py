import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import argparse
import sys

# Set random seed for reproducibility
tf.random.set_seed(42)

# Constants

IMG_SIZE = 224  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 20

# Paths
TRAIN_DIR = "Training_set"
VAL_DIR = "Validation_set"
TEST_DIR = "Test_set"

def load_data(csv_path, image_ext='.png'):
    """Load and prepare the dataset labels.

    Adds a `filename` column constructed from the `ID` column. If the ID
    values already contain an extension the code will use it directly.
    """
    df = pd.read_csv(csv_path)

    # Derive label columns: everything after ID and Disease_Risk
    label_columns = df.columns[2:]

    # Build filename column: if ID already has an extension, keep it; otherwise append image_ext
    id_str = df['ID'].astype(str)
    has_ext = id_str.str.contains(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", case=False, regex=True)
    df['filename'] = np.where(has_ext, id_str, id_str + image_ext)

    return df, label_columns

def create_model(num_classes):
    """Create and compile the MobileNetV2 model."""
    # Load MobileNetV2 with pre-trained weights (fall back if imagenet weights can't be loaded)
    try:
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    except Exception as e:
        print(f"Warning: failed to load imagenet weights ({e}); initialising MobileNetV2 without weights.")
        base_model = MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers for multi-label classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def create_data_generators():
    """Create data generators for training and validation."""
    # Use MobileNetV2 preprocess_input function which scales pixels to the range [-1,1]
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    return train_datagen, val_datagen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='.', help='Base path that contains Training_set/ Validation_set/ Test_set and CSVs')
    parser.add_argument('--image-ext', type=str, default='.png', help='Image file extension to append when building filenames')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    data_dir = args.data_dir
    image_ext = args.image_ext
    batch_size = args.batch_size
    epochs = args.epochs

    # If running on Kaggle, try to locate the dataset under /kaggle/input
    if os.path.exists('/kaggle/input'):
        # prefer a dataset that already contains the expected subfolders
        kaggle_base = '/kaggle/input'
        if os.path.exists(os.path.join(kaggle_base, TRAIN_DIR)):
            data_dir = kaggle_base
        else:
            # search each subfolder for Training_set
            for entry in os.listdir(kaggle_base):
                candidate = os.path.join(kaggle_base, entry)
                if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, TRAIN_DIR)):
                    data_dir = candidate
                    break

    # Load training and validation CSVs (try dataset subfolders first)
    train_csv = os.path.join(data_dir, TRAIN_DIR, 'RFMiD_Training_Labels.csv')
    val_csv = os.path.join(data_dir, VAL_DIR, 'RFMiD_Validation_Labels.csv')
    if not os.path.exists(train_csv):
        train_csv = os.path.join(data_dir, 'RFMiD_Training_Labels.csv')
    if not os.path.exists(val_csv):
        val_csv = os.path.join(data_dir, 'RFMiD_Validation_Labels.csv')

    train_df, label_columns = load_data(train_csv, image_ext=image_ext)
    val_df, _ = load_data(val_csv, image_ext=image_ext)

    num_classes = len(label_columns)
    print(f"Number of classes: {num_classes}")

    # Create model
    model = create_model(num_classes)

    # Create data generators
    train_datagen, val_datagen = create_data_generators()

    # Use the filename column (created from ID) because images are multi-label
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(data_dir, TRAIN_DIR),
        x_col="filename",
        y_col=label_columns.tolist(),
        class_mode="raw",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        validate_filenames=False
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=os.path.join(data_dir, VAL_DIR),
        x_col="filename",
        y_col=label_columns.tolist(),
        class_mode="raw",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
        validate_filenames=False
    )

    # Save outputs to Kaggle working dir when available
    output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    best_model_path = os.path.join(output_dir, 'best_model.h5')
    final_model_path = os.path.join(output_dir, 'final_model.h5')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor='val_loss'
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # Save the final model
    model.save(final_model_path)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_csv = os.path.join(output_dir, 'training_history.csv')
    hist_df.to_csv(hist_csv, index=False)

    print(f"Training completed successfully! Models and history saved to: {output_dir}")

if __name__ == "__main__":
    main()
