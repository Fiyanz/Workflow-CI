import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 4
CLASS_NAMES = ['MP (kutu daun)', 'BT (kutu kebul)', 'T (thrips)', 'C (ulat)']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str,
                        default='/kaggle/input/datasets/indraagustian/red-chili-pepper-pests-dataset')
    parser.add_argument('--preprocessed-dir', type=str, default='red_chili_pepper_pests_preprocessed')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def load_preprocessed_data(preprocessed_dir):
    train_files = np.load(os.path.join(preprocessed_dir, 'train_files.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(preprocessed_dir, 'train_labels.npy'))
    val_files = np.load(os.path.join(preprocessed_dir, 'val_files.npy'), allow_pickle=True)
    val_labels = np.load(os.path.join(preprocessed_dir, 'val_labels.npy'))
    test_files = np.load(os.path.join(preprocessed_dir, 'test_files.npy'), allow_pickle=True)
    test_labels = np.load(os.path.join(preprocessed_dir, 'test_labels.npy'))
    class_weight_dict = np.load(os.path.join(preprocessed_dir, 'class_weights.npy'), allow_pickle=True).item()
    return train_files, train_labels, val_files, val_labels, test_files, test_labels, class_weight_dict


def load_raw_data(dataset_dir):
    import glob
    from sklearn.model_selection import train_test_split

    CLASS_MAP = {
        'kutu-daun': 0, 'kutu-kebul': 1, 'thrips': 2, 'thrips-baru': 2, 'ulat': 3,
    }

    def get_class_from_filename(filename):
        basename = os.path.basename(filename)
        prefix = basename.split('--')[0].lower()
        if prefix in CLASS_MAP:
            return CLASS_MAP[prefix]
        label_path = filename.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    return int(first_line.split()[0])
        raise ValueError(f"Cannot determine class for: {filename}")

    def collect_images_and_labels(image_dir, max_images=1000):
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:max_images]
        labels, valid_files = [], []
        for img_path in image_files:
            try:
                label = get_class_from_filename(img_path)
                labels.append(label)
                valid_files.append(img_path)
            except ValueError:
                pass
        return [str(f) for f in valid_files], labels

    train_files, train_labels = collect_images_and_labels(os.path.join(dataset_dir, 'train', 'images'))
    val_files_1, val_labels_1 = collect_images_and_labels(os.path.join(dataset_dir, 'val', 'images'))
    val_files_2, val_labels_2 = collect_images_and_labels(os.path.join(dataset_dir, 'val', 'valid', 'images'))
    val_files = val_files_1 + val_files_2
    val_labels = val_labels_1 + val_labels_2
    test_files, test_labels = collect_images_and_labels(os.path.join(dataset_dir, 'test', 'images'))

    all_train_files = train_files + val_files
    all_train_labels = train_labels + val_labels
    train_files_final, val_files_final, train_labels_final, val_labels_final = train_test_split(
        all_train_files, all_train_labels, test_size=0.15, random_state=42
    )

    class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

    return train_files_final, train_labels_final, val_files_final, val_labels_final, test_files, test_labels, class_weight_dict


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")


def parse_image(file_path, label):
    img = tf.io.read_file(tf.cast(file_path, tf.string))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, list(IMAGE_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(tf.cast(label, tf.int32), tf.float32)
    label = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
    return img, label


def augment_image(image, label, augmentation):
    return augmentation(image), label


def create_dataset(file_paths, labels, batch_size=BATCH_SIZE, shuffle=False, augment=False, augmentation=None):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment and augmentation is not None:
        ds = ds.map(lambda x, y: augment_image(x, y, augmentation), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = keras.Input(shape=input_shape, name='input_image')

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.1)(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.15)(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.2)(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.SpatialDropout2D(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs, outputs, name='pest_classifier_cnn')


def main():
    args = parse_args()

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'file:./mlruns'))
    mlflow.set_experiment('red_chili_pepper_pests_ci')

    if os.path.exists(args.preprocessed_dir):
        (train_files, train_labels, val_files, val_labels,
         test_files, test_labels, class_weight_dict) = load_preprocessed_data(args.preprocessed_dir)
    else:
        (train_files, train_labels, val_files, val_labels,
         test_files, test_labels, class_weight_dict) = load_raw_data(args.dataset_dir)

    augmentation = build_augmentation()
    train_ds = create_dataset(train_files, train_labels, batch_size=args.batch_size, shuffle=True, augment=True, augmentation=augmentation)
    val_ds = create_dataset(val_files, val_labels, batch_size=args.batch_size, shuffle=False, augment=False)
    test_ds = create_dataset(test_files, test_labels, batch_size=args.batch_size, shuffle=False, augment=False)

    model = build_cnn_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name='ci_pipeline_run'):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_accuracy)

        model.save('model_pest_classification.keras')
        mlflow.log_artifact('model_pest_classification.keras')

        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print("CI Pipeline training complete!")


if __name__ == '__main__':
    main()
