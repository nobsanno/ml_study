import tensorflow as tf
from tensorflow import keras
import os

def down_kaggle_pet_images():
    os.system(r'curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip')
    os.system(r'unzip -q kagglecatsanddogs_3367a.zip')

def remove_illegal_jpg():
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

def prepare_train_data(imgdir, image_size, batch_size, sigen=False):
    if (sigen):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            imgdir,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            imgdir,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            imgdir,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            imgdir,
            labels='inferred',
            label_mode='categorical',
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=batch_size,
        )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    return (train_ds, val_ds)
