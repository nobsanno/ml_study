import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

imgfile = "../../training/image_classification_from_scratch/PetImages/Cat/6779.jpg"

img = plt.imread(imgfile)
plt.imshow(img)
plt.show()

image_size = (180, 180)
img = keras.preprocessing.image.load_img(
    imgfile, target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

modelfile = '../../training/image_classification_from_scratch/model_of_image_classification_for_dog_and_cat.h5'
model = load_model(modelfile)
predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)
