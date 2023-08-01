from keras.models import load_model
from keras_preprocessing import image
import numpy as np


class Classification:
    def __init__(self):
        self.prediction_model = None
        self.generator_model = None

    def init_prediction_modul(self, prediction_model_name, generator_model_name):
        self.prediction_model = load_model(prediction_model_name)
        self.generator_model = load_model(generator_model_name, compile=False)
        self.generator_model.compile()

    def predict_from_array(self, array_to_predict):
        return self.prediction_model.predict(array_to_predict)

    def de_shadow_image(self, input_image):
        return self.generator_model.predict(input_image, steps=1)

    @staticmethod
    def convert_image_to_model_array(convertable_image):
        array = image.img_to_array(convertable_image)
        array = np.expand_dims(array, axis=0)
        array /= 255
        return array
