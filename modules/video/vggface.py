from keras_vggface.vggface import VGGFace
from keras import Model

img_width, img_height = 224, 224

vgg_model = VGGFace(input_shape=(img_height, img_width, 3), include_top=True, weights='vggface')
inter_layer_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc6').output)

def extractFeature(image_list):
    feature = inter_layer_model.predict(image_list)
    return feature
