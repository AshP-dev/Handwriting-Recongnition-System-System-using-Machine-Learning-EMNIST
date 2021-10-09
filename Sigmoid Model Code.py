import tensorflow as tf
from tensorflow import keras
import gradio
import gradio as gr
from urllib.request import urlretrieve
import os

urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5", "mnist-model.h5")

#
(x_train, y_train),(x_test, y_test)= tf.keras.datasets.mnist.load_data()
x_train = x_train/ 255.0,
x_test = x_test/ 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape =(28,28)),
    tf.keras.layers.Dense(392,activation ='relu'),
    tf.keras.layers.Dense(256,activation ='relu'),
    tf.keras.layers.Dense(128,activation ='relu'),
    tf.keras.layers.Dense(64,activation ='relu'),
    tf.keras.layers.Dense(32,activation ='relu'),
    tf.keras.layers.Dense(10,activation = 'sigmoid')])

model.compile(optimizer ='adam', loss ='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(x_train,y_train, validation_data = (x_test, y_test), epochs = 50)


def recognize_digit(image):
    image = image.reshape(1, -1)
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

im = gradio.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=True, source="canvas")

iface = gr.Interface(
    recognize_digit, 
    im, 
    gradio.outputs.Label(num_top_classes=3),
    #live=True,
    interpretation="default",
    capture_session=True,
)

iface.test_launch()

if __name__ == "__main__":
    iface.launch()
