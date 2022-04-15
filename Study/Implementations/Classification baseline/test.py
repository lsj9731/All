from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import requests

def show(idx, title):
  plt.figure(figsize=(12, 3))
  plt.imshow(test_images[idx].reshape(28,28))
  plt.axis('off')
  plt.title('\n\n{}'.format(title), fontdict={'size': 16})
  plt.show()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
test_images = test_images / 255.0

# reshape for feeding into the model
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

rando = random.randint(0,len(test_images)-1)
show(rando, 'An Example Image: {}'.format(class_names[test_labels[rando]]))

data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


# send data using POST request and receive prediction result
headers = {"content-type": "application/json"}
json_response = requests.post('http://169.56.89.210:8501/v1/models/mnist_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']
# show first prediction result
for i in range(0,3):
    show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i]))

# set model version and send data using POST request and receive prediction result
json_response = requests.post('http://169.56.89.210:8501/v1/models/mnist_model/versions/3:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

# show all prediction result
for i in range(0,3):
    show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
    class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i]))