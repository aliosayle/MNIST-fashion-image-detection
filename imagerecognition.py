import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
#model.fit(train_images, train_labels, epochs=100)

# Save the model
#model.save('fashion_mnist_model.h5')
#print("Model saved successfully.")


loaded_model = keras.models.load_model('fashion_mnist_model.h5')

# Load the test image
test_image_path = '/Users/aliossaily/Desktop/AI/20230707_170128.jpg'  # Provide the path to your test image
test_image = Image.open(test_image_path).convert('L')  # Convert to grayscale
test_image = test_image.resize((28, 28))  # Resize to match the input shape
test_image = np.array(test_image) / 255.0  # Normalize the pixel values

# Make predictions on a test image
#test_image_index = 979
#test_image = test_images[test_image_index]
test_image = np.expand_dims(test_image, axis=0)
predictions = loaded_model.predict(test_image)

# Convert predictions to class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]
print(predicted_class_name)

# Display the test image and prediction
plt.figure()
plt.imshow(test_image, cmap=plt.cm.binary)
plt.title("Predicted: " + predicted_class_name)
plt.xticks([])
plt.yticks([])
plt.show()
