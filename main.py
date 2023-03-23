import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

def baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    print(len(trainSet_images), len(trainSet_labels), len(validSet_images), len(validSet_labels))
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # Normalize pixel values (0 and 1)
    trainSet_images = trainSet_images.astype('float32') / 255
    validSet_images = validSet_images.astype('float32') / 255
    # Reshape the images to 28x28x1
    trainSet_images = np.expand_dims(trainSet_images, axis=-1)
    validSet_images = np.expand_dims(validSet_images, axis=-1)
    # Convert labels
    trainSet_labels = keras.utils.to_categorical(trainSet_labels, 10)
    validSet_labels = keras.utils.to_categorical(validSet_labels, 10)

    # Train the model
    baseline = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))
    # Save the training history
    with open('./baseline.json', 'w') as f:
        json.dump(baseline.history, f)

def plotting(filename):
    # Load the history
    with open(filename, 'r') as f:
        baseline = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(baseline['loss'], label='Training Loss')
    plt.plot(baseline['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the training/validation accuracy per epoch
    plt.plot(baseline['accuracy'], label='Training accuracy')
    plt.plot(baseline['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # create new sets 0.8/0.2
    train_num = int(0.8 * len(train_images))
    trainSet_images = train_images[:train_num]
    trainSet_labels = train_labels[:train_num]

    validSet_images = train_images[train_num:]
    validSet_labels = train_labels[train_num:]

    baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels)

    # filename = './baseline.json'
    # plotting(filename)






    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()