import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

def baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
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

    # Train the model
    baseline = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/baseline_model.h5')
    with open('./data/baseline_history.json', 'w') as f:
        json.dump(baseline.history, f)

# increase number of kernels
def variant1(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    variant1 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant1_model.h5')
    with open('./data/variant1_history.json', 'w') as f:
        json.dump(variant1.history, f)


# increase kernel size
def variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(32, (5, 5), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    variant1 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant2_model.h5')
    with open('./data/variant2_history.json', 'w') as f:
        json.dump(variant1.history, f)

def plotting(filename):
    # load model
    # model = tf.keras.saving.load_model(filename)

    # Load the history
    with open(filename, 'r') as f:
        history = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename[:-5]+'_loss.png')
    plt.show()
    plt.close()

    # Plot the training/validation accuracy per epoch
    plt.plot(history['accuracy'], label='Training accuracy')
    plt.plot(history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename[:-5] + '_acc.png')
    plt.show()
    plt.close()

def comparison(filename1, filename2):
    # Load the history
    with open(filename1, 'r') as f:
        history1 = json.load(f)
    with open(filename2, 'r') as f:
        history2 = json.load(f)

    # Plot the training/validation loss per epoch
    plt.plot(history1['loss'], label='Training Loss 1', color='C0')
    plt.plot(history1['val_loss'], label='Validation Loss 1', color='C1')
    plt.plot(history2['loss'], label='Training Loss 2',linestyle='--', color='C0')
    plt.plot(history2['val_loss'], label='Validation Loss 2',linestyle='--', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename1[:-13] + '_' + filename2[7:-13] + '_loss.png')
    plt.show()
    plt.close()

    # Plot the training/validation accuracy per epoch
    plt.plot(history1['accuracy'], label='Training accuracy 1', color='C0')
    plt.plot(history1['val_accuracy'], label='Validation accuracy 1', color='C1')
    plt.plot(history2['accuracy'], label='Training accuracy 2',linestyle='--', color='C0')
    plt.plot(history2['val_accuracy'], label='Validation accuracy 2', linestyle='--', color='C1')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(filename1[:-13] + '_' + filename2[7:-13] + '_acc.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # create new sets 0.8/0.2
    train_num = int(0.8 * len(train_images))
    trainSet_images = train_images[:train_num]
    trainSet_labels = train_labels[:train_num]

    validSet_images = train_images[train_num:]
    validSet_labels = train_labels[train_num:]

    # Normalize pixel values (0 and 1)
    trainSet_images = trainSet_images.astype('float32') / 255
    validSet_images = validSet_images.astype('float32') / 255
    # Reshape the images to 28x28x1
    trainSet_images = np.expand_dims(trainSet_images, axis=-1)
    validSet_images = np.expand_dims(validSet_images, axis=-1)
    # Convert labels
    trainSet_labels = keras.utils.to_categorical(trainSet_labels, 10)
    validSet_labels = keras.utils.to_categorical(validSet_labels, 10)

    # baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename1 = './data/baseline_history.json'
    # plotting(filename1)

    # variant1(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    # filename2 = './data/variant1_history.json'
    # plotting(filename2)
    # comparison(filename1, filename2)

    variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    filename2 = './data/variant2_history.json'
    plotting(filename2)
    comparison(filename1, filename2)


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
    # plt.show()'./data/baseline_model.h5')