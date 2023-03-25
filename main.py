import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.datasets import fashion_mnist
from PIL import Image
import matplotlib.pyplot as plt

def baseline(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
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
        layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # Train the model
    variant1 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant1_model.h5')
    with open('./data/variant1_history.json', 'w') as f:
        json.dump(variant1.history, f)


# change activation
def variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='valid', activation='tanh'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # Train the model
    variant2 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant2_model.h5')
    with open('./data/variant2_history.json', 'w') as f:
        json.dump(variant2.history, f)

# change layer types
def variant3(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    # Model architecture
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])


    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    variant3 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant3_model.h5')
    with open('./data/variant3_history.json', 'w') as f:
        json.dump(variant3.history, f)

# change padding
def variant4(trainSet_images, trainSet_labels, validSet_images, validSet_labels):
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    variant4 = model.fit(trainSet_images, trainSet_labels, batch_size=128, epochs=15,
                         validation_data=(validSet_images, validSet_labels))

    # Save the model and training history
    model.save('./data/variant4_model.h5')
    with open('./data/variant4_history.json', 'w') as f:
        json.dump(variant4.history, f)


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

def testing(filename):
    model = tf.keras.models.load_model(filename)

    # Preprocess the image
    image = Image.open("./images.jpeg").convert('L')
    image = image.resize((28, 28))  # Resize the image to (28, 28)
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)  # Reshape the image to (1, 28, 28, 1) to match the model's input shape

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    print("Predicted class:", predicted_class)

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
    filename0 = './data/baseline_history.json'
    # plotting(filename0)

    # variant1(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    # filename1 = './data/variant1_history.json'
    # plotting(filename1)
    # comparison(filename0, filename1)

    # variant2(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    # filename2 = './data/variant2_history.json'
    # plotting(filename2)
    # comparison(filename0, filename2)

    # variant3(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    # filename3 = './data/variant3_history.json'
    # plotting(filename3)
    # comparison(filename0, filename3)

    # variant4(trainSet_images, trainSet_labels, validSet_images, validSet_labels)
    # filename4= './data/variant4_history.json'
    # plotting(filename4)
    # comparison(filename0, filename4)

    testing('./data/variant3_model.h5')

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