import matplotlib.pyplot as plt
from tensorflow import keras

model_cnf = {
    'activation_1': 'relu',
    'activation_2': 'softmax',
    'first_layer_neurons': 128,
    'second_layer_neurons': 10,
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy'
}

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(model_cnf['first_layer_neurons'], activation=model_cnf['activation_1']),
    keras.layers.Dense(model_cnf['second_layer_neurons'], activation=model_cnf['activation_2'])
])

opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, momentum=0.9)

model.compile(optimizer=model_cnf['optimizer'],
              loss=model_cnf['loss'],
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model accuracy for: {model_cnf["activation_1"]}, {model_cnf["activation_2"]}, '
          f'{model_cnf["first_layer_neurons"]}, {model_cnf["second_layer_neurons"]}, '
          f'{model_cnf["optimizer"]}, {model_cnf["loss"]}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Model loss for: {model_cnf["activation_1"]}, {model_cnf["activation_2"]}, '
          f'{model_cnf["first_layer_neurons"]}, {model_cnf["second_layer_neurons"]}, '
          f'{model_cnf["optimizer"]}, {model_cnf["loss"]}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
