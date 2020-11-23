import numpy as np
import resources.datasets as data
import matplotlib.pyplot as plt
from datatype.db_raw import RawDB
from datatype.db_bag_of_word import BagOfWordDB
from classifiers.neural_network_classifier import NeuralNetworkClassifier
from classifiers.deep_neural_classifier import DeepNeuralClassifier



db = BagOfWordDB()

modelNN = NeuralNetworkClassifier(db.input_shape()[1])
modelDNN = DeepNeuralClassifier(db.input_shape()[1])

print("============================",db.input_shape(),"============================")

x_val_array = db.X()[:2000]
y_val = db.y()[:2000]
partial_x_train_array = db.X()[2000:]
partial_y_train_array = db.y()[2000:]


history = modelDNN.train(partial_x_train_array,
                    partial_y_train_array,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val_array, y_val))

#plot the results :
history_dict = history.history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.subplot(211)
plt.plot(epochs, loss, 'b.', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(212)
plt.plot(epochs, acc, 'b.', label='Training accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()