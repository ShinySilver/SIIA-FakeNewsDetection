import numpy as np
import resources.datasets as data
import matplotlib.pyplot as plt
from datatype.db_raw import RawDB
from datatype.db_bag_of_word import BagOfWordDB
from classifiers.neural_network_classifier import NeuralNetworkClassifier
from classifiers.deep_neural_classifier import DeepNeuralClassifier
from sklearn.model_selection import train_test_split

def plot(history,label='NN',color='r'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    plt.subplot(211)
    plt.plot(epochs, loss, color+'.', label='Training loss')
    plt.plot(epochs, val_loss, color+'-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, acc, color+'.', label=label)
    plt.plot(epochs, val_acc, color+'-')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.legend()
   
db = BagOfWordDB()

modelNN = NeuralNetworkClassifier(db.input_shape()[1])
modelDNN = DeepNeuralClassifier(db.input_shape()[1])

print("============================",db.input_shape(),"============================")


x_val_array, y_val, partial_x_train_array, partial_y_train_array = \
            train_test_split(db.X,db.y,train_size=0.7)


historyDNN = modelDNN.train(partial_x_train_array,
                    partial_y_train_array,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val_array, y_val))

#plot the results :
history_dictDNN = historyDNN.history

historyNN = modelNN.train(partial_x_train_array,
                    partial_y_train_array,
                    epochs=100,
                    batch_size=512,
                    validation_data=(x_val_array, y_val))

#plot the results :
plot(historyNN,label='NN',color='r')
plot(historyDNN,label='DNN',color='g')
plt.show()

