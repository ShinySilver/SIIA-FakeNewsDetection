import numpy as np

import resources
import resources.datasets as data
import matplotlib.pyplot as plt
from datatype.db_raw import RawDB
from datatype.db_bag_of_word import BagOfWordDB
from datatype.db_tf_idf import TfIdfDB
from classifiers.neural_network_classifier import NeuralNetworkClassifier
from classifiers.deep_neural_classifier import DeepNeuralClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf

def plot(history,label='NN',color='r'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    plt.subplot(211)
    plt.plot(epochs, loss, color+'--', label=f'Training loss for {label}')
    plt.plot(epochs, val_loss, color+'-', label=f'Validation loss for {label}')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, acc, color+'--', label=f'Accuracy for {label}')
    plt.plot(epochs, val_acc, color+'-', label=f'Valid° accuracy for {label}')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.legend()
    
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))
    
dbs = (('TF-IDF',TfIdfDB()),('BoW', BagOfWordDB()))
models = (('NN', NeuralNetworkClassifier), ('DNN', DeepNeuralClassifier))

colors = ('r','g','b','c','m','y')
color_id = 0

for model_type in models:
    for db_type in dbs:
        db = db_type[1]
        model = model_type[1](db.input_shape()[1])
        print(db_type[0])
        print("============================",db.input_shape(),"============================")


        x_val_array, partial_x_train_array, y_val, partial_y_train_array = \
                    train_test_split(db.X(),db.y(),train_size=0.7)

        partial_x_train_array = convert_sparse_matrix_to_sparse_tensor(partial_x_train_array)
        x_val_array = convert_sparse_matrix_to_sparse_tensor(x_val_array)

        history = model.train(partial_x_train_array,
                            partial_y_train_array,
                            epochs=100,
                            batch_size=512,
                            validation_data=(x_val_array, y_val))
        plot(history,label=f'{model_type[0]} with {db_type[0]}', color=colors[color_id])
        color_id+=1
plt.show()

