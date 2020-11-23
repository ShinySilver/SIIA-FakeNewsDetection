import resources.datasets as data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import models
from keras import layers


def short_print(db, head):
    print(f"###############\n#    {head}\n###############\n")
    for key, value in db.items():
        print(f'###############\n# "{key}" (shape {np.array(value).shape}):\n{value[:min(len(value),50)]}\n')

db_raw = data.load_raw()
db = data.load_BoW()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(len(db['dictionnary']),)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_val_array = db["text_bow"][:2000]
y_val = db["class_id"][:2000]
partial_x_train_array = db["text_bow"][2000:]
partial_y_train_array = db["class_id"][2000:]

history = model.fit(partial_x_train_array,
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

plt.plot(epochs, loss, 'b.', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(epochs, acc, 'b.', label='Training accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()