
import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics


dataset = pandas.read_csv("dataset.csv")
dataset = dataset.sample(frac=1)
#The dataset is flattened, we need to transform it back to a matrix (back to a picture).

target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0

#28 by 28 matrixes and each one of them contains one element. 
data = data.reshape(-1, 28, 28, 1)

split_number = 4
kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)


results_accuracy = []
results_confusion_matrix = []

for training_index, test_index in kfold_object.split(data):
  data_training = data[training_index]
  target_training = target[training_index]
  data_test = data[test_index]
  target_test = target[test_index]
  
  machine = Sequential()
  #We want to add another layer, and that layer is the convolution operation. 
  #32 by trial and error, kernel size 3x3 is the smallest size, the value of the next pixel is affected by the nearest 9 pixels,
  #making the kernel size very big makes the model flexible but it doesn't tell you a lot of information
  #Reminder: we are doing convolution and we want to identify the digit by the edges of the digit and the shape of them. We also want to specify
  #any "dirt" that may get in the way of identification issues. 
  machine.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
  machine.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))


  
  #We now need a layer to flatten back
  machine.add(layers.Flatten())

  machine.add(layers.Dense(128, activation="relu"))
  machine.add(layers.Dense(64, activation="relu"))
  machine.add(layers.Dense(10, activation="softmax"))

  machine.compile(optimizer="sgd", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=['accuracy'])
    
  machine.fit(data_training, target_training, epochs=30, batch_size=64)
  
  new_target = numpy.argmax(machine.predict(data_test), axis=-1)
  results_accuracy.append(metrics.accuracy_score(target_test, new_target))
  results_confusion_matrix.append(metrics.confusion_matrix(target_test, new_target))
  
print(results_accuracy)
for i in results_confusion_matrix:
  print(i)
    