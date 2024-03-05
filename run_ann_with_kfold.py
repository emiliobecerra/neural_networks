# import keras
# from keras.models import Sequential
# from keras import layers


# import pandas
# import numpy

# from sklearn.model

# dataset = pandas.read_csv("dataset.csv")

# dataset = dataset.sample(frac=1)


# target = dataset.iloc[:,-1].values
# data = dataset.iloc[:,:-1].values
# data = data/255.0


# machine = Sequential()



# machine.add(layers.Dense(512, 
# 			activation="sigmoid",
# 			input_shape=(data.shape[1],)
# 				))

# machine.add(layers.Dense(128, 
# 			activation="sigmoid"))

# machine.add(layers.Dense(64, 
# 			activation="sigmoid"))

# machine.add(layers.Dense(10, 
# 			activation="softmax"))


# machine.compile(optimizer="sgd", 
# 	loss="sparse_categorical_crossentropy",
# 	metrics=['accuracy'])

# machine.fit(data_training, target_training, epochs=30, batch_size=64)

# machine.fit(data, target, epochs=90, batch_size=64)

# new_target = numpy.argmax(machine.predict(new_data), axis = -1)


# #Simulating real world with new data
# new_data = pandas.read_csv("new_data.csv")
# filename_list = new_data.iloc[:,-1].values
# new_data = new_data.iloc[:,:-1].values
# new_data = new_data/255.0

# prediction = numpy.argmax(machine.predict(new_data), axis = -1)

# result = pandas.DataFrame()
# result['filename'] = filename_list
# result['prediction'] = predict

# print(results)


import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics



dataset = pandas.read_csv("dataset.csv")
dataset = dataset.sample(frac=1)


target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0

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
  machine.add(layers.Dense(512, 
              activation="relu", 
              input_shape=(data.shape[1],)  
              ))
  machine.add(layers.Dense(256, 
              activation="relu"))
  machine.add(layers.Dense(128, 
              activation="relu"))
  machine.add(layers.Dense(64, 
              activation="relu"))
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
    