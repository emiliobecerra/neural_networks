import keras
from keras.models import Sequential
from keras import layers


import pandas
import numpy

dataset = pandas.read_csv("dataset.csv")

#shuffle dataset
dataset = dataset.sample(frac=1)


#-1 to include all possible columns. 
target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values
data = data/255.0


#This is a sequential model
machine = Sequential()

#First layer needs to input
#Last layer needs to predict

# Dense defines the type of layer: Each node is connected
# We pick 512 arbitrarially as a starting number of features. There's going to be trial and errors
# Sigmoid means the sigmoid activation function: digits are between 0 and 1.
# Data.shape gets the number of pixels from the data: 784
# activation function "relu" is a substitute for sigmoid. 
# relu is non-linear, accepts inputs from negative infinity to positive infinity, isn't bounded to f(x)=1 so it obtains more information f(x) -> infinity
machine.add(layers.Dense(512, 
			activation="sigmoid",
			input_shape=(data.shape[1],)
				))

# Second layer, 128 features
machine.add(layers.Dense(128, 
			activation="sigmoid"))

#Third layer, 64 features
machine.add(layers.Dense(64, 
			activation="sigmoid"))

# Final layer, our images could be 0-9, so 10
machine.add(layers.Dense(10, 
			activation="softmax"))

# We need to inject an optimizer to turn this into a machine.
# Sgd, stochastic gradient descent.
# Sparse categorical cross entropy
# Accuracy, backwards propogation: 
# we check the outcome with our dataset. 
# We use this to adjust our "beta" later on to improve the accuracy
machine.compile(optimizer="sgd", 
	loss="sparse_categorical_crossentropy",
	metrics=['accuracy'])

machine.fit(data, target, epochs=90, batch_size=64)
#Batch_size: Consult 64 pictures out of the 42 thousand of pictures we have.
#Epochs: how many rounds of backward propogation until you stop.
#Rule, these two numbers multiplied together should be bigger than your data size.

# Accuracy: 0.9187
# The accuracy score talking about the best fitting model
# When we do validation, the validation accuracy could be small since we picked 64 pictures.

# New_target is the prediction
# Argmax, we're going to get a bunch of probabilities, we are still going to want the maximum probability. 
new_target = numpy.argmax(machine.predict(new_data), axis=-1)


#Simulating real world with new data
new_data = pandas.read_csv("new_data.csv")
filename_list = new_data.iloc[:,-1].values
new_data = new_data.iloc[:,:-1].values
new_data = new_data/255.0

prediction = numpy.argmax(machine.predict(new_data), axis = -1)

result = pandas.DataFrame()
result['filename'] = filename_list
result['prediction'] = predict

print(results)



# import keras
# from keras.models import Sequential
# from keras import layers

# import pandas
# import numpy


# dataset = pandas.read_csv("dataset.csv")
# dataset = dataset.sample(frac=1)


# target = dataset.iloc[:,-1].values
# data = dataset.iloc[:,:-1].values
# data = data/255.0

# machine = Sequential()
# machine.add(layers.Dense(512, 
#             activation="relu", 
#             input_shape=(data.shape[1],)  
#             ))
# machine.add(layers.Dense(128, 
#             activation="relu"))
# machine.add(layers.Dense(64, 
#             activation="relu"))
# machine.add(layers.Dense(10, activation="softmax"))
# machine.compile(optimizer="sgd", 
#                 loss="sparse_categorical_crossentropy", 
#                 metrics=['accuracy'])
  

# machine.fit(data, target, epochs=30, batch_size=64)





# new_data = pandas.read_csv("new_data.csv")
# filename_list = new_data.iloc[:,-1].values
# new_data = new_data.iloc[:,:-1].values
# new_data = new_data/255.0

# prediction = numpy.argmax(machine.predict(new_data), axis=-1)

# result = pandas.DataFrame()
# result['filename'] = filename_list
# result['prediction'] = prediction

# print(result)















