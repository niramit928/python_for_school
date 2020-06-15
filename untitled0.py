import keras
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from emnist import extract_training_samples
from emnist import extract_test_samples
import cv2
import sys
import os

# load train and test dataset
#def load_dataset2():

def load_dataset():
	# load dataset
	trainX, trainY = extract_training_samples('letters')
	testX, testY = extract_test_samples('letters')
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(27, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores = list()
	histories = {'accuracy': [],'val_accuracy': [] ,'loss': [], 'val_loss': []}
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	counter  = 0
	for train_ix, test_ix in kfold.split(dataX):
		counter += 1
		print("we are at fold number : " + str(counter))
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		for key in histories.keys():
			histories[key].extend(history.history[key])
	return scores, histories, model
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
		# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(histories['loss'], color='blue', label='train')
	pyplot.plot(histories['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.title('Classification Accuracy')
	pyplot.plot(histories['accuracy'], color='blue', label='train')
	pyplot.plot(histories ['val_accuracy'], color='orange', label='test')
	pyplot.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	print("finished loading")
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	print("finished normalization")
	# evaluate model
	scores, histories,model = evaluate_model(trainX, trainY)
	print("finished training")
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
	name = input("how do you want to name the new model?")
	model.save(name)
	return model
 
# entry point, run the test harness


def make_to_letter(result):
	final_result = np.argmax(result)
	options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	print(options[final_result])

def image_norm(img):
	img = cv2.imread(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("name",img)
	cv2.waitKey(0)
	dim = (28,28)
	img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
	img = img.astype('float32')
	img = img.reshape(1,28,28,1)
	img = img/255.0
	return img

print(" \n\n\nplease choose an action - type 'help' for help")
con = True
model = None
while(con):
	y = input()
	if y == "help":
		print("type 'train' to train a model \ntype 'load' to load  a model \ntype 'test' to test a model \ntype 'info' to get information on the model \ntype 'finish' to finish")
	elif y == "test":
		if model == None:
			print("you have to first load or train a model")
		else:
			path = input("enter path to image - ")
			img = image_norm(path)
			make_to_letter(model.predict(img))
	elif y == "load":

		name = input("please enter model path\n")
		try:

			model = keras.models.load_model(name)
			print("a model has been loaded")
		except:
			print("it seems there is no model there,type 'load' to try again")

	elif y == "train":
		print("training model")
		model = run_test_harness()
	elif y == "finish":
		print("shutting down")
		con = False
	elif y == "info":
		print("this model is a deep learning neural network that uses the emnist extended database to train, its goal is to be able to detect written english letters")
	else:
		print("command not understood, please try again, you can type 'help' for help")






