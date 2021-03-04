import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import random


def sigmoid(x):
	return 1/(1+np.exp(-x))


class Network:
	def __init__(self,layers):
		self.num_layers = len(layers)
		self.layers = layers
		self.biases = [np.random.randn(y,1) for y in layers[1:]]
		self.weights = [np.random.randn(y,x) for (x,y) in zip(layers[:-1],layers[1:])]

	def feedforward(self,inp):
		for b,w in zip(self.biases,self.weigths):
			inp = sigmoid(np.dot(w,inp)+b)

		return inp

	def backprop(self,x,y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		

	def update_mini_batch(self,mini_batch,eta):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]




	def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		if test_data:
			n_test = len(test_data) 
			n = len(training_data)
		
		for j in range(epochs):
			random.shuffle(training_data)
			
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta) 
		
			if test_data:
				print("Epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {} complete".format(j))




