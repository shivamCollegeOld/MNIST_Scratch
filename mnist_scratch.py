import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import random


def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return np.exp(-x)*(sigmoid(x)**2)

class Network:
	def __init__(self,layers):
		self.num_layers = len(layers)
		self.layers = layers
		self.biases = [np.random.randn(y,1) for y in layers[1:]]
		self.weights = [np.random.randn(y,x) for (x,y) in zip(layers[:-1],layers[1:])]
		self.neurons = [np.zeros(l,1) for l in layers]

	def feedforward(self,inp):
		self.neurons[0] = inp 
		for l in range(1,self.num_layers):
			inp = sigmoid(np.dot(self.weights[l-1],inp)+self.biases[l-1])
			self.neurons[l] = inp

		return inp 

	def backprop(self,y):
		# y is the desired output

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_a = [np.zeros(acti.shape) for acti in self.neurons]

		nabla_a[-1] = 2*(self.neurons[-1]-y)
		for l in range(2,self.num_layers+1):
			acti_grad = np.array([])
			for j in range(self.layers[-l]):
				temp = 0
				for k in range(self.layers[-l+1]):
					zk = np.dot(self.weights[-l+1][k],self.neurons[-l])+self.biases[-l+1][k]
					temp += nabla_a[-l+1][k]*self.weights[-l+1][k][j]*sigmoid_der(zk)
				acti_grad = np.append(acti_grad,temp,axis=0)
			nabla_a[-l] = acti_grad


		for l in range(1,self.num_layers):
			weight_grad = np.array([])
			bias_grad = np.array([])
			for j in range(self.layers[-l]):
				zj = np.dot(self.weights[-l][j],self.neurons[-l-1])+self.biases[-l][j]
				temp_b = nabla_a[-l][j]*sigmoid_der(zj)
				temp_w = np.array([])
				for k in range(self.layers[-l-1]):
					acti_k = self.neurons[-l-1][k]
					dummy = nabla_a[-l][j]*sigmoid_der(zj)*acti_k
					temp_w = np.append(temp_w,dummy,axis=0)
				weight_grad = np.append(weight_grad,temp_w,axis=0)
				bias_grad = np.append(bias_grad,temp_b,axis=0)
			nabla_w[-l] = weight_grad
			nabla_b[-l] = bias_grad
					
		return nabla_w,nabla_b

	def update_mini_batch(self,mini_batch,eta):
		m = len(mini_batch)
		for sample in mini_batch:
			y = sample.label
			nabla_w,nabla_b = self.backprop(y)

			self.weights = [w-(eta/n)*nw for w,nw in zip(self.weights,nabla_w)]
			self.biases = [b-(eta/n)*nb for b,nb in zip(self.biases,nabla_b)]

	def evaluate(self,test_data):
		test_results = [(np.argmax(feedforward(x)),y) for x,y in test_data]

		return sum((x==y) for x,y in test_results)

	
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




