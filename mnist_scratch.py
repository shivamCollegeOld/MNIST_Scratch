import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import random
import idx2numpy as idx
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return sigmoid(x)*(1-sigmoid(x))

class Network:
	def __init__(self,layers):
		self.num_layers = len(layers)
		self.layers = layers
		self.biases = [np.random.normal(0,0.0001,(y,1)) for y in layers[1:]]
		self.weights = [np.random.normal(0,0.0001,(y,x)) for (x,y) in zip(layers[:-1],layers[1:])]
		self.neurons = [np.zeros((l,1),dtype=float) for l in layers]

	def feedforward(self,inp):
		self.neurons[0] = inp 
		for l in range(1,self.num_layers):
			inp = sigmoid(np.dot(self.weights[l-1],inp)+self.biases[l-1])
			self.neurons[l] = inp

		return inp 

	def backprop(self,y):
		# y is the desired output

		nabla_b = [np.zeros(b.shape,dtype=float) for b in self.biases]
		nabla_w = [np.zeros(w.shape,dtype=float) for w in self.weights]
		nabla_a = [np.zeros(acti.shape,dtype=float) for acti in self.neurons]

		# print(nabla_w[0].shape)
		# print(nabla_b[0].shape)

		nabla_a[-1] = 2.0*(self.neurons[-1]-y)
		for l in range(2,self.num_layers+1):
			acti_grad = []
			for j in range(self.layers[-l]):
				temp = 0.0
				for k in range(self.layers[-l+1]):
					zk = np.dot(self.weights[-l+1][k],self.neurons[-l])+self.biases[-l+1][k]
					temp += nabla_a[-l+1][k]*self.weights[-l+1][k][j]*sigmoid_der(zk)
				acti_grad.append(temp)
			nabla_a[-l] = np.asarray(acti_grad)


		for l in range(1,self.num_layers):
			weight_grad = []
			bias_grad = []
			for j in range(self.layers[-l]):
				zj = np.dot(self.weights[-l][j],self.neurons[-l-1])+self.biases[-l][j]
				temp_b = nabla_a[-l][j]*sigmoid_der(zj)
				temp_w = []
				for k in range(self.layers[-l-1]):
					acti_k = self.neurons[-l-1][k]
					dummy = nabla_a[-l][j]*sigmoid_der(zj)*acti_k
					temp_w.append(dummy)
				# weight_grad = np.append(weight_grad,temp_w,axis=0)
				weight_grad.append(temp_w)
				# bias_grad = np.append(bias_grad,temp_b,axis=0)
				bias_grad.append(temp_b)
			weight_grad = np.asarray(weight_grad)
			bias_grad = np.asarray(bias_grad)
			weight_grad = np.reshape(weight_grad,self.weights[-l].shape)
			bias_grad = np.reshape(bias_grad,self.biases[-l].shape)
			nabla_w[-l] = weight_grad
			nabla_b[-l] = bias_grad
		
		# print("shape of weights = {} , shape of grad_weights = {}".format(self.weights[0].shape,nabla_w[0].shape))
		# print("shape of biases = {} , shape of grad_biases = {}".format(self.biases[0].shape,nabla_b[0].shape))
		return nabla_w,nabla_b

	def update_mini_batch(self,mini_batch,eta):
		m = len(mini_batch)
		for sample in mini_batch:
			y = sample[1]
			nabla_w,nabla_b = self.backprop(y)

			self.weights = [w-(eta/m)*nw for w,nw in zip(self.weights,nabla_w)]
			self.biases = [b-(eta/m)*nb for b,nb in zip(self.biases,nabla_b)]

	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for x,y in test_data]

		ans = 0
		for x,y in test_results:
			if x == y:
				ans += 1 

		return ans
	
	def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		n = len(training_data)
		if test_data:
			n_test = len(test_data) 
			
		
		for j in range(epochs):
			random.shuffle(training_data)
			
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			i = 1
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)
				print("Mini batch {}/{} done in epoch {}".format(i,len(training_data)/mini_batch_size,j))
				i += 1 
		
			if test_data:
				print("Epoch {}: {} / {}".format(j,self.evaluate(test_data),n_test))
			else:
				print("Epoch {} complete".format(j))




training_images = idx.convert_from_file('train-images.idx3-ubyte')
training_labels = idx.convert_from_file('train-labels.idx1-ubyte')

training_data = []
for x,y in zip(training_images,training_labels):
	training_data.append((x,y))

for i in range(len(training_data)):
	digit = training_data[i][1]
	converted = np.zeros((10,1),dtype=float)
	converted[digit] = 1.0
	flattened = training_data[i][0].flatten()
	flattened = flattened/255
	flattened = np.reshape(flattened,(784,1))
	training_data[i] = (flattened,converted)


model = Network([784,20,10])
# output = model.feedforward(training_data[0][0])
# print(output)
# print(output.shape)
training_data_new = training_data[:5000]
test_data = training_data[500:700]
model.SGD(training_data_new,30,10,0.1,test_data)















