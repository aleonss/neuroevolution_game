import random
import numpy as np
import matplotlib.pyplot as plt


class SigmoidNeuron:

	def __init__(self,weights=[],bias=0):
		self.weights=weights
		self.bias=bias

	def mrand(self,n_input):
		w = []
		for i in range(0, n_input):
			w.append(random.uniform(-2.0, 2.0))
		self.weights = w
		self.bias = random.uniform(-2.0, 2.0)


	def feed(self,x):
		summ = 0.0
		for i,w in enumerate(self.weights):
			summ+= w*x[i]
		return 1.0 / (1.0 + np.exp(-summ -self.bias) )

	def upgrade_wb(self, delta, input, rate):
		for j, w in enumerate(self.weights):
			self.weights[j] += (rate * delta * input[j])
		self.bias += (rate * delta)

	def to_str(self):
		res = "\tweights:\t"+ self.weights.__str__()+"\n"
		res += "\t\tbias:\t"+ self.bias.__str__()+"\n"

		return res


class NeuronLayer:
	def __init__(self,n_input=0,n_neurons=0):
		neurons =[]
		for i in range(0,n_neurons):
			sn = SigmoidNeuron()
			sn.mrand(n_input)
			neurons.append(sn)
		self.neurons=neurons

	def feed(self,x):
		res = []
		for n in self.neurons:
			res.append(n.feed(x))
		return res


	def upgrade_wb(self, delta, input, rate):
		for i,n in enumerate(self.neurons):
			self.neurons[i].upgrade_wb(delta[i], input, rate)
		return self.feed(input)


	def get_weight(self):
		res = []
		for n in self.neurons:
			res.append(n.weights)
		return res

	def get_bias(self):
		res = []
		for n in self.neurons:
			res.append(n.bias)
		return res

	def to_str(self):
		res = ""
		for i,n in enumerate(self.neurons):
			res+= "\tn"+i.__str__() +"\n\t"
			res+= n.to_str()
		return res


class NeuralNetwork:

	def __init__(self,layers):
		self.layers=layers

	def feed(self,input):
		res=[]
		x = input
		for i,l in enumerate(self.layers):
			res = l.feed(x)
			x = res
		return res



	def forward_feeding(self,input):
		res = []
		outputs =[]
		x = input
		for i,l in enumerate(self.layers):
			res = l.feed(x)
			outputs.append(res)
			x = res

		return res, outputs



	def error_backpropagation(self, outputs, expected_output):
		n_layers = self.layers.__len__()
		output = outputs[outputs.__len__()-1]
		output_layer = self.layers[n_layers-1]

		#print("expected_output  ",expected_output)
		#print("output           ",output)


		error=[]
		delta=[]
		deltam=[]

		for i,n in enumerate(output_layer.neurons):
			e=(expected_output[i] - output[i])
			d = e*(output[i] * (1.0 - output[i]))
			error.append(e)
			delta.append(d)

		deltam.append(delta)

		for i in range(2,n_layers+1):
			il = n_layers -i
			inl = n_layers -i +1

			ndelta= delta
			delta = []
			l = self.layers[il]
			nl = self.layers[inl]
			loutput = outputs[il]
			nweight = nl.get_weight()

			for i, n in enumerate(l.neurons):
				e = 0.0
				for j,w in enumerate(nweight) :
					for w in nweight[j]:
						e += w * ndelta[j]
				d = e * (loutput[i] * (1.0 - loutput[i]))
				error.append(e)
				delta.append(d)
			deltam.append(delta)

		return deltam[::-1]



	def upgrade_wb(self, deltam, input, learn_rate, outputs):
		for i,l in enumerate(self.layers):
			l.upgrade_wb(deltam[i], input, learn_rate)
			input = outputs[i]


	def get_weight(self):
		res = []
		for l in self.layers:
			res.append(l.get_weight())
		return res

	def get_bias(self):
		res = []
		for l in self.layers:
			res.append(l.get_bias())
		return res

	def to_str(self):
		res = ""
		for i,l in enumerate(self.layers):
			res+= "layer "+i.__str__() +"\n"
			res+= l.to_str()
		return res





'''
make_layers([2,4,2,1])
layers = [
	NeuronLayer(2, 4),  # 2 input, # 4 neuron
	NeuronLayer(4, 2),  # 4 input, # 2 neuron
	NeuronLayer(2, 1),  # 2 input, # 1 neuron
]
'''
def make_layers(nneurons):
	layers = []
	for i in range(0,nneurons.__len__()-1):
		layers.append(NeuronLayer(nneurons[i], nneurons[i+1]))
	return layers


def plot_nn_2D(nn, n_input, x, title):
	fig, ax = plt.subplots()
	color = ['red', 'blue']

	for j in range(0, n_input):
		res, outputs = nn.forward_feeding(x[j])
		c = res[0] > 0.5
		xx = x[j][0]
		yy = x[j][1]
		scale = 20
		ax.scatter(xx, yy, c=color[c], s=scale, alpha=0.5, edgecolors='none')

	plt.title(title)
	plt.show()




def binary_classifiers(real,pred):
	tp = 0.0
	fn = 0.0
	fp = 0.0
	tn = 0.0
	for r,p in zip(real,pred):
		if(r and p):
			tp+=1
		elif(r and (not p)):
			fn+=1
		elif((not r) and p):
			fp+=1
		elif((not r) and (not p)):
			tn+=1
	return tp,fn,fp,tn


def get_performance(real, pred):
	tp, fn, fp, tn = binary_classifiers(real,pred)
	accuracy = (tp + tn) / (tp + tn + fp + fn)

	if(tp==0):
		precision = 0
		recall = 0
	else:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
	print("Performance::")
	print("\tAccuracy: ", accuracy)
	print("\tPrecision:", precision )
	print("\tRecall:   ", recall ,"\n" )
	return accuracy,precision,recall





