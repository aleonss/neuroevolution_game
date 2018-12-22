import numpy as np
import itertools
import copy

from neural_network import *




class individual:
	def __init__(self, nn):
		self.nn = nn
		self.genes_domain = [-2.0 , 2.0]

	def cross_over(self, ind):
		this_genes = nn2array(self.nn)
		ind_genes = nn2array(ind.nn)

		k = np.random.randint(0, this_genes.__len__())
		new_genes = this_genes[:k] + ind_genes[k:]

		new_nn = array2nn(self.nn, new_genes)
		#print("\n\nNeural ayy::\n", new_nn.to_str())

		return individual(new_nn)

	def mutate(self, mutation_rate):
		new_genes = []
		old_genes = nn2array(self.nn)

		for g in old_genes:
			if (np.random.rand() < mutation_rate):
				g.randomnize()
			new_genes.append(g)

		new_nn = array2nn(self.nn, new_genes)
		return individual(new_nn)


	def get_ans(self, input):
		return self.nn.forward_feeding(input)[0][0]




class population:
	def __init__(self, n, nn_layout, fitness_function, ):
		self.n = n
		self.fitness_function = fitness_function
		self.individuals = []

		for i in range(0, n):
			layers = make_layers(nn_layout)
			ind = individual(NeuralNetwork(layers))
			self.individuals.append(ind)

	def evaluate_fitness(self, input):
		pfit = []
		for i in self.individuals:
			ians = i.get_ans(input)
			ifit = self.fitness_function(ians, input)
			pfit.append(ifit)
		return pfit

	def get_fittest(self, input):
		pop_fitness = self.evaluate_fitness(input)
		fittest_index = (np.argsort(pop_fitness)[::-1][0])
		return self.individuals[fittest_index]

	def selection(self, input, selection_ratio):
		mating_pool = []
		pop_fitness = self.evaluate_fitness(input)
		n_parent = int(1 - self.n * selection_ratio)

		fittest_index = np.argsort(pop_fitness)[n_parent:][::-1]

		for index in fittest_index:
			mating_pool.append(self.individuals[index])
			# print("\t",self.individuals[index].get_ans())
		return mating_pool

	def cross_over(self, mating_pool):
		children = []

		for comb in list(itertools.combinations(mating_pool, 2)):
			children.append(comb[0].cross_over(comb[1]))
			# print(comb[0].cross_over(comb[1]).get_ans())

		while (children.__len__() < self.n):
			rand_parent0 = mating_pool[np.random.randint(0, mating_pool.__len__())]
			rand_parent1 = mating_pool[np.random.randint(0, mating_pool.__len__())]
			children.append(rand_parent0.cross_over(rand_parent1))

		return children[:self.n]

	def mutation(self, children, mutation_rate):
		offspring = []
		for c in children:
			offspring.append(c.mutate(mutation_rate))
		return offspring
