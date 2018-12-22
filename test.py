


from neural_network import *
from genetic_algorithm import *



## NEURAL NETWORK

def logic_xor(x,y):
	return np.logical_xor(x,y)



def make_int_inputs(ninputs, in_range):
	inputs = []
	for i in range(0,ninputs):
		xx= random.randint(in_range[0],in_range[1])
		xy= random.randint(in_range[0],in_range[1])
		inputs.append([xx,xy])
	return inputs



def real_values(logic_funct,ninputs, in_range=[0,1]):
	inputs = make_int_inputs(ninputs, in_range)
	results = []
	for x in inputs:
		results.append(logic_funct(x[0],x[1]))
	return inputs,results





def nn_learn(nn, logic_funct, trainings = 1000, learn_rate = 0.5, in_range=[0,1], vervose=False ):
	if (vervose):
		print("Learning::")
		error = 0.0
		iters = 0
		size = trainings / 10

	inputs, results = real_values(logic_funct,trainings, in_range)

	for x,real in zip(inputs,results):
		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, [real])
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		if(vervose):
			error += abs(real -res[0])
			if( (iters % size == 0) and (iters!=0) ):
				ratio= (error/iters)
				error=0.0
				print(iters,"\t error:",ratio)
			iters+=1

	return nn

def test_funct(sn,logic_funct, ntests=100, in_range=[0,1]):
	xreal=[]
	xpred=[]

	inputs, results = real_values(logic_funct,ntests, in_range)

	for x,real in zip(inputs,results):
		res = sn.feed(x)
		if (type(res) == list):
			result = sn.feed(x)[0] > 0.5
		else:
			result = sn.feed(x) > 0.5
		xreal.append(real)
		xpred.append(result)

	return xreal, xpred




def test_xor():
	print("\nTEST: XOR")
	layers = make_layers([2,4,4,1])
	nn = NeuralNetwork(layers)
	nn = nn_learn(nn, logic_xor, learn_rate=0.2, trainings=30000, in_range=[0,1],vervose=True)
	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_xor)
	get_performance(xreal,xpred)
	inputs = make_int_inputs(100,[0,1])
	#plot_nn_2D(nn, 100, inputs, "xor")






## GA

# functions
def identity_function(x):
    return x

def ftan_avg(x):
    xx = np.average(x)
    return np.tan(xx)


# fitness functions

def sequence_fitness_function(answer, solution):
    fit = 0
    for a, s in zip(answer, solution):
        fit += (a == s)
    return fit

def max_fitness_function(answer, solution):
    return answer
def min_fitness_function(answer, solution):
    return -answer



# evolve

def evolve_discrete(solution, genes_domain, fitness_function, function, pop_size, selection_ratio, mutation_rate):
    print("\n>>\t",solution,"<< solution")
    n_genes = solution.__len__()

    pop = population(pop_size, n_genes, genes_domain, "discrete", function, fitness_function)
    fittest = pop.get_fittest(solution)

    iterations = 0
    max_iterations = 100


    while (fittest.get_ans() != solution and (iterations<max_iterations)):

        mating_pool = pop.selection(solution, selection_ratio)
        children = pop.cross_over(mating_pool)
        pop.individuals = pop.mutation(children, mutation_rate, "discrete", )
        fittest = pop.get_fittest(solution)

        iterations += 1
        print(iterations,"\t", fittest.get_ans(), sequence_fitness_function(fittest.get_ans(),solution))

    print("fittest result:",fittest.get_ans())
    print("fittest genes: ",fittest.genes)
    return fittest



def evolve_continuous(n_genes, solution, genes_domain, fitness_function, function, pop_size, selection_ratio, mutation_rate):
    print("\n>>\t",solution,"<< solution")

    pop = population(pop_size, n_genes, genes_domain, "continuous", function, fitness_function)
    fittest = pop.get_fittest(solution)

    iterations = 0
    max_iterations = 50

    while ((fittest.get_ans() != solution) and (iterations<max_iterations) ):

        mating_pool = pop.selection(solution, selection_ratio)
        children = pop.cross_over(mating_pool)
        pop.individuals = pop.mutation(children, mutation_rate, "continuous")
        fittest = pop.get_fittest(solution)

        iterations += 1
        print(iterations,"\t", fittest.get_ans())

    print("fittest result:",fittest.get_ans())
    print("fittest genes: ",fittest.genes)
    return fittest







def sequence_examples():

    pop_size = 30
    mutation_rate = 0.2
    selection_ratio = 0.5

    solution = list("bart")
    genes_domain = list(string.ascii_lowercase)
    evolve_discrete(solution, genes_domain, sequence_fitness_function, identity_function, pop_size, selection_ratio, mutation_rate)

    solution = list("1234")
    genes_domain = list(string.digits)
    evolve_discrete(solution, genes_domain, sequence_fitness_function, identity_function, pop_size, selection_ratio, mutation_rate)



def maximization_examples():

    pop_size = 50
    mutation_rate = 0.2
    genes_domain = [-5.0,5.0] # range

    selection_ratio = 0.5
    n_genes = 2
    evolve_continuous(n_genes, 9999.0, genes_domain, max_fitness_function, ftan_avg, pop_size, selection_ratio, mutation_rate)




'''

test_xor()

sequence_examples()
maximization_examples()

'''




def test():
	print("\nTEST: XOR")
	layers = make_layers([2,4,4,1])
	nn = NeuralNetwork(layers)
	nn = nn_learn(nn, logic_xor, learn_rate=0.2, trainings=30000, in_range=[0,1],vervose=True)


	print("Neural Network::\n",nn.to_str())

	xreal, xpred = test_funct(nn, logic_xor)
	get_performance(xreal,xpred)
	#inputs = make_int_inputs(100,[0,1])
	#plot_nn_2D(nn, 100, inputs, "xor")


test()
'''
layers = make_layers([2,4,4,1])
nn = NeuralNetwork(layers)

print("Learning::")
error = 0.0
iters = 0
size = trainings / 10
inputs, results = real_values(logic_funct,trainings, in_range)
for x,real in zip(inputs,results):
	res, outputs = nn.forward_feeding(x)
	deltam = nn.error_backpropagation(outputs, [real])
	nn.upgrade_wb(deltam, x, learn_rate, outputs)
		error += abs(real -res[0])
	if( (iters % size == 0) and (iters!=0) ):
		ratio= (error/iters)
		error=0.0
		print(iters,"\t error:",ratio)
	iters+=1


pop_size=8


pop = population(pop_size, n_genes, genes_domain, "discrete", function, fitness_function)
fittest = pop.get_fittest(solution)
iterations = 0
max_iterations = 100

while (iterations < max_iterations):
	mating_pool = pop.selection(solution, selection_ratio)
	children = pop.cross_over(mating_pool)
	pop.individuals = pop.mutation(children, mutation_rate, "discrete", )
	fittest = pop.get_fittest(solution)
		iterations += 1
	print(iterations, "\t", fittest.get_ans(), sequence_fitness_function(fittest.get_ans(), solution))
	print("fittest result:", fittest.get_ans())
print("fittest genes: ", fittest.genes)


print("Neural Network::\n", nn.to_str())
xreal, xpred = test_funct(nn, logic_xor)
get_performance(xreal, xpred)


'''