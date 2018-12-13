import numpy as np
import itertools
import string


class individual:
    def __init__(self, genes, genes_domain, function):
        self.genes = genes
        self.genes_domain = genes_domain
        self.function = function

    def cross_over(self, ind):
        k = np.random.randint(0, self.genes.__len__())
        new_genes = self.genes[:k] + ind.genes[k:]
        return individual(new_genes, self.genes_domain,self.function)

    def mutate_discrete(self, mutation_rate):
        new_genes = []
        domain_len = self.genes_domain.__len__()
        for g in self.genes:
            if (np.random.rand() < mutation_rate):
                new_genes.append(self.genes_domain[np.random.randint(0, domain_len)])
            else:
                new_genes.append(g)

        return individual(new_genes, self.genes_domain,self.function)

    def mutate_continuous(self, mutation_rate):
        new_genes = []
        domain_len = self.genes_domain[1]- self.genes_domain[0]
        for g in self.genes:
            if (np.random.rand() < mutation_rate):
                gen_value = self.genes_domain[0] + np.random.random_sample() * domain_len
                new_genes.append(gen_value)
                #new_genes.append(self.genes_domain[np.random.randint(0, domain_len)])
            else:
                new_genes.append(g)

        return individual(new_genes, self.genes_domain,self.function)



    def get_ans(self):
        #print(self.function)
        return self.function(self.genes)



class population:
    def __init__(self, n, n_genes, genes_domain, domain_type, function, fitness_function, ):
        self.n = n
        self.n_genes = n_genes
        self.genes_domain = genes_domain
        self.fitness_function = fitness_function
        self.function = function
        self.individuals = []

        if(domain_type=="discrete"):
            for i in range(0, n):
                self.individuals.append(self.random_discrete_individual())
        elif(domain_type=="continuous"):
            for i in range(0, n):
                self.individuals.append(self.random_continuous_individual())

    def random_discrete_individual(self):
        genes = []
        domain_len = self.genes_domain.__len__()
        for i in range(0, self.n_genes):
            genes.append(self.genes_domain[np.random.randint(0, domain_len)])

        return individual(genes, self.genes_domain, self.function)


    def random_continuous_individual(self):
        genes = []
        domain_len = self.genes_domain[1]- self.genes_domain[0]
        for i in range(0, self.n_genes):
            gen_value = self.genes_domain[0]+ np.random.random_sample()*domain_len
            genes.append(gen_value)

        #print(genes,domain_len)
        return individual(genes, self.genes_domain, self.function)



    def evaluate_fitness(self, solution):
        pfit = []
        for i in self.individuals:
            ians = i.get_ans()
            ifit = self.fitness_function(ians, solution)
            pfit.append(ifit)
        return pfit

    def get_fittest(self, solution):
        pop_fitness = self.evaluate_fitness(solution)
        fittest_index = (np.argsort(pop_fitness)[::-1][0])
        return self.individuals[fittest_index]

    def selection(self, solution, selection_ratio):
        mating_pool = []
        pop_fitness = self.evaluate_fitness(solution)
        n_parent = int(1 - self.n * selection_ratio)

        fittest_index = np.argsort(pop_fitness)[n_parent:][::-1]

        for index in fittest_index:
            mating_pool.append(self.individuals[index])
            #print("\t",self.individuals[index].get_ans())
        return mating_pool

    def cross_over(self, mating_pool):
        children = []

        for comb in list(itertools.combinations(mating_pool, 2)):
            children.append(comb[0].cross_over(comb[1]))
            #print(comb[0].cross_over(comb[1]).get_ans())

        while (children.__len__() < self.n):
            rand_parent0 = mating_pool[np.random.randint(0, mating_pool.__len__())]
            rand_parent1 = mating_pool[np.random.randint(0, mating_pool.__len__())]
            children.append(rand_parent0.cross_over(rand_parent1))

        return children[:self.n]

    def mutation(self, children, mutation_rate,  domain_type):
        offspring = []
        if(domain_type=="discrete"):
            for c in children:
                offspring.append(c.mutate_discrete(mutation_rate))
        elif(domain_type=="continuous"):
            for c in children:
                offspring.append(c.mutate_continuous(mutation_rate))
        return offspring



