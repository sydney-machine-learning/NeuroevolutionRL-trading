import numpy as np
import time
import random

class G3PCX(object):
    def __init__(self, population_size, num_variables, max_limits, min_limits, max_evals=500000):
        self.initialize_parameters()
        self.sp_size = self.children + self.family
        self.population = np.random.uniform(min_limits[0], max_limits[0], size=(population_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(population_size)]
        self.sub_pop = np.random.uniform(min_limits[0], max_limits[0], size=(self.sp_size, num_variables))  #[SpeciesPopulation(num_variables) for count in xrange(NPSize)]
        self.fitness = np.zeros(population_size)
        self.sp_fit  = np.zeros(self.sp_size)
        self.best_index = 0
        self.best_fit = 0
        self.worst_index = 0
        self.worst_fit = 0
        self.rand_parents =  self.num_parents
        self.temp_index =  np.arange(0, population_size)
        self.rank =  np.arange(0, population_size)
        self.list = np.arange(0, self.sp_size)
        self.parents = np.arange(0, population_size)
        self.population_size = population_size
        self.num_variables = num_variables
        self.num_evals = 0
        self.max_evals = max_evals
        self.problem = 1

    def fitness_function(self, x):    #  function  (can be any other function, model or even a neural network)
        fit = 0.0
        if self.problem == 1: # rosenbrock
            for j in range(x.size -1):
                fit += (100.0*(x[j]*x[j] - x[j+1])*(x[j]*x[j] - x[j+1]) + (x[j]-1.0)*(x[j]-1.0))
        elif self.problem ==2:  # ellipsoidal - sphere function
            for j in range(x.size):
                fit = fit + ((j+1)*(x[j]*x[j]))
        return fit # note we will maximize fitness, hence minimize error

    def initialize_parameters(self):
        self.epsilon = 1e-40  # convergence
        self.sigma_eta = 0.1
        self.sigma_zeta = 0.1
        self.children = 2
        self.num_parents = 3
        self.family = 2

    def rand_normal(self, mean, stddev):
        if (not G3PCX.n2_cached):
            #choose a point x,y in the unit circle uniformly at random
            x = np.random.uniform(-1,1,1)
            y = np.random.uniform(-1,1,1)
            r = x*x + y*y
            while (r == 0 or r > 1):
                x = np.random.uniform(-1,1,1)
                y = np.random.uniform(-1,1,1)
                r = x*x + y*y
            # Apply Box-Muller transform on x, y
            d = np.sqrt(-2.0*np.log(r)/r)
            n1 = x*d
            G3PCX.n2 = y*d
            # scale and translate to get desired mean and standard deviation
            result = n1*stddev + mean
            G3PCX.n2_cached = True
            return result
        else:
            G3PCX.n2_cached = False
            return G3PCX.n2*stddev + mean

    def evaluate(self):
        self.fitness[0] = self.fitness_function(self.population[0,:])
        self.best_fit = self.fitness[0]
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i,:])
            if (self.best_fit> self.fitness[i]):
                self.best_fit =  self.fitness[i]
                self.best_index = i
        self.num_evals += 1

    # calculates the magnitude of a vector
    def mod(self, List):
        sum = 0
        for i in range(self.num_variables):
            sum += (List[i] * List[i] )
        return np.sqrt(sum)

    def parent_centric_xover(self, current):
        centroid = np.zeros(self.num_variables)
        tempar1 = np.zeros(self.num_variables)
        tempar2 = np.zeros(self.num_variables)
        temp_rand = np.zeros(self.num_variables)
        d = np.zeros(self.num_variables)
        D = np.zeros(self.num_parents)
        temp1, temp2, temp3 = (0,0,0)
        diff = np.zeros((self.num_parents, self.num_variables))
        for i in range(self.num_variables):
            for u in range(self.num_parents):
                centroid[i]  = centroid[i] +  self.population[self.temp_index[u],i]
        centroid   = centroid / self.num_parents
        # calculate the distace (d) from centroid to the index parent self.temp_index[0]
        # also distance (diff) between index and other parents are computed
        for j in range(1, self.num_parents):
            for i in range(self.num_variables):
                if j == 1:
                    d[i]= centroid[i]  - self.population[self.temp_index[0],i]
                diff[j, i] = self.population[self.temp_index[j], i] - self.population[self.temp_index[0],i]
            if (self.mod(diff[j,:]) < self.epsilon):
                print('Points are very close to each other. Quitting this run')
                return 0
        dist = self.mod(d)
        if (dist < self.epsilon):
            print ("\nError -  points are very close to each other. Quitting this run\n")
            return 0
        # orthogonal directions are computed
        for j in range(1, self.num_parents):
            temp1 = self.inner(diff[j,:] , d )
            if ((self.mod(diff[j,:]) * dist) == 0):
                print("Division by zero")
                temp2 = temp1 / (1)
            else:
                temp2 = temp1 / (self.mod(diff[j,:]) * dist)
            temp3 = 1.0 - np.power(temp2, 2)
            D[j] = self.mod(diff[j]) * np.sqrt(np.abs(temp3))
        D_not = 0.0
        for i in range(1, self.num_parents):
            D_not += D[i]
        D_not /= (self.num_parents - 1) # this is the average of the perpendicular distances from all other parents (minus the index parent) to the index vector
        G3PCX.n2 = 0.0
        G3PCX.n2_cached = False
        for i in range(self.num_variables):
            tempar1[i] = self.rand_normal(0,  self.sigma_eta * D_not) #rand_normal(0, D_not * sigma_eta);
            tempar2[i] = tempar1[i]
        if(np.power(dist, 2) == 0):
            print(" division by zero: part 2")
            tempar2  = tempar1
        else:
            tempar2  = tempar1  - (np.multiply(self.inner(tempar1, d) , d )  ) / np.power(dist, 2.0)
        tempar1 = tempar2
        self.sub_pop[current,:] = self.population[self.temp_index[0],:] + tempar1
        rand_var = self.rand_normal(0, self.sigma_zeta)
        for j in range(self.num_variables):
            temp_rand[j] =  rand_var
        self.sub_pop[current,:] += np.multiply(temp_rand ,  d )
        self.sp_fit[current] = self.fitness_function(self.sub_pop[current,:])
        self.num_evals += 1
        return 1

    def inner(self, ind1, ind2):
        sum = 0.0
        for i in range(self.num_variables):
            sum += (ind1[i] * ind2[i])
        return  sum

    def sort_population(self):
        dbest = 99
        for i in range(self.children + self.family):
            self.list[i] = i
        for i in range(self.children + self.family - 1):
            dbest = self.sp_fit[self.list[i]]
            for j in range(i + 1, self.children + self.family):
                if(self.sp_fit[self.list[j]]  < dbest):
                    dbest = self.sp_fit[self.list[j]]
                    temp = self.list[j]
                    self.list[j] = self.list[i]
                    self.list[i] = temp

    def replace_parents(self): #here the best (1 or 2) individuals replace the family of parents
        for j in range(self.family):
            self.population[ self.parents[j],:]  =  self.sub_pop[ self.list[j],:] # Update population with new species
            fx = self.fitness_function(self.population[ self.parents[j],:])
            self.fitness[self.parents[j]]   =  fx
            self.num_evals += 1

    def family_members(self): #//here a random family (1 or 2) of parents is created who would be replaced by good individuals
        swp = 0
        for i in range(self.population_size):
            self.parents[i] = i
        for i in range(self.family):
            randomIndex = random.randint(0, self.population_size - 1) + i # Get random index in population
            if randomIndex > (self.population_size-1):
                randomIndex = self.population_size-1
            swp = self.parents[randomIndex]
            self.parents[randomIndex] = self.parents[i]
            self.parents[i] = swp

    def find_parents(self): #here the parents to be replaced are added to the temporary subpopulation to assess their goodness against the new solutions formed which will be the basis of whether they should be kept or not
        self.family_members()
        for j in range(self.family):
            self.sub_pop[self.children + j, :] = self.population[self.parents[j],:]
            fx = self.fitness_function(self.sub_pop[self.children + j, :])
            self.sp_fit[self.children + j]  = fx
            self.num_evals += 1

    def random_parents(self ):
        for i in range(self.population_size):
            self.temp_index[i] = i
        swp=self.temp_index[0]
        self.temp_index[0]=self.temp_index[self.best_index]
        self.temp_index[self.best_index]  = swp
        # best is always included as a parent and is the index parent
        # this can be changed for solving a generic problem
        for i in range(1, self.rand_parents):
            index= np.random.randint(self.population_size)+i
            if index > (self.population_size-1):
                index = self.population_size-1
            swp=self.temp_index[index]
            self.temp_index[index]=self.temp_index[i]
            self.temp_index[i]=swp

    def evolve(self, outfile):
        tempfit = 0
        prevfitness = 99
        self.evaluate()
        tempfit= self.fitness[self.best_index]
        while(self.num_evals < self.max_evals):
            tempfit = self.best_fit
            self.random_parents()
            for i in range(self.children):
                tag = self.parent_centric_xover(i)
                if (tag == 0):
                    break
            if tag == 0:
                break
            self.find_parents()
            self.sort_population()
            self.replace_parents()
            self.best_index = 0
            tempfit = self.fitness[0]
            for x in range(1, self.population_size):
                if(self.fitness[x] < tempfit):
                    self.best_index = x
                    tempfit  =  self.fitness[x]
            if self.num_evals % 197 == 0:
                print(self.fitness[self.best_index])
                print(self.num_evals, 'num of evals\n\n\n')
            np.savetxt(outfile, [ self.num_evals, self.best_index, self.best_fit], fmt='%1.5f', newline="\n")
        print(self.sub_pop, '  sub_pop')
        print(self.population[self.best_index], ' best sol')
        print(self.fitness[self.best_index], ' fitness')

if __name__ == '__main__':
    
    # OUTPUT FILE FOR POPULATION
    outfile=open('pop_.txt','w')
    
    # STOPPING FITNESS VALUE
    MinCriteria = 0.005
    
    # SET SEED VALUE
    random.seed(time.time())
    
    # EVOLUTION PARAMETERS
    max_evals = 700000
    pop_size =  100
    num_varibles = 5
    max_limits = np.repeat(5, num_varibles)
    min_limits = np.repeat(-5, num_varibles)

    # INITIALIZE CLASS OBJECT
    g3pcx  = G3PCX(pop_size, num_varibles, max_limits, min_limits, max_evals)
    
    # EVOLVE
    g3pcx.evolve(outfile)