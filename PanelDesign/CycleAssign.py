import numpy as np
import scanpy as sc
import random
import os
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

class GeneticCycleChoose():
    def __init__(self, pop_size, num_generations, panel, threads = 128):
        
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.threads = threads
        
        adata = sc.read_h5ad("/media/duan/DuanLab_Data/openFISH/Panel_Design/Brain20240911/ABA_scGIST_MAGIC.h5ad")
        self.MAGIC_df = adata.to_df(layer = "MAGIC")
        self.panel = panel
        
    # Fitness function
    def Comb_spearmanr(self, genes_perm:np.ndarray):
        
        CYCLE_ORDER = {
            "R1": genes_perm[0:10],
            "R2": genes_perm[10:20],
            "R3": genes_perm[20:30],
            "R4": genes_perm[30:40],
            "R5": genes_perm[40:50],
            "R6": genes_perm[50:60],
            "R7": genes_perm[60:70],
            "R8": genes_perm[70:80],
            "R9": genes_perm[80:90],
            "R10": genes_perm[90:100],
            "R11": genes_perm[100:109],
        }
        
        comb_R = 0
        
        for tmp_comb in CYCLE_ORDER.values():
            
            spearmanr_matrix = spearmanr(self.MAGIC_df.loc[:, tmp_comb]).statistic
            spearmanr_mean = spearmanr_matrix[np.tril_indices(spearmanr_matrix.shape[0], k=-1)].mean()
            
            comb_R += spearmanr_mean

        return comb_R

    # Individual Selection
    def selection(self, population:list[np.ndarray]):
        scaler = MinMaxScaler(feature_range=(1, 10)) # scale the data to enlarge the differences
        population_spearmanr = Parallel(n_jobs=self.threads, backend='loky')(delayed(self.Comb_spearmanr)(x) for x in population)
        print(f"Present minimum spearmanr aggregation: {min(population_spearmanr)}")
        scaled_spearmanr = scaler.fit_transform(np.array(population_spearmanr).reshape(-1,1))
        fitness = [1 / x for x in scaled_spearmanr.flatten()]
        idx = np.random.choice(len(population), size=len(population), p=fitness/np.sum(fitness))
        return [population[i] for i in idx]

    # CX
    def crossover(self, parent1:list[str], parent2:list[str]):

        child = [-1] * len(parent1)

        idx = np.random.choice(len(parent1), 1)[0]

        while child[idx] == -1:
            child[idx] = parent1[idx]
            idx = parent1.index(parent2[idx])

        for i in range(len(child)):
            if child[i] == -1:
                child[i] = parent2[i]

        return child

    # mutation
    def mutation(self, individual):
        idx1, idx2 = np.random.choice(len(individual), size=2, replace=False)
        tmp1 = individual[idx1]
        tmp2 = individual[idx2]
        individual[idx1] = tmp2
        individual[idx2] = tmp1
        return individual

    # Genetic algorithm main function
    def genetic_algorithm(self):

        genes = self.panel
        population = [list(np.random.permutation(genes)) for _ in range(self.pop_size)]

        prod_list = []
        for ig in range(self.num_generations):
            
            print(f"Generation {ig}:")
            new_population = []

            while len(new_population) < self.pop_size:
                parent1_idx, parent2_idx = np.random.choice(len(population), size=2, replace=False)
                parent1 = list(population[parent1_idx])
                parent2 = list(population[parent2_idx])
                child = self.crossover(parent1, parent2)
                if np.random.rand() < 0.5:  # 50% probability mutate
                    child = self.mutation(child)
                new_population.append(np.array(child))

            population = self.selection(new_population)

        population_spearmanr = Parallel(n_jobs=self.threads, backend='loky')(delayed(self.Comb_spearmanr)(x) for x in population)
        best_index = population_spearmanr.index(min(population_spearmanr))
        best_perm = population[best_index]
        
        return best_perm

    
if __name__ == "__main__":
    
    panel = np.load("Panel.npy")
    
    num_generations = 500
    pop_size = 300
    ga = GeneticCycleChoose(pop_size = pop_size, num_generations = num_generations, panel = panel, threads = 128)
    
    import time

    start_time = time.time()
    
    best_perm = ga.genetic_algorithm()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Run time is：{elapsed_time/3600}hours.")
    
    np.save(f"./best_perm_110.npy", best_perm)