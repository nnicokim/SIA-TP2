import numpy as np
from Individual import Individual

class GeneticEngine:
    def __init__(self, target_img, config, selection_strategy, crossover_strategy, mutation_strategy, survival_strategy):
        self.config = config
        self.target = np.array(target_img.convert('RGB'))
        self.w, self.h = target_img.size
        
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.survival_strategy = survival_strategy
        
        self.pop_size = config['algorithm']['pop_size']
        self.n_triangles = config['algorithm']['n_triangles']
        
        self.pop = [Individual(self.n_triangles, self.w, self.h) for _ in range(self.pop_size)]

    def calculate_fitness(self, ind):
        res = np.array(ind.render(), dtype=np.float32)
        target = self.target.astype(np.float32)
        return 1 / (np.mean((target - res) ** 2) + 1)

    def evolve_step(self, current_gen, max_generations):
        for ind in self.pop:
            if ind.fitness == 0:
                ind.fitness = self.calculate_fitness(ind)
        
        fitnesses = [ind.fitness for ind in self.pop]
        parents = self.selection_strategy.select(self.pop, fitnesses, self.pop_size)
        
        offspring = []
        
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % self.pop_size]
            
            c1_genes, c2_genes = self.crossover_strategy.crossover(p1.genes, p2.genes)
            
            c1_genes = self.mutation_strategy.mutate(c1_genes, current_gen=current_gen, max_gen=max_generations)
            c2_genes = self.mutation_strategy.mutate(c2_genes, current_gen=current_gen, max_gen=max_generations)
            
            offspring.append(Individual(self.n_triangles, self.w, self.h, c1_genes))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(self.n_triangles, self.w, self.h, c2_genes))
                
        for ind in offspring:
            ind.fitness = self.calculate_fitness(ind)
            
        self.pop = self.survival_strategy.select_next_generation(self.pop, offspring, self.pop_size)