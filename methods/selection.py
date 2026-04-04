from abc import ABC, abstractmethod
import numpy as np
import random

class SelectionMethod(ABC):
    @abstractmethod
    def select(self, population, fitnesses, k):
        """
        Selecciona k individuos de la población.
        :param population: Lista de objetos Individual.
        :param fitnesses: Lista de valores de fitness correspondientes a la population.
        :param k: Cantidad de individuos a seleccionar.
        :return: Lista de individuos seleccionados.
        """
        pass

class EliteSelection(SelectionMethod):
    def select(self, population, fitnesses, k):
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1], reverse=True)
        selected = [ind for ind, fit in pop_fit[:k]]
        return selected


class RouletteSelection(SelectionMethod):
    def select(self, population, fitness, k):
        import numpy as np
        total_fitness = sum(fitness)
        
        if total_fitness == 0:
            probs = [1.0 / len(population)] * len(population)
        else:
            probs = [f / total_fitness for f in fitness]
            
        selected = np.random.choice(population, size=k, p=probs)
        return selected.tolist()
    
class BoltzmannSelection(SelectionMethod):
    def __init__(self, t0=100.0, tc=1.0):
        """
        :param t0: Temperatura inicial (alta).
        :param tc: Temperatura final (baja).
        """
        self.t0 = t0
        self.tc = tc

    def select(self, population, fitnesses, k, **kwargs):
        current_gen = kwargs.get('current_gen', 0)
        max_gen = kwargs.get('max_gen', 1)
        
        T = self.t0 - (current_gen / max_gen) * (self.t0 - self.tc)
        
        T = max(T, 0.0001) 
        
        fitnesses_array = np.array(fitnesses)
        
        max_fit = np.max(fitnesses_array)
        exp_vals = np.exp((fitnesses_array - max_fit) / T)
        
        probs = exp_vals / np.sum(exp_vals)
        
        selected = np.random.choice(population, size=k, p=probs)
        return selected.tolist()
    
class UniversalSelection(SelectionMethod):
    def select(self, population, fitnesses, k):
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            
            return np.random.choice(population, size=k).tolist()

        distance = total_fitness / k
        
        start_point = random.uniform(0, distance)
        
        pointers = [start_point + i * distance for i in range(k)]
        
        selected = []
        current_pointer_idx = 0
        cumulative_fitness = 0.0
        
        for i, ind in enumerate(population):
            cumulative_fitness += fitnesses[i]
            
            while current_pointer_idx < len(pointers) and pointers[current_pointer_idx] <= cumulative_fitness:
                selected.append(ind)
                current_pointer_idx += 1
                
        while len(selected) < k:
            selected.append(population[-1])
            
        return selected

class RankingSelection(SelectionMethod):
    def select(self, population, fitnesses, k):
        n = len(population)
        if n == 0:
            return []

        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1])
        
        sorted_population = [ind for ind, fit in pop_fit]

        ranks = np.arange(1, n + 1)  
        total_rank_sum = (n * (n + 1)) / 2 
        
        probs = ranks / total_rank_sum

        selected = np.random.choice(sorted_population, size=k, p=probs)
        
        return selected.tolist()
    
class DeterministicTournamentSelection(SelectionMethod):
    def __init__(self, tournament_size=3):
        """
        :param tournament_size: Cantidad de individuos que compiten en cada torneo (M).
        """
        self.tournament_size = tournament_size

    def select(self, population, fitnesses, k):
        selected = []
        pop_fit = list(zip(population, fitnesses))
        
        for _ in range(k):
            tournament = random.choices(pop_fit, k=self.tournament_size)
            
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
            
        return selected


class ProbabilisticTournamentSelection(SelectionMethod):
    def __init__(self, tournament_size=2, threshold=0.75):
        """
        :param tournament_size: Cantidad de individuos en el torneo.
        :param threshold: Probabilidad (p) de que gane el mejor individuo (suele estar entre 0.5 y 1.0).
        """
        self.tournament_size = tournament_size
        self.threshold = threshold

    def select(self, population, fitnesses, k):
        selected = []
        pop_fit = list(zip(population, fitnesses))
        
        for _ in range(k):
            tournament = random.choices(pop_fit, k=self.tournament_size)
            
            tournament.sort(key=lambda x: x[1], reverse=True)
            
            winner_ind = tournament[-1][0] 
            
            for ind, fit in tournament:
                r = random.random()
                if r < self.threshold:
                    winner_ind = ind
                    break 
                    
            selected.append(winner_ind)
            
        return selected