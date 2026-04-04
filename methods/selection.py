from abc import ABC, abstractmethod

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
        # Calculamos la suma total para la ruleta
        total_fitness = sum(fitness)
        
        if total_fitness == 0:
            probs = [1.0 / len(population)] * len(population)
        else:
            probs = [f / total_fitness for f in fitness]
            
        # Elegimos k individuos basados en su probabilidad
        selected = np.random.choice(population, size=k, p=probs)
        return selected.tolist()
