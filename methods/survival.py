from abc import ABC, abstractmethod

class SurvivalMethod(ABC):
    @abstractmethod
    def select_next_generation(self, old_population, offspring, pop_size):
        """
        Selecciona la siguiente generación dado la población actual y la descendencia.
        :param old_population: Lista de Individuos actuales.
        :param offspring: Lista de Individuos generados por cruza y mutación.
        :param pop_size: Tamaño de la población objetivo.
        :return: Nueva lista de Individuos.
        """
        pass

class AdditiveSurvival(SurvivalMethod):
    def __init__(self, selection_method):
        self.selection_method = selection_method
        
    def select_next_generation(self, old_population, offspring, pop_size):
        # Supervivencia aditiva: competimos todos contra todos
        combined = old_population + offspring
        fitnesses = [ind.fitness for ind in combined]
        # Usamos el método de selección pasado para elegir a los sobrevivientes
        survivors = self.selection_method.select(combined, fitnesses, pop_size)
        return survivors

class ExclusiveSurvival(SurvivalMethod):
    def select_next_generation(self, old_population, offspring, pop_size):
        # Supervivencia exclusiva: solo la descendencia sobrevive
        # Puede implementarse con selección elitista simple para los mejores de los hijos,
        # o tal cual si len(offspring) == pop_size
        return offspring[:pop_size]
