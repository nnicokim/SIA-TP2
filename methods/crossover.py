import numpy as np
from abc import ABC, abstractmethod

class CrossoverMethod(ABC):
    @abstractmethod
    def crossover(self, p1_genes, p2_genes):
        """
        Cruza dos padres y retorna los genes hijas.
        :param p1_genes: Genes del padre 1 (numpy array)
        :param p2_genes: Genes del padre 2 (numpy array)
        :return: Tupla con genes de los hijos (child1_genes, child2_genes)
        """
        pass

class OnePointCrossover(CrossoverMethod):
    def crossover(self, p1_genes, p2_genes):
        cut = len(p1_genes) // 2
        
        c1_genes = np.concatenate([p1_genes[:cut], p2_genes[cut:]])
        c2_genes = np.concatenate([p2_genes[:cut], p1_genes[cut:]])
        
        return c1_genes, c2_genes
