import numpy as np
from abc import ABC, abstractmethod

class MutationMethod(ABC):
    def __init__(self, mutation_rate, w, h):
        self.mutation_rate = mutation_rate
        self.w = w
        self.h = h

    @abstractmethod
    def mutate(self, genes, **kwargs):
        """
        Aplica mutación sobre los genes de un individuo.
        :param genes: Genes a mutar (numpy array mutado in-place o copia)
        :return: Genes mutados (numpy array)
        """
        pass

class GeneMutation(MutationMethod):
    def mutate(self, genes, **kwargs):
        new_genes = genes.copy()  
        
        for i in range(len(new_genes)):
            if np.random.rand() < self.mutation_rate:
                col_idx = np.random.randint(0, 10)
                if col_idx < 6: # Es coordenada
                    new_genes[i, col_idx] = np.random.randint(0, self.w if col_idx % 2 == 0 else self.h)
                else: # Es color/alpha
                    new_genes[i, col_idx] = np.random.randint(0, 256)
                    
        return new_genes

class UniformMutation(MutationMethod):
    def mutate(self, genes, **kwargs):
        new_genes = genes.copy()
        

        for i in range(len(new_genes)):
            for j in range(10): # 6 coordenadas + 4 colores
                if np.random.rand() < self.mutation_rate:
                    # Si toca mutar, se asigna un valor nuevo bajo Distribución Uniforme
                    if j < 6: # Es una coordenada
                        if j % 2 == 0:
                            new_genes[i, j] = np.random.randint(0, self.w)
                        else:
                            new_genes[i, j] = np.random.randint(0, self.h)
                    else: # Es un canal de color o opacidad (RGBA)
                        new_genes[i, j] = np.random.randint(0, 256)
                        
        return new_genes

class NoUniformMutation(MutationMethod):
    def mutate(self, genes, **kwargs):

        current_gen = kwargs.get('current_gen', 0)
        max_gen = kwargs.get('max_gen', 1)

        new_genes = genes.copy()

        for i in range(len(new_genes)):
            for j in range(10):
                if np.random.rand() < self.mutation_rate:
                    force = 1 - (current_gen / max_gen)
                    if j < 6:
                        if j % 2 == 0:
                            new_genes[i, j] = np.clip(new_genes[i, j] + np.random.normal(0, force * self.w), 0, self.w)
                        else:
                            new_genes[i, j] = np.clip(new_genes[i, j] + np.random.normal(0, force * self.h), 0, self.h)
                    else:
                        new_genes[i, j] = np.clip(new_genes[i, j] + np.random.normal(0, force * 255), 0, 255)
        
        return new_genes
        
        

