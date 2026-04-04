from methods.selection import EliteSelection, RouletteSelection, BoltzmannSelection,UniversalSelection, RankingSelection, DeterministicTournamentSelection, ProbabilisticTournamentSelection
from methods.crossover import OnePointCrossover
from methods.mutation import GeneMutation, UniformMutation, NoUniformMutation
from methods.survival import AdditiveSurvival, ExclusiveSurvival

class MethodFactory:
    @staticmethod
    def create_selection(method_name, **kwargs):
        method_name = method_name.lower()
        if method_name == "elite":
            return EliteSelection()
        elif method_name == "roulette" or method_name == "ruleta":
            return RouletteSelection()
        elif method_name == "boltzmann":
            t0 = kwargs.get('t0', 100.0)
            tc = kwargs.get('tc', 1.0)
            return BoltzmannSelection(t0, tc)
        elif method_name == "universal":
            return UniversalSelection()
        elif method_name == "ranking":
            return RankingSelection()
        elif method_name == "deterministic_tournament":
            tourney_size = kwargs.get('tournament_size', 3)
            return DeterministicTournamentSelection(tourney_size)
        elif method_name == "probabilistic_tournament":
            tourney_size = kwargs.get('tournament_size', 2)
            threshold = kwargs.get('threshold', 0.75)
            return ProbabilisticTournamentSelection(tourney_size, threshold)
            
        raise ValueError(f"Selection method '{method_name}' not supported.")

    @staticmethod
    def create_crossover(method_name, **kwargs):
        method_name = method_name.lower()
        if method_name == "one_point":
            return OnePointCrossover()
        # Agregar cruce dos puntos, uniforme, anular, etc.
        raise ValueError(f"Crossover method '{method_name}' not supported.")

    @staticmethod
    def create_mutation(method_name, mutation_rate, w, h, **kwargs):
        method_name = method_name.lower()
        if method_name == "gene":
            return GeneMutation(mutation_rate, w, h)
        elif method_name == "uniform":
            return UniformMutation(mutation_rate, w, h)
        # Agregar MultiGen, No Uniforme, etc.
        elif method_name == "nouniform":
            return NoUniformMutation(mutation_rate, w, h)
        raise ValueError(f"Mutation method '{method_name}' not supported.")

    @staticmethod
    def create_survival(method_name, selection_method=None, **kwargs):
        method_name = method_name.lower()
        if method_name == "additive":
            if selection_method is None:
                raise ValueError("Additive survival requires a selection method.")
            return AdditiveSurvival(selection_method)
        elif method_name == "exclusive":
            return ExclusiveSurvival()
        # Puedes agregar implementaciones adicionales aquí.
        raise ValueError(f"Survival method '{method_name}' not supported.")
