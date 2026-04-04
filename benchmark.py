import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image

from Engine import GeneticEngine
from methods.factory import MethodFactory

# 1. Definimos los métodos por defecto (los que quedan fijos)
DEFAULTS = {
    'selection': 'ruleta',
    'crossover': 'one_point',
    'mutation': 'gene',
    'survival': 'exclusive'
}

# 2. Definimos todas las opciones a iterar por cada categoría
OPTIONS = {
    'selection': ['elite', 'ruleta', 'universal', 'boltzmann', 'ranking', 'deterministic_tournament', 'probabilistic_tournament'],
    'crossover': ['one_point', 'two_point', 'uniform', 'annular'],
    'mutation': ['gene', 'uniform', 'nouniform'],
    'survival': ['additive', 'exclusive']
}

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_ga_and_get_history(setup, config, target_img):
    """
    Instancia el motor genético con un setup específico y devuelve el historial de fitness.
    """
    w, h = target_img.size
    alg_config = config['algorithm']
    generations = alg_config.get('max_generations', 100)
    mutation_rate = alg_config['mutation'].get('rate', 0.1)

    sel_strategy = MethodFactory.create_selection(setup['selection'])
    cross_strategy = MethodFactory.create_crossover(setup['crossover'])
    mut_strategy = MethodFactory.create_mutation(setup['mutation'], mutation_rate, w, h)
    
    surv_strategy = MethodFactory.create_survival(setup['survival'], selection_method=sel_strategy)

    engine = GeneticEngine(
        target_img, 
        config, 
        sel_strategy, 
        cross_strategy, 
        mut_strategy, 
        surv_strategy
    )

    fitness_history = []
    
    for gen in range(generations):
        engine.evolve_step(current_gen=gen, max_generations=generations)
        best_fitness = max(ind.fitness for ind in engine.pop)
        fitness_history.append(best_fitness)
        
        if gen % 20 == 0:
            print(f"  Gen {gen}/{generations} | Fitness: {best_fitness:.6f}")
            
    return fitness_history

def main():
    config = load_config()
    
    img_path = config['image']['path']
    res_w, res_h = config['image']['resolution']
    target_img = Image.open(img_path).resize((res_w, res_h))
    
    os.makedirs("figs", exist_ok=True)

    print("=== INICIANDO BENCHMARK ===")
    
    for category, methods_to_test in OPTIONS.items():
        print(f"\n--- Evaluando Categoría: {category.upper()} ---")
        
        category_dir = os.path.join("figs", category)
        os.makedirs(category_dir, exist_ok=True)
        
        for method in methods_to_test:
            print(f"> Probando {category}: {method}...")
            
            current_setup = DEFAULTS.copy()
            current_setup[category] = method
            
            history = run_ga_and_get_history(current_setup, config, target_img)
            
            plt.figure(figsize=(8, 5))
            plt.plot(history, label=f'{method}', color='blue', linewidth=2)
            plt.title(f'Evolución del Fitness\nCategoría: {category} | Método: {method}', fontsize=14)
            plt.xlabel('Generaciones', fontsize=12)
            plt.ylabel('Fitness', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            save_path = os.path.join(category_dir, f"{method}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close() 
            
            print(f"  Guardado -> {save_path}")

    print("\n=== BENCHMARK FINALIZADO CON ÉXITO ===")

if __name__ == "__main__":
    main()