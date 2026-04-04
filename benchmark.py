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
    'selection': ['elite', 'ruleta', 'boltzmann', 'universal', 'ranking', 'deterministic_tournament', 'probabilistic_tournament'],
    'crossover': ['one_point', 'two_point', 'uniform', 'annular'],
    'mutation': ['gene', 'uniform', 'nouniform'],
    'survival': ['additive', 'exclusive']
}

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_ga_and_get_history(setup, config, target_img):
    """
    Instancia el motor genético con un setup específico y devuelve el historial de fitness
    y el MEJOR individuo de la última generación.
    """
    w, h = target_img.size
    alg_config = config['algorithm']
    generations = alg_config.get('max_generations', 100)
    mutation_rate = alg_config['mutation'].get('rate', 0.1)

    # Creamos las estrategias para esta corrida específica
    selection_strategy = MethodFactory.create_selection(
        setup['selection'], 
        tournament_size=alg_config['selection'].get('tournament_size', 3),
        t0=100.0, tc=1.0
    )
    crossover_strategy = MethodFactory.create_crossover(setup['crossover'])
    mutation_strategy = MethodFactory.create_mutation(setup['mutation'], mutation_rate, w, h)
    survival_strategy = MethodFactory.create_survival(setup['survival'], selection_method=selection_strategy)

    # Inicializamos el motor
    engine = GeneticEngine(target_img, config, selection_strategy, crossover_strategy, mutation_strategy, survival_strategy)

    history = []
    
    # Ciclo evolutivo
    for gen in range(generations):
        engine.evolve_step(current_gen=gen, max_generations=generations)
        best = max(engine.pop, key=lambda ind: ind.fitness)
        history.append(best.fitness)

    # Obtener el mejor de todos al finalizar
    best_final = max(engine.pop, key=lambda ind: ind.fitness)
    
    return history, best_final

def main():
    config = load_config()
    
    img_path = config['image']['path']
    res_w, res_h = config['image']['resolution']
    target_img = Image.open(img_path).resize((res_w, res_h))
    
    os.makedirs("figs", exist_ok=True)

    print("=== INICIANDO BENCHMARK CON SISTEMA DE CACHÉ ===")
    
    for category, methods_to_test in OPTIONS.items():
        print(f"\n--- Evaluando Categoría: {category.upper()} ---")
        
        category_dir = os.path.join("figs", category)
        os.makedirs(category_dir, exist_ok=True)
        
        for method in methods_to_test:
            print(f"> Probando {category}: {method}...")
            
            # Definimos las rutas de los archivos de salida
            csv_path = os.path.join(category_dir, f"{method}.csv")
            img_path = os.path.join(category_dir, f"{method}.png")
            graph_path = os.path.join(category_dir, f"{method}_grafico.png")
            
            history = []
            
            # CHECK CACHÉ: Verificamos si ya corrimos este método antes
            if os.path.exists(csv_path):
                print(f"  [CACHÉ] Archivo encontrado. Saltando ejecución y leyendo datos de {csv_path}")
                with open(csv_path, 'r', encoding='utf-8') as f:
                    history = [float(line.strip()) for line in f.readlines()]
            else:
                print(f"  [NUEVO] Ejecutando el motor genético...")
                current_setup = DEFAULTS.copy()
                current_setup[category] = method
                
                # Corremos el algoritmo y recibimos historia + mejor individuo
                history, best_individual = run_ga_and_get_history(current_setup, config, target_img)
                
                # GUARDAR OUTPUT 1: El archivo .csv con la historia
                with open(csv_path, 'w', encoding='utf-8') as f:
                    for val in history:
                        f.write(f"{val}\n")
                        
                # GUARDAR OUTPUT 2: La imagen del mejor individuo
                best_individual.render().save(img_path)
                print(f"  [GUARDADO] CSV y PNG guardados en la carpeta {category_dir}")
            
            # GUARDAR OUTPUT 3: El Gráfico
            plt.figure(figsize=(8, 5))
            plt.plot(history, label=f'{method}', color='blue', linewidth=2)
            plt.title(f'Evolución del Fitness\nCategoría: {category} | Método: {method}', fontsize=14)
            plt.xlabel('Generaciones', fontsize=12)
            plt.ylabel('Fitness', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.savefig(graph_path)
            plt.close()

if __name__ == '__main__':
    main()