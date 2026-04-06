import os
import csv
import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Engine import GeneticEngine
from methods.factory import MethodFactory

# 1. Defaults recomendados para usar como base de comparación
DEFAULTS = {
    'selection': 'elite',
    'crossover': 'one_point',
    'mutation': 'nouniform',
    'survival': 'additive'
}

# 2. Todas las opciones disponibles para iterar
OPTIONS = {
    'selection': ['elite', 'ruleta', 'universal', 'boltzmann', 'ranking', 'deterministic_tournament', 'probabilistic_tournament'],
    'crossover': ['one_point', 'two_point', 'uniform', 'annular'],
    'mutation': ['gene', 'uniform', 'nouniform'],
    'survival': ['additive', 'exclusive']
}

def smooth(y, box_pts):
    """Aplica una media móvil para suavizar curvas ruidosas"""
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')

    padding = np.full(box_pts - 1, y_smooth[0]) if len(y_smooth) > 0 else []
    return np.concatenate((padding, y_smooth))

def load_config(path="config.yaml"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def run_ga_and_get_history(setup, config, target_img):
    w, h = target_img.size
    
    # Extraemos parámetros (Ojo: forzamos límite estricto sin infinito en benchmark)
    pop_size = config['algorithm'].get('pop_size', 50)
    n_triangles = config['algorithm'].get('n_triangles', 50)
    generations = config['algorithm'].get('generations', 20000)
    mutation_rate = config['algorithm']['mutation'].get('rate', 0.1)

    # Factory instantiation
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
        
        # Guardamos historial (evaluado sobre el mejor sujeto y de forma espaciada para reducir tamaño de archivo)
        if gen % 50 == 0 or gen == generations - 1:
            best = max(engine.pop, key=lambda ind: ind.fitness)
            fitness_history.append(best.fitness)
        
        if gen > 0 and gen % 1000 == 0:
            print(f"    Gen {gen}/{generations} | Fitness: {best.fitness:.6f}")
            
    # Devolvemos tanto el historial numérico como la imagen generada
    final_best = max(engine.pop, key=lambda ind: ind.fitness)
    return fitness_history, final_best.render()

def main():
    config = load_config()
    
    # Resolucion y lista de imagenes desde benchmark si existe, si no fallback
    res_w, res_h = config['image'].get('resolution', [64, 64])
    images = config.get('benchmark', {}).get('images', ["images/polonia.png"])
    
    print("=== INICIANDO BENCHMARK ===")
    print(f"Imágenes a testear: {images}")
    
    # Bucle 1: Por cada Imagen
    for img_path in images:
        print(f"\n=====================================")
        print(f"PROCESANDO IMAGEN: {img_path}")
        print(f"=====================================")
        
        try:
            target_img = Image.open(img_path).resize((res_w, res_h))
        except Exception as e:
            print(f"[ERROR] No se pudo cargar '{img_path}'. Saltando a la siguiente...")
            continue
            
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for category, methods_to_test in OPTIONS.items():
            if len(methods_to_test) <= 1:
                continue # No hay nada que comparar si hay 1 solo método listado
                
            print(f"\n--- Evaluando Comparativa de Categoría: {category.upper()} ---")
            
            category_dir = os.path.join("results", "benchmark", img_name, category)
            
            # --- MECANISMO DE RESUME / CACHÉ A NIVEL CATEGORÍA REMOVIDO ---
            # (Lo haremos a nivel de método guardando/leyendo CSV)
            
            # Crear directorio si no existe
            os.makedirs(category_dir, exist_ok=True)
            
            # Preparamos UNA única figura donde dibujaremos las curvas de todos los métodos
            plt.figure(figsize=(10, 6))
            
            # También iremos guardando el fitness final de cada uno para hacer un gráfico de barras
            final_fitnesses = {}
            
            # Archivo global de reporte
            summary_csv_path = os.path.join("results", "benchmark", "summary_results.csv")
            # Crear cabeceras si no existe
            if not os.path.exists(summary_csv_path):
                with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["image", "category", "method", "final_fitness", "history_path", "image_path"])
            
            # Bucle 3: Ejecutamos cada método para esa categoría
            for method in methods_to_test:
                print(f"  > Evaluando {method}...")
                
                history_csv_path = os.path.join(category_dir, f"history_{method}.csv")
                img_save_path = os.path.join(category_dir, f"resultado_{method}.png")
                
                # --- MECANISMO DE CACHÉ (LECTURA DE CSV) ---
                if os.path.exists(history_csv_path) and os.path.exists(img_save_path):
                    print(f"    [CACHE] Encontrado history en CSV. Cargando sin simular...")
                    with open(history_csv_path, 'r', encoding='utf-8') as f:
                        history = [float(line.strip()) for line in f if line.strip()]
                else:
                    # Armar la configuración temporal
                    current_setup = DEFAULTS.copy()
                    current_setup[category] = method
                    
                    # Ejecutar algoritmos para conseguir el historial de la generación y la imagen
                    history, result_image = run_ga_and_get_history(current_setup, config, target_img)
                    
                    # --- GUARDADO EN CSV Y CACHÉ ---
                    # 1. Guardar el array history en su propio CSV para reconstruir graficos futuros
                    with open(history_csv_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(str(v) for v in history))
                    
                    # 2. Guardar la imagen de ese método
                    result_image.save(img_save_path)
                    print(f"    Imagen final guardada en: {img_save_path}")
                    
                    # 3. Anexar al Super CSV de reporte Global
                    with open(summary_csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([img_name, category, method, history[-1], history_csv_path, img_save_path])
                
                # Registramos fitness final para el plot de barras
                final_fitnesses[method] = history[-1]
                
                # Suavizamos la línea (ej: ventana de 5% de las generaciones)
                window_size = max(5, int(len(history) * 0.05))
                smoothed_history = smooth(history, window_size)
                
                # Dibujamos iterativamente la línea en el plot
                # Usamos un alpha=0.9 para que se vean bien
                plt.plot(smoothed_history, label=f'{method} (Final: {history[-1]:.5f})', linewidth=2.0, alpha=0.8)
                

                
            # --- GUARDAR GRÁFICO DE LÍNEAS (HISTORIAL) ---
            plt.title(f'Comparativa de {category.upper()} (Media Móvil)\nen imagen "{img_name}"', fontsize=14)
            plt.xlabel('Generaciones', fontsize=12)
            plt.ylabel('Fitness', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Poner leyenda afuera si hay muchos metodos
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plot_save_path = os.path.join(category_dir, f"plot_historial_{category}.png")
            plt.savefig(plot_save_path, bbox_inches='tight')
            plt.close() 
            
            # --- GUARDAR GRÁFICO DE BARRAS (FITNESS FINAL) ---
            plt.figure(figsize=(10, 6))
            methods_keys = list(final_fitnesses.keys())
            fitness_vals = list(final_fitnesses.values())
            
            bars = plt.bar(methods_keys, fitness_vals, color='skyblue', edgecolor='black')
            plt.title(f'Fitness Final Alcanzado por Método de {category.upper()}\nen imagen "{img_name}"', fontsize=14)
            plt.ylabel('Fitness Final', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Agregar el valor en texto encima de cada barra
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.5f}', ha='center', va='bottom', fontsize=9, rotation=90)
            
            bar_save_path = os.path.join(category_dir, f"plot_barras_{category}.png")
            plt.savefig(bar_save_path, bbox_inches='tight')
            plt.close()
            
            print(f"*** Gráficos comparativos guardados en directorio: {category_dir} ***")

    print("\n=== BENCHMARK FINALIZADO CON ÉXITO ===")

if __name__ == "__main__":
    main()