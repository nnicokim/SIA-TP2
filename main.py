import yaml
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

from Engine import GeneticEngine
from methods.factory import MethodFactory

def load_config(path="config.yaml"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)

def main():
    config = load_config()
    
    img_path = config['image']['path']
    try:
        res_w, res_h = config['image']['resolution']
        target_img = Image.open(img_path).resize((res_w, res_h))
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] No se pudo cargar la imagen: '{img_path}'.")
        print(f"Asegúrate de que la imagen exista en la misma carpeta o de escribir bien la ruta.")
        print(f"Detalle del error: {e}\n")
        sys.exit(1)
    
    alg_config = config['algorithm']
    
    # Instantiate methods
    selection_method = MethodFactory.create_selection(
        alg_config['selection']['method']
    )
    
    crossover_method = MethodFactory.create_crossover(
        alg_config['crossover']['method']
    )
    
    mutation_method = MethodFactory.create_mutation(
        alg_config['mutation']['method'],
        mutation_rate=alg_config['mutation']['rate'],
        w=res_w, h=res_h
    )
    
    survival_method = MethodFactory.create_survival(
        alg_config['survival']['method'],
        selection_method=selection_method
    )
    
    engine = GeneticEngine(
        target_img, 
        config, 
        selection_method, 
        crossover_method, 
        mutation_method, 
        survival_method
    )
    
    generations = alg_config['generations']
    target_fitness = alg_config.get('target_fitness', 0.0)
    
    print(f"Iniciando evolución para {img_path} ({res_w}x{res_h}) por {generations} generaciones...")
    if target_fitness > 0:
        print(f"Se frenará anticipadamente si el fitness alcanza: {target_fitness}")
    
    # Manejar salidas
    out_dir = config.get('output', {}).get('dir', "results")
    save_step = config.get('output', {}).get('save_step', 0)
    
    # Si vas a hacer muchas pruebas seguidas, capaz conviene hacer subcarpetas por imagen/run
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(out_dir, "experimental", base_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    fitness_history = []
    generations_logged = []
    gif_frames = []
    
    for gen in range(generations):
        engine.evolve_step(current_gen=gen, max_generations=generations)
        
        best = max(engine.pop, key=lambda ind: ind.fitness)
        
        if gen % 10 == 0: # Mostramos feedback textual en consola
            print(f"Generación {gen:4d} | Fitness: {best.fitness:.6f}")
            fitness_history.append(best.fitness)
            generations_logged.append(gen)
            
            # Acumulamos frames en memoria para el GIF si save_step > 0
            if save_step > 0 and gen % save_step == 0:
                frame = best.render()
                # Ya no guardamos el PNG individual en disco para no generar basura. Se acumula directo en memoria.
                gif_frames.append(frame)
                
        # Condición de corte por fitness cumplido
        if target_fitness > 0 and best.fitness >= target_fitness:
            print(f"\n¡Corte temprano! El fitness objetivo ({target_fitness}) fue alcanzado en la generación {gen}.")
            # Asegurar que grabamos la ultima gen si cortamos temprano
            fitness_history.append(best.fitness)
            generations_logged.append(gen)
            break
                
    # Evaluamos al final de nuevo por si se rompió el bucle o terminó natural
    best = max(engine.pop, key=lambda ind: ind.fitness)
    final_frame = best.render()
    
    final_filepath = os.path.join(out_dir, "final_result.png")
    final_frame.save(final_filepath)
    print(f"Evolución terminada. Mejor fitness: {best.fitness:.6f}. Resultado guardado en {final_filepath}")
    
    # 1. Generar y guardar GIF
    if save_step > 0 and gif_frames:
        gif_path = os.path.join(out_dir, "evolution_timelapse.gif")
        gif_frames.append(final_frame) # Agregamos el frame final
        # 100ms per frame, ajusta duration a gusto (100 = 10fps)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=150, 
            loop=0
        )
        print(f"GIF Timelapse guardado en {gif_path}")
        
    # 2. Generar gráfico de Fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations_logged, fitness_history, color='purple', linewidth=2)
    plt.title(f'Evolución del Fitness\n{base_name} - {res_w}x{res_h} - {engine.n_triangles} triángulos', fontsize=14)
    plt.xlabel('Generaciones', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(out_dir, "fitness_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de fitness guardado en {plot_path}")

if __name__ == "__main__":
    main()