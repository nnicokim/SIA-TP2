import numpy as np
from PIL import Image, ImageDraw

class Individual:
    def __init__(self, n_triangles, w, h, genes=None):
        self.w, self.h = w, h
        if genes is not None:
            self.genes = genes
        else:
            # Cada fila es un triángulo: [x1, y1, x2, y2, x3, y3, R, G, B, A]
            self.genes = np.random.rand(n_triangles, 10)
            # Escalamos coordenadas al tamaño de imagen y colores a 255
            self.genes[:, [0,2,4]] *= w
            self.genes[:, [1,3,5]] *= h
            self.genes[:, 6:] *= 255
        
        self.fitness = 0

    def render(self):
        img = Image.new('RGB', (self.w, self.h), (255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        for g in self.genes:
            coords = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5])]
            color = tuple(g[6:].astype(int))
            draw.polygon(coords, fill=color)
        return img