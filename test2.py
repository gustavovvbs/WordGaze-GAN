import numpy as np
import torch
import matplotlib.pyplot as plt

# Definindo o layout do teclado QWERTY
keyboard_layout = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['Z','X','C','V','B','N','M']
]

# Constantes
TOTAL_POINTS = 128
ASPECT_RATIO = 1.4  # Altura / Largura

# Função para calcular os centros das teclas do teclado
def get_key_centers():
    key_centers = {}
    
    # Definir largura e altura arbitrária do teclado
    width = 10.0
    height = width * ASPECT_RATIO

    padding = width * 0.05
    key_width = (width - 2 * padding) / 10  # 10 teclas na linha superior
    key_height = height / 4

    for row_idx, row in enumerate(keyboard_layout):
        y = padding + row_idx * (key_height + 0.1) + key_height / 2
        row_padding = (width - len(row) * (key_width + 0.1) + 0.1) / 2
        
        for key_idx, key in enumerate(row):
            x = row_padding + key_idx * (key_width + 0.1) + key_width / 2
            key_centers[key.upper()] = np.array([x, y])
    
    return key_centers, width, height

# Normalizar para o intervalo [-1, 1]
def normalize(points, width, height):
    norm_points = points.copy()
    norm_points[:, 0] = 2 * (norm_points[:, 0] / width) - 1  # Normaliza o eixo X
    norm_points[:, 1] = 2 * (norm_points[:, 1] / height) - 1  # Normaliza o eixo Y
    return norm_points

# Mapeamento da palavra para centros das teclas
def map_word_to_centers(word, key_centers):
    word = word.upper()
    centers = []

    
    for char in word:
        if char in key_centers:
            centers.append(key_centers[char])
        else:
            raise ValueError(f"Character '{char}' is not present on the keyboard.")
    
    return np.array(centers)

# Gerar pontos do protótipo
def generate_prototype_points(centers):
    if centers.shape[0] == 0:
        return np.array([])

    points = [centers[0]]
    points_per_segment = (TOTAL_POINTS - len(centers)) / (len(centers) - 1)

    for i in range(1, len(centers)):
        start, end = centers[i - 1], centers[i]
        points.append(start)

        for j in range(1, int(points_per_segment) + 1):
            t = j / (int(points_per_segment) + 1)
            interpolated_point = (1 - t) * start + t * end
            points.append(interpolated_point)

    points = np.array(points)

    if len(points) > TOTAL_POINTS:
        points = points[:TOTAL_POINTS]
    elif len(points) < TOTAL_POINTS:
        last_point = points[-1]
        while len(points) < TOTAL_POINTS:
            points = np.vstack([points, last_point])

    return points

# Função para gerar protótipos para um batch de palavras
def generate_batch_prototypes(words):
    key_centers, width, height = get_key_centers()
    batch_prototypes = []
    
    for word in words:
        centers = map_word_to_centers(word, key_centers)
        points = generate_prototype_points(centers)
        normalized_points = normalize(points, width, height)
        batch_prototypes.append(normalized_points)

    # Converte para tensor PyTorch (batch_size, TOTAL_POINTS, 2)
    return torch.tensor(batch_prototypes, dtype=torch.float32)






