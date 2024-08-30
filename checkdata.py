import json
import matplotlib.pyplot as plt

# Carregar o arquivo JSON
with open('gestures_data.json', 'r') as file:
    data = json.load(file)

import torch
import numpy as np
import matplotlib.pyplot as plt

keyboard = {
    'q': (0.05, 0.16666666666666669), 'w': (0.15, 0.16666666666666669), 'e': (0.25, 0.16666666666666669),
    'r': (0.35, 0.16666666666666669), 't': (0.45, 0.16666666666666669), 'y': (0.55, 0.16666666666666669),
    'u': (0.65, 0.16666666666666669), 'i': (0.75, 0.16666666666666669), 'o': (0.85, 0.16666666666666669),
    'p': (0.95, 0.16666666666666669), 
    
    'a': (0.1, 0.5), 's': (0.2, 0.5), 'd': (0.3, 0.5),
    'f': (0.4, 0.5), 'g': (0.5, 0.5), 'h': (0.6, 0.5), 'j': (0.7, 0.5), 'k': (0.8, 0.5),
    'l': (0.9, 0.5), 

    'z': (0.2, 0.8333333333333334), 'x': (0.3, 0.8333333333333334),
    'c': (0.4, 0.8333333333333334), 'v': (0.5, 0.8333333333333334), 'b': (0.6, 0.8333333333333334),
    'n': (0.7, 0.8333333333333334), 'm': (0.8, 0.8333333333333334)
}

def transform_coordinates(x, y):
    """ Apply the basis transformation to the coordinates. """
    x_new = x * 2 - 1
    y_new = -y * 2 + 1
    return x_new, y_new

def get_coordinates(sentence):
    coordinates = []

    for char in sentence:
        if char in keyboard:
            x, y = keyboard[char]
            x_transformed, y_transformed = transform_coordinates(x, y)
            coordinates.append([x_transformed, y_transformed])
            
    return coordinates

def get_word_prototype(word):
    try:
        coordinates = get_coordinates(word)

        x, y = zip(*coordinates)
        x, y = np.array(x), np.array(y)

        n = 128
        k = len(coordinates)

        path_points = []

        for i in range(len(coordinates) - 1):
            num_points = (n - k) // (k - 1)  # Calculate the number of points between each pair of coordinates

            x_values = np.linspace(coordinates[i][0], coordinates[i+1][0], num_points, endpoint=False) 
            y_values = np.linspace(coordinates[i][1], coordinates[i+1][1], num_points, endpoint=False)

            path_points.extend(list(zip(x_values, y_values)))
        
        path_points.append(coordinates[-1])

        if len(path_points) < 128:
            while len(path_points) < 128:
                path_points.append(coordinates[-1])
        elif len(path_points) > 128:
            path_points = path_points[:128]

        path_pointsnp = np.array(path_points)

        ones = np.zeros((128, 1))
        path_pointsnp = np.hstack((path_pointsnp, ones))

        # Plot the word path
        plt.figure(figsize=(10, 6))
        plt.plot(path_pointsnp[:, 0], path_pointsnp[:, 1], marker='o')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title(f"Path for word: {word}")
        plt.show()

        return torch.tensor(path_pointsnp, dtype=torch.float32)
    except Exception as e:
        print(f'Error with word: {word}: {e}')
        return torch.zeros((128, 3), dtype=torch.float32)

def get_batch_prototypes(words):
    prototypes = []
    for word in words:
        prototypes.append(get_word_prototype(word.lower()))
    
    return torch.stack(prototypes)

# Função para plotar um gesto
def plot_gesture(gesture):
    x = [point[0] for point in gesture['path']]
    y = [-point[1] for point in gesture['path']]
    z = [point[2] for point in gesture['path']]

    for item in x:
        if item >1:
            print(item)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title(f"Gesto: {gesture['word']}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    # plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()

# Plotar o primeiro gesto como exemplo
plot_gesture(data[2])
words = ['bottle']
sla = get_batch_prototypes(words)
