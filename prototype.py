import torch
import numpy as np
import matplotlib.pyplot as plt

keyboard = {
    'q': (-0.9, -0.6666666666666666), 'w': (-0.7, -0.6666666666666666), 'e': (-0.5, -0.6666666666666666),
    'r': (-0.3, -0.6666666666666666), 't': (-0.1, -0.6666666666666666), 'y': (0.1, -0.6666666666666666),
    'u': (0.3, -0.6666666666666666), 'i': (0.5, -0.6666666666666666), 'o': (0.7, -0.6666666666666666),
    'p': (0.9, -0.6666666666666666), 
    
    'a': (-0.8, 0.0), 's': (-0.6, 0.0), 'd': (-0.4, 0.0),
    'f': (-0.2, 0.0), 'g': (0.0, 0.0), 'h': (0.2, 0.0), 'j': (0.4, 0.0), 'k': (0.6, 0.0),
    'l': (0.8, 0.0), 

    'z': (-0.6, 0.6666666666666666), 'x': (-0.4, 0.6666666666666666),
    'c': (-0.2, 0.6666666666666666), 'v': (0.0, 0.6666666666666666), 'b': (0.2, 0.6666666666666666),
    'n': (0.4, 0.6666666666666666), 'm': (0.6, 0.6666666666666666)
}

def get_coordinates(sentence):
    coordinates = []

    for char in sentence:
        if char in keyboard:
            coordinates.append([keyboard[char][0], keyboard[char][1]])
            
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
        plt.figure(figsize=(6, 6))
        plt.plot(path_pointsnp[:, 0], path_pointsnp[:, 1], marker='o')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
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

if __name__ == '__main__':
    
    words = ['Hello']

    sla = get_batch_prototypes(words)
    
    print(sla)
    print(sla.shape)
