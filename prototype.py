import matplotlib.pyplot as plt 
import numpy as np 

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
    return [(keyboard[char][0], keyboard[char][1]) for char in sentence]

def get_word_prototype(word):
    coordinates = get_coordinates(word)

    fig, ax = plt.subplots()

    for key, coord in keyboard.items():
        if key == ' ':
            ax.add_patch(plt.Rectangle((coord[1]-1.13, -coord[0]-0.6), 4.75, 1, fill=None))
        else:
            ax.text(coord[1], -coord[0], key, ha='center', va='center')
            ax.add_patch(plt.Rectangle((coord[1]-0.4, -coord[0]-0.4), 0.8, 0.8, fill=None))

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

    # ax.plot(path_pointsnp[:, 1], -path_pointsnp[:, 0], 'bo', markersize=3, linestyle='-')
    # ax.set_aspect('equal')
    # ax.set_xlim(-1, 10)
    # ax.set_ylim(-4, 1)
    # ax.axis('off')

    # plt.show()

    return path_pointsnp

def get_batch_prototypes(words):
    prototypes = np.array([get_word_prototype(word) for word in words])
    return prototypes

if __name__ == '__main__':
    words = ['insper', 'rolamole', 'cadeira']
    batch_prototypes = get_batch_prototypes(words)
    print(batch_prototypes)
    print(batch_prototypes.shape)
