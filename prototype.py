import matplotlib.pyplot as plt 
import numpy as np 

keyboard = {
        'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
        'a': (1, 0.25), 's': (1, 1.25), 'd': (1, 2.25), 'f': (1, 3.25), 'g': (1, 4.25), 'h': (1, 5.25), 'j': (1, 6.25), 'k': (1, 7.25), 'l': (1, 8.25),
        'z': (2, 0.75), 'x': (2, 1.75), 'c': (2, 2.75), 'v': (2, 3.75), 'b': (2, 4.75), 'n': (2, 5.75), 'm': (2, 6.75),
        ' ': (3, 3.5)  
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
            num_points = (n-k) // k

            x_values = np.linspace(coordinates[i][0], coordinates[i+1][0], num_points) 
            y_values = np.linspace(coordinates[i][1], coordinates[i+1][1], num_points)

            path_points.extend(list(zip(x_values, y_values)))
        
        path_points.append(coordinates[-1])

        path_pointsnp = np.array(path_points)

        ax.plot(path_pointsnp[:, 1], -path_pointsnp[:, 0], 'bo', markersize=3, linestyle='-')
        ax.set_aspect('equal')
        ax.set_xlim(-1, 10)
        ax.set_ylim(-4, 1)
        ax.axis('off')

        plt.show()

        return path_points


            


