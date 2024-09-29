import os 
import json 
import matplotlib.pyplot as plt 
import numpy as np
from test2 import generate_batch_prototypes

file_path = 'C:/Users/gugu1/Documents/GitHub/swipetest/experiments/processed_gesture_data.json'

with open(file_path, 'r') as file_path:
    loaded_file = json.load(file_path)




points = loaded_file[100]['swipe']
word = [loaded_file[100]['word']]

prototype = generate_batch_prototypes(word)

prot_x = [point[0] for point in prototype[0]]
prot_y = [point[1] for point in prototype[0]]


x = [point[0] for point in points]
y = [point[1] for point in points]

plt.scatter(x, y)
plt.plot(prot_x, prot_y, marker='o', color='red')
plt.gca().invert_yaxis()
plt.show()

print(loaded_file[100]['word'])