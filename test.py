import os 
import json

with open('gestures_data.json', 'r') as f:
    data = json.load(f)

print(len(data[93]['path']))