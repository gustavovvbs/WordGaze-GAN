import os
import re
import json
import numpy as np
import pandas as pd

def preprocess_gesture(df):
    num_points = len(df)
    if num_points > 128:
        sampled_indices = np.sort(np.random.choice(range(1, num_points - 1), 126, replace=False))
        sampled_df = df.iloc[sampled_indices]
        df = pd.concat([df.iloc[[0]], sampled_df, df.iloc[[-1]]]).reset_index(drop=True)
    elif num_points < 128:
        df = df.apply(pd.to_numeric, errors='ignore')
        df = df.set_index(np.linspace(0, 1, num_points)).reindex(np.linspace(0, 1, 128)).interpolate('linear').reset_index(drop=True)
    return df

def normalize_and_adjust_timestamps(df):
    keyb_width = df.iloc[0]['keyb_width']
    keyb_height = df.iloc[0]['keyb_height']

    # normaliza x e y em relacao às dimensoes do teclado
    df["x_pos"] = (df["x_pos"] / keyb_width) * 2 - 1
    df["y_pos"] = -(df["y_pos"] / keyb_height) * 2 + 1
    # for x in df['x_pos']:
    #     if x > 1 or x < -1:
    #         print(x)
    # for y in df['y_pos']:
    #     if y > 1 or y < -1:
    #         print(y)


    #caclcula os timestamps(diferencas entre dois pontos seguidos e divide por 1000 p ser em segundos)
    df["timestamp"] = (df["timestamp"].diff().fillna(0) / 1000).astype(float)
    df = df[df["timestamp"] >= 0]
    df = df[df["timestamp"] <= 1]

    if df["timestamp"].min() < 0:
        print("Negative timestamps found")
        print(df[df["timestamp"] < 0])

    # Calculate the number of rows needed to reach 128
    num_rows_to_add = 128 - len(df)

    if num_rows_to_add > 0:
        # Create a DataFrame with the padding rows
        padding_rows = pd.DataFrame([df.iloc[-1]] * num_rows_to_add, columns=df.columns)
        
        # Concatenate the original DataFrame with the padding rows
        df = pd.concat([df, padding_rows], ignore_index=True)

    tensors = df[["x_pos", "y_pos", "timestamp"]].to_numpy()
    return tensors

def log_treater(folder):
    logs = os.listdir(folder)
    logs = sorted(logs)

    all_gestures = []

    column_names = [
        "sentence", "timestamp", "keyb_width", "keyb_height", "event",
        "x_pos", "y_pos", "x_radius", "y_radius", "angle", "word", "is_err"
    ]

    for log in logs: 
        if log.endswith('.log'):
            with open(os.path.join(folder, log)) as file:
                content = file.readlines()
                for line_number, line in enumerate(content[1:], 1):  # pula o header
                    line = line.strip()
                    if not line:
                        continue  #pula linha vazia
                    try:
                        values = re.split(r'\s+', line)
                        entry = dict(zip(column_names, values))
                    except Exception as e:
                        print(f"Skipping invalid line {line_number} in {log} due to error: {e}")
                        print(f"Line content: {line}")
                        continue

                    if 'word' not in entry:
                        print(f"Skipping line {line_number} in {log} due to missing 'word' key")
                        print(f"Line content: {line}")
                        continue

                    required_keys = ['timestamp', 'x_pos', 'y_pos', 'word', 'keyb_width', 'keyb_height']
                    if not all(key in entry for key in required_keys):
                        print(f"Skipping line {line_number} in {log} due to missing required keys")
                        print(f"Line content: {line}")
                        continue

                    try:
                        tensor = {
                            'timestamp': int(entry['timestamp']),
                            'x_pos': int(entry['x_pos']),
                            'y_pos': int(entry['y_pos']),
                            'keyb_width': float(entry['keyb_width']),
                            'keyb_height': float(entry['keyb_height'])
                        }
                    except ValueError as e:
                        print(f"Skipping line {line_number} in {log} due to invalid values: {e}")
                        print(f"Line content: {line}")
                        continue

                    gesture_entry = {'word': entry['word'], 'path': [tensor]}
                    all_gestures.append(gesture_entry)


    # separa por palavra
    grouped_gestures = {}
    for gesture in all_gestures:
        word = gesture['word']
        if word not in grouped_gestures:
            grouped_gestures[word] = []
        grouped_gestures[word].append(gesture['path'][0])

    # processa os gestos por palavra
    processed_gestures = []
    for word, gestures in grouped_gestures.items():
        df = pd.DataFrame(gestures)
        df = preprocess_gesture(df)
        tensors = normalize_and_adjust_timestamps(df)
        processed_gestures.append({'word': word, 'path': tensors.tolist()})

    # salva todos os gestos em um json
    with open('gestures_data.json', 'w') as f:
        json.dump(processed_gestures, f)

    print(f"Processed and saved gestures data in gestures_data.json")

if __name__ == '__main__':
    log_treater('C:/Users/gugu1/Documents/GitHub/swipetest/experiments/logs/')