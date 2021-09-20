import json
import matplotlib.pyplot as plt


all_labels = []

with open("./label_data_0531.json", "r") as f:
    # data = json.load(f)
    for i, line in enumerate(f):
        all_labels.append(json.loads(line))

        if i == 19:
            break



print(all_labels[0]['h_samples'])