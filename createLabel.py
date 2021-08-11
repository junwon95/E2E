import csv

import pandas as pd

f = open('C:/Users/junwonseo95/Desktop/E2E DATASET/confidence98_transcripts_result.txt', 'r', encoding='utf8')

lines = f.readlines()

print('create_char_labels started..')

label_list = list()
label_freq = list()

with open('data/aihub/aihub_labels.csv', 'r', encoding='utf8') as f2:
    labels = csv.reader(f2, delimiter=',')
    next(labels)

    for row in labels:
        label_list.append(row[1])
        label_freq.append(row[2])


for i, line in enumerate(lines):
    print(i)
    for ch in line.strip('\n'):
        if ch not in label_list:
            label_list.append(ch)
            label_freq.append(1)
            print(ch)
        elif label_list.index(ch) > 2000:
            label_freq[label_list.index(ch)] += 1

label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

label_list = label_list[3:]
label_freq = label_freq[3:]

for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
    label['id'].append(idx+3)
    label['char'].append(ch)
    label['freq'].append(freq)

label_df = pd.DataFrame(label)
label_df.to_csv("data/aihub/e2e_labels.csv", encoding="utf-8", index=False)
