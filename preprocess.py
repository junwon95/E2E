import re

import yaml

from ksponspeech import KsponSpeechVocabulary

f = open('C:/Users/junwonseo95/Desktop/E2E DATASET/confidence98_transcripts_result.txt', 'r', encoding='utf8')
f2 = open('C:/Users/junwonseo95/Desktop/E2E DATASET/confidence98_ner_link.txt', 'r', encoding='utf8')
f3 = open('C:/Users/junwonseo95/Desktop/E2E DATASET/transcripts.txt', 'w', encoding='cp949')

lines = f.readlines()
links = f2.readlines()

with open('data/config.yaml') as f:
    opt = yaml.load(f, Loader=yaml.FullLoader)
vocab = KsponSpeechVocabulary(opt['vocab_path'])

for i, (link, line) in enumerate(zip(links,lines)):
    link = link.strip('\n')
    line = line.strip('\n')
    print(i)
    transcript = ' '.join([vocab.vocab_dict[c] for c in line])
    f3.write(link + '\t' + line + '\t' + transcript + '\n')

