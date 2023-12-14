import json
from random import shuffle
import os

def read_raw(folders, file_name):
    raw = []
    for folder in folders:
        if os.path.exists(os.path.join(folder, file_name)):
            file = [json.loads(line) for line in open(os.path.join(folder, file_name)).readlines()]
            raw.extend(file)
    return raw

def dataset_maker(raw):
    dataset = []
    for line in raw:
        dialog = line['dialog']
        for i, turn in enumerate(dialog[1:]):
            persona = "none"
            if 'persona_text' in turn:
                persona = turn['persona_text']
            history = dialog[i]['id'] + ': ' + dialog[i]['text'] + '\n' + turn['id'] + ': ' + turn['text']
            dataset.append({'history': history, 'persona': persona})
    return dataset

def save_jsonl(jsonl_object, file):
    if os.path.exists(file):
        raise ValueError(file + ' already exists!')
    with open(file, 'a+') as f:
        for line in jsonl_object:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


dataset_folder = 'msc/msc_personasummary'
sessions = [os.path.join(dataset_folder, 'session_' + str(i)) for i in range(1, 5)]

train_raw = read_raw(sessions, 'train.txt')
test_raw = read_raw(sessions, 'test.txt')
valid_raw = read_raw(sessions, 'valid.txt')

train = dataset_maker(train_raw)
test = dataset_maker(test_raw)
valid = dataset_maker(valid_raw)

save_jsonl(train, 'train.jsonl')
save_jsonl(test, 'test.jsonl')
save_jsonl(valid, 'valid.jsonl')