from sklearn import metrics
import numpy as np
import json


if __name__ == '__main__':
    model_name = 'gpt4_zsl'
    label_path = f'./root/test_reflexion/evaluation/{model_name}.jsonl'
    prediction_path = f'./root/bert_cls_sampled3/evaluation_{model_name}/predict_results.txt'

    with open(prediction_path) as f:
        raw = f.readlines()
        raw = raw[1:]
        gpt4_nu = []
        for r in raw:
            gpt4_nu.append(r.split('\t')[1].strip())

    with open(label_path) as f:
        raw = f.readlines()
        label = []
        for r in raw:
            label.append(json.loads(r)['label'])

    label_str = "Chef, Classical Music Artist, Table Tennis Player, Entomologist, Lacrosse Player, Astronaut, Medician, Fashion Designer, Horse Trainer, Ambassador, Photographer, Engineer, Formula One Racer, Comedian, Martial Artist, Chess Player, Painter, Soccer Player, Tennis Player, Architect, Cyclist, Basketball Player, Congressman, Baseball Player"
    label_str_list = label_str.split(', ')
    str2idx = {}
    for i, s in enumerate(label_str_list):
        str2idx[s] = i

    gpt4_nu_idx = []
    for d in gpt4_nu:
        gpt4_nu_idx.append(str2idx[d])

    label_idx = []
    for d in label:
        label_idx.append(str2idx[d])

    macro_averaged_precision = metrics.precision_score(gpt4_nu_idx, label_idx, average='macro')
    macro_averaged_recall = metrics.recall_score(gpt4_nu_idx, label_idx, average='macro')
    macro_averaged_f1 = metrics.f1_score(gpt4_nu_idx, label_idx, average='macro')

    print(f"macro_averaged_precision: {macro_averaged_precision}\n")
    print(f"macro_averaged_recall: {macro_averaged_recall}\n")
    print(f"macro_averaged_f1: {macro_averaged_f1}\n")
