from sklearn import metrics
import numpy as np
import json


if __name__ == '__main__':
    # model_name = 'gpt4_u_preview'
    # label_path = f'./root/test_reflexion/evaluation/{model_name}.jsonl'
    # prediction_path = f'./root/bert_cls_sampled3/evaluation_{model_name}/predict_results.txt'
    prediction_path = './root/reddit_llama3_cls/azure/evaluation_reddit_clss_reddit_clss_5_act_meta-llama-Llama-2-70b-chat-hf_pe_gpt4-turbo-128k_ue_meta-llama-Meta-Llama-3-70B-Instruct_parser_gpt-35-turbo-0301_pass_at_k_1_reddit_no-utility_False_COT_False_p-threshold_10_mem-len_3.txt'
    label_path = './benchmarks/Reddit_synthetic/synthetic_dataset.jsonl'

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
            # label.append(json.loads(r)['label'])
            label.append(json.loads(r)['personality']['occupation'])

    # label_str = "Chef, Classical Music Artist, Table Tennis Player, Entomologist, Lacrosse Player, Astronaut, Medician, Fashion Designer, Horse Trainer, Ambassador, Photographer, Engineer, Formula One Racer, Comedian, Martial Artist, Chess Player, Painter, Soccer Player, Tennis Player, Architect, Cyclist, Basketball Player, Congressman, Baseball Player"
    label_str = ("software engineer, shop owner, surgeon, structural engineer, data scientist, part-time graphic designer, "
                 "college professor, web developer, part-time film editor, fashion designer, marketing manager, psychologist,"
                 " architect, part-time retail worker, part-time waiter, retiree, game developer, junior software developer, "
                 "high school principal, nurse, lawyer, art curator, financial manager, museum curator, chef, university professor,"
                 " part-time tutor, retired CEO, business development manager, astronomer, financial analyst, graphic designer, "
                 "research scientist, environmental consultant, health inspector")
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
    acc = metrics.accuracy_score(gpt4_nu_idx, label_idx)

    print(f"macro_averaged_precision: {macro_averaged_precision}\n")
    print(f"macro_averaged_recall: {macro_averaged_recall}\n")
    print(f"macro_averaged_f1: {macro_averaged_f1}\n")
    print(f"accuracy: {acc}\n")
