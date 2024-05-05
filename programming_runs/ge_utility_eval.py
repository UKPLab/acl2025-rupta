import tiktoken
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm


def string_edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)

    # 有一个字符串为空串
    if n * m == 0:
        return n + m

    # DP 数组
    D = [[0] * (m + 1) for _ in range(n + 1)]

    # 边界状态初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    # 计算所有 DP 值
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if s1[i - 1] != s2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)

    return D[n][m]


def token_edit_distance(s1, s2):
    enc = tiktoken.get_encoding("cl100k_base")
    token_list1 = enc.encode(s1)
    token_list2 = enc.encode(s2)
    #print(token_list1)
    #print(token_list2)
    return string_edit_distance(token_list1, token_list2)


if __name__ == '__main__':
    root_path = './root/test_reflexion/evaluation'
    model_path = 'azure.jsonl'
    original_data = []
    distance_list = []
    similarity_list = []
    with open(os.path.join(root_path, model_path), 'r') as f:
        raw = f.readlines()
        for r in raw:
            original_data.append(json.loads(r))
    emb_model = SentenceTransformer("all-mpnet-base-v2")
    for d in tqdm(original_data):
        distance_list.append(token_edit_distance(d['text'], d['anonymized_text']))
        o_emb = emb_model.encode(d['text'])
        a_emb = emb_model.encode(d['anonymized_text'])
        similarity_list.append(o_emb.dot(a_emb))
    print(f"Leveshtein distance: {sum(distance_list)/len(distance_list)}\n")
    print(f"Similarity: {sum(similarity_list)/len(similarity_list)}\n")
