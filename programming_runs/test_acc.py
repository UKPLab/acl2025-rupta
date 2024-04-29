from utils import enumerate_resume, write_jsonl, make_printv
from generators import generator_factory, model_factory
import tqdm
from typing import List
import ezsheets
import time


def run_test_acc(
    dataset: List[dict],
    pe_model_name: str,
    language: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    p_threshold: int,
    no_utility: bool,
    is_leetcode: bool = False,
    rag_data_path: str = '',
    rag_num: int = 5,
    rag_embed_cache_dir: str = '',
) -> None:
    gen = generator_factory(language)
    model = model_factory(pe_model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    confidence_score_list = []
    candidate_list_list = []
    success = []
    result = {}

    for i, item in enumerate_resume(tqdm.tqdm(dataset), log_path):
        privacy_confidence_evaluation = gen.privacy_confidence_evaluation(model, item['anonymized_text'], item['people'])
        confidence_score = int(privacy_confidence_evaluation["Confidence Score"])
        confidence_score_list.append(confidence_score)

        if "candidate_list" in item.keys():
            candidate_list = item['candidate_list']
        else:
            candidate_list = None
        privacy_seletion_evaluation = gen.privacy_selection_evaluation(model, item['anonymized_text'], item['people'],
                                                                      candidate_list)
        if "candidate_list" not in item.keys():
            candidate_list_list.append(privacy_seletion_evaluation["candidate_list"])
        if privacy_seletion_evaluation["success"]:
            num_success += 1
            success.append(True)
        else:
            num_success += 0
            success.append(False)

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
    result['confidence_score'] = confidence_score_list
    result['rank_avg'] = sum(confidence_score_list)/len(confidence_score_list)
    result['success'] = success
    result['success_rate'] = num_success/num_items
    result['num_success'] = num_success
    result['candidate_list'] = candidate_list_list
    model.print_usage()
    print(f"log path: {log_path}\n")
    write_jsonl(log_path, [result], append=False)
    ss = ezsheets.Spreadsheet('1-uHO5DnE32WmImaucvHaVMvasO2mGh2eqWfWYksXljI')
    sheet = ss[0]
    update_idx = sheet.getColumn(1).index('') + 1
    update_row = sheet.getRow(update_idx)

    name2column = {'gpt-35-turbo-0301': 7, 'gpt-4': 1, 'gpt4-turbo-128k': 4}

    update_row[name2column[model.name]], update_row[name2column[model.name] + 1] = (model.prompt_tokens,
                                                                                    model.completion_tokens)
    update_row[0] = time.ctime()
    sheet.refresh()
    update_idx = sheet.getColumn(1).index('') + 1
    sheet.updateRow(update_idx, update_row)