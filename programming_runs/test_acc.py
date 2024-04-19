from utils import enumerate_resume, write_jsonl, make_printv
from generators import generator_factory, model_factory
import tqdm
from typing import List


def run_test_acc(
    dataset: List[dict],
    model_name: str,
    language: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    p_threshold: int,
    no_utility: bool,
    is_leetcode: bool = False
) -> None:
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    rank = []
    success = []
    result = {}
    completion_tokens = 0
    prompt_tokens = 0
    for i, item in enumerate_resume(tqdm.tqdm(dataset), log_path):
        privacy_evaluation = gen.privacy_reflex(model, item['anonymized_text'], 'None', p_threshold,
                                                False)
        completion_tokens += privacy_evaluation['usage_1']['completion_tokens']
        prompt_tokens += privacy_evaluation['usage_1']['prompt_tokens']
        if "usage_2" in privacy_evaluation.keys():
            completion_tokens += privacy_evaluation['usage_2']['completion_tokens']
            prompt_tokens += privacy_evaluation['usage_2']['prompt_tokens']
        if "usage_3" in privacy_evaluation.keys():
            completion_tokens += privacy_evaluation['usage_3']['completion_tokens']
            prompt_tokens += privacy_evaluation['usage_3']['prompt_tokens']
        if privacy_evaluation["Confirmation"] == "Yes":
            num_success += 1
            success.append(True)
        else:
            num_success += 0
            success.append(False)
        rank.append(privacy_evaluation["rank"])

        print_v(
            f'completed {i+1}/{num_items}: acc = {round(num_success/(i+1), 2)}')
    result['rank'] = rank
    result['rank_avg'] = sum(rank)/len(rank)
    result['success'] = success
    result['success_rate'] = num_success/num_items
    result['num_success'] = num_success
    print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")
    print(f"log path: {log_path}\n")
    write_jsonl(log_path, [result], append=False)