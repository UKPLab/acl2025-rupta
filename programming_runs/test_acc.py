from utils import enumerate_resume, write_jsonl, make_printv
from generators import generator_factory, model_factory

from typing import List


def run_test_acc(
    dataset: List[dict],
    model: str,
    language: str,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    p_threshold: int,
    no_utility: bool,
    is_leetcode: bool = False
) -> None:
    gen = generator_factory(language)
    model = model_factory(model)

    print_v = make_printv(verbose)

    num_items = len(dataset)
    num_success = 0
    rank = []
    success = []
    result = {}
    for i, item in enumerate_resume(dataset, log_path):
        privacy_evaluation = gen.privacy_reflex(model, item['Anonymized text'], 'None', p_threshold,
                                                no_utility)
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
    write_jsonl(log_path, [result], append=False)