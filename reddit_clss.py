from utils import enumerate_resume, write_jsonl, make_printv
from generators import generator_factory, model_factory
import tqdm
from typing import List
import ezsheets
import time


def run_reddit_clss(
    dataset: List[dict],
    ue_model_name: str,
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
    model = model_factory(ue_model_name)

    result_str = "index\tprediction\n"
    label_set = set(("software engineer, shop owner, surgeon, structural engineer, data scientist, part-time graphic designer, "
                     "college professor, web developer, part-time film editor, fashion designer, marketing manager, psychologist,"
                     " architect, part-time retail worker, part-time waiter, retiree, game developer, junior software developer, "
                     "high school principal, nurse, lawyer, art curator, financial manager, museum curator, chef, university professor,"
                     " part-time tutor, retired CEO, business development manager, astronomer, financial analyst, graphic designer, "
                     "research scientist, environmental consultant, health inspector").split(', '))

    for i, item in enumerate_resume(tqdm.tqdm(dataset), log_path):
        try:
            clss_output = gen.clssification(model, item['response'])
            assert clss_output['Occupation'] in label_set
        except Exception as e:
            print(e)
            print(f"\nSkipping {i}\n")
            print(clss_output)
            print('\n\n')
            clss_output = {'Occupation': "None"}

        result_str += f"{i}\t{clss_output['Occupation']}\n"

    model.print_usage()
    print(f"log path: {log_path}\n")
    with open(log_path.replace('.jsonl', '.txt'), 'w') as f:
        f.write(result_str)
    ss = ezsheets.Spreadsheet('1-uHO5DnE32WmImaucvHaVMvasO2mGh2eqWfWYksXljI')
    sheet = ss[0]
    update_idx = sheet.getColumn(1).index('') + 1
    update_row = sheet.getRow(update_idx)

    name2column = {'gpt-35-turbo-0301': 7, 'gpt-4': 1, 'gpt4-turbo-128k': 4}

    if model.name in name2column.keys():

        update_row[name2column[model.name]], update_row[name2column[model.name] + 1] = (model.prompt_tokens,
                                                                                        model.completion_tokens)
        update_row[0] = time.ctime()
        sheet.refresh()
        update_idx = sheet.getColumn(1).index('') + 1
        sheet.updateRow(update_idx, update_row)