import tqdm

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
# from executors import executor_factory
from generators import generator_factory, model_factory
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import JSONLoader
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.storage import LocalFileStore
# from langchain.embeddings import CacheBackedEmbeddings

from typing import List


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    mem_len: int,
    p_threshold: int,
    is_leetcode: bool = False,
    no_utility: bool = False,
    rag_data_path: str = '',
    rag_num: int = 5,
    rag_embed_cache_dir: str = '',
) -> None:
    gen = generator_factory(language)
    model = model_factory(model_name)
    completion_tokens = 0
    prompt_tokens = 0

    # if rag_data_path.endswith('.csv'):
    #     loader = CSVLoader(file_path=rag_data_path, source_column="text")
    #     # "./programming_runs/benchmarks/Wiki_People/DBP_wiki_data.csv"
    # else:
    #     assert rag_data_path.endswith('.jsonl')
    #
    #     def metadata_func(record: dict, metadata: dict) -> dict:
    #         metadata['l1'] = record.get("l1")
    #         metadata['l2'] = record.get("l2")
    #         metadata['l3'] = record.get('l3')
    #         name = record.get('wiki_name')
    #         name = name.replace('_', ' ')
    #         if '(' in name:
    #             name = name.replace(name[name.index('('):name.index(')') + 1], '')
    #         metadata['wiki_name'] = name
    #         metadata['word_count'] = record.get('word_count')
    #         return metadata
    #
    #     loader = JSONLoader(
    #         file_path=rag_data_path,
    #         # './programming_runs/benchmarks/Wiki_People/All_data_for_retrieval.jsonl'
    #         jq_schema='.',
    #         content_key='text',
    #         metadata_func=metadata_func,
    #         json_lines=True)
    # data = loader.load()
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
    # # "all-MiniLM-L6-v2"
    # store = LocalFileStore(rag_embed_cache_dir)
    # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     embeddings, store, namespace="llm-embedder"
    # )
    # vectorstore = Chroma.from_documents(documents=data, embedding=cached_embedder)
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": rag_num})

    for i, item in enumerate_resume(tqdm.tqdm(dataset[109:]), log_path):
        # try:
        cur_pass = 0
        complete = False
        acc_reward = 0
        privacy_reflections = []
        utility_reflections = []
        rewritings = []
        # detection_i = gen.detect(item["text"], model)
        people = item["people"]
        # s_entity = detection_i["Sensitive entities"]
        # completion_tokens += detection_i['usage']['completion_tokens']
        # prompt_tokens += detection_i['usage']['prompt_tokens']

        while cur_pass < pass_at_k and not complete:
            privacy_reflections.append(f"pass: {cur_pass}")
            utility_reflections.append(f"pass: {cur_pass}")
            rewritings.append(f"pass: {cur_pass}")

            # first attempt
            cur_rewriting = gen.rewrite(item["text"], model, "simple", detection_result=None,
                                        temperature=0.0)
            rewritings.append(cur_rewriting)
            completion_tokens += cur_rewriting['usage']['completion_tokens']
            prompt_tokens += cur_rewriting['usage']['prompt_tokens']

            privacy_evaluation = gen.privacy_reflex(model, rewritings[-1]['Anonymized text'], people, p_threshold,
                                                    no_utility, None)
            privacy_score = privacy_evaluation["Confirmation"]
            privacy_feedback = privacy_evaluation["Advice"]
            privacy_reflections.append(privacy_evaluation)
            completion_tokens += privacy_evaluation['usage_1']['completion_tokens']
            prompt_tokens += privacy_evaluation['usage_1']['prompt_tokens']
            if "usage_2" in privacy_evaluation.keys():
                completion_tokens += privacy_evaluation['usage_2']['completion_tokens']
                prompt_tokens += privacy_evaluation['usage_2']['prompt_tokens']
            if "usage_3" in privacy_evaluation.keys():
                completion_tokens += privacy_evaluation['usage_3']['completion_tokens']
                prompt_tokens += privacy_evaluation['usage_3']['prompt_tokens']

            if not no_utility:
                utility_evaluation = gen.utility_reflex(item['text'], model, rewritings[-1]['Anonymized text'], item['label'], privacy_score)
                utility_score = utility_evaluation["Confirmation"]
                utility_feedback = utility_evaluation["Advice"]
                utility_reflections.append(utility_evaluation)
                completion_tokens += utility_evaluation['usage_1']['completion_tokens']
                prompt_tokens += utility_evaluation['usage_1']['prompt_tokens']
                if "usage_2" in utility_evaluation.keys():
                    completion_tokens += utility_evaluation['usage_2']['completion_tokens']
                    prompt_tokens += utility_evaluation['usage_2']['prompt_tokens']
                if "usage_3" in utility_evaluation.keys():
                    completion_tokens += utility_evaluation['usage_3']['completion_tokens']
                    prompt_tokens += utility_evaluation['usage_3']['prompt_tokens']
            else:
                utility_evaluation = {'Confirmation': 'Yes', 'Advice': ''}
                utility_score = utility_evaluation["Confirmation"]
                utility_feedback = utility_evaluation["Advice"]
                utility_reflections.append(utility_evaluation)

            # if solved, exit early
            if privacy_score == 'No' and utility_score == 'Yes':
                complete = True
                acc_reward = p_threshold + 1 + 100 if not no_utility else p_threshold + 1
                break

            cur_iter = 1
            complete = False
            acc_reward = 0
            while cur_iter <= max_iters:
                # prev_rewriting_feedback_str = ""
                # prev_rewriting_feedback_str += "Anonymized text:\n" + cur_rewriting['Anonymized text'] + '\n'
                # if privacy_score == 'Yes':
                #     prev_rewriting_feedback_str += "Privacy feedback:\n" + privacy_feedback + '\n'
                # if utility_score == 'No':
                #     prev_rewriting_feedback_str += "Utility feedback:\n" + utility_feedback + '\n'

                # apply self-reflection in the next attempt
                if no_utility:
                    prev_rewriting = cur_rewriting['raw_text']
                    acc_reward += int(privacy_evaluation['rank'])
                else:
                    prev_rewriting = ''
                    h_idx = 1
                    acc_reward = 0
                    if len(rewritings) > mem_len:
                        p_rer = rewritings[-mem_len:]
                        p_pr = privacy_reflections[-mem_len:]
                        p_ur = utility_reflections[-mem_len:]
                    else:
                        p_rer = rewritings
                        p_pr = privacy_reflections
                        p_ur = utility_reflections
                    for rewriting, p_r, u_r in zip(p_rer, p_pr, p_ur):
                        if type(rewriting) is str:
                            continue
                        prev_rewriting += f"Edition: {h_idx}\nEditing results; {rewriting['Anonymized text']}\nPrivacy score: {p_r['rank']}\nUtility score: {u_r['Confidence Score']}\n"
                        if p_r['Confirmation'] == 'Yes':
                            prev_rewriting += f"Reward: {p_r['rank']}\n\n"
                            acc_reward += int(p_r['rank'])
                        else:
                            prev_rewriting += f"Reward: {u_r['Confidence Score']}\n\n"
                            acc_reward += int(u_r['Confidence Score'])
                        h_idx = h_idx + 1
                cur_rewriting = gen.rewrite(
                    input_text=item["text"],
                    model=model,
                    strategy="reflexion",
                    prev_rewriting=prev_rewriting,
                    reflection_privacy=privacy_feedback,
                    reflection_utility=utility_feedback,
                    privacy_score=privacy_score,
                    utility_score=utility_score,
                    detection_result=None,
                    p_threshold=p_threshold,
                    no_utility=no_utility
                )
                rewritings.append(cur_rewriting)
                completion_tokens += cur_rewriting['usage']['completion_tokens']
                prompt_tokens += cur_rewriting['usage']['prompt_tokens']
                # if "usage_privacy" in cur_rewriting.keys():
                #     completion_tokens += cur_rewriting['usage_privacy']['completion_tokens']
                #     prompt_tokens += cur_rewriting['usage_privacy']['prompt_tokens']
                # if "usage_utility" in cur_rewriting.keys():
                #     completion_tokens += cur_rewriting['usage_utility']['completion_tokens']
                #     prompt_tokens += cur_rewriting['usage_utility']['prompt_tokens']


                # get self-reflection
                text_tobe_evaluated = cur_rewriting['Anonymized text']
                # if 'Anonymized text' in cur_rewriting.keys():
                #     text_tobe_evaluated = cur_rewriting['Anonymized text']
                # else:
                #     text_tobe_evaluated = cur_rewriting['Specialized text']
                privacy_evaluation = gen.privacy_reflex(model, text_tobe_evaluated, people, p_threshold, no_utility,
                                                        None)
                privacy_score = privacy_evaluation["Confirmation"]
                privacy_feedback = privacy_evaluation["Advice"]
                privacy_reflections.append(privacy_evaluation)
                completion_tokens += privacy_evaluation['usage_1']['completion_tokens']
                prompt_tokens += privacy_evaluation['usage_1']['prompt_tokens']
                if "usage_2" in privacy_evaluation.keys():
                    completion_tokens += privacy_evaluation['usage_2']['completion_tokens']
                    prompt_tokens += privacy_evaluation['usage_2']['prompt_tokens']
                if "usage_3" in privacy_evaluation.keys():
                    completion_tokens += privacy_evaluation['usage_3']['completion_tokens']
                    prompt_tokens += privacy_evaluation['usage_3']['prompt_tokens']

                if not no_utility:
                    utility_evaluation = gen.utility_reflex(item['text'], model, text_tobe_evaluated, item['label'],
                                                            privacy_score)
                    utility_score = utility_evaluation["Confirmation"]
                    utility_feedback = utility_evaluation["Advice"]
                    utility_reflections.append(utility_evaluation)
                    completion_tokens += utility_evaluation['usage_1']['completion_tokens']
                    prompt_tokens += utility_evaluation['usage_1']['prompt_tokens']
                    if "usage_2" in utility_evaluation.keys():
                        completion_tokens += utility_evaluation['usage_2']['completion_tokens']
                        prompt_tokens += utility_evaluation['usage_2']['prompt_tokens']
                    if "usage_3" in utility_evaluation.keys():
                        completion_tokens += utility_evaluation['usage_3']['completion_tokens']
                        prompt_tokens += utility_evaluation['usage_3']['prompt_tokens']
                else:
                    utility_evaluation = {'Confirmation': 'Yes', 'Advice': ''}
                    utility_score = utility_evaluation["Confirmation"]
                    utility_feedback = utility_evaluation["Advice"]
                    utility_reflections.append(utility_evaluation)

                # if solved, check if it passes the real tests, exit early
                if privacy_score == 'No' and utility_score == 'Yes':
                    complete = True
                    break

                cur_iter += 1
            cur_pass += 1

        item["rewritings"] = rewritings
        item["privacy_reflections"] = privacy_reflections
        item["utility_reflections"] = utility_reflections
        item["complete"] = 'False' if not complete else 'True'
        item["acc_reward"] = acc_reward
        # item["detection_result"] = detection_i
        write_jsonl(log_path, [item], append=True)
        print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")
        print(f"log path: {log_path}\n")
        # except:
        #     print(f"{i}-th example failed")
