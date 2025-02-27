import tqdm

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from generators import generator_factory, model_factory
import time
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders import JSONLoader
# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.storage import LocalFileStore
# from langchain.embeddings import CacheBackedEmbeddings

from typing import List


def run_reflexion(
    dataset: List[dict],
    pe_model_name: str,
    ue_model_name: str,
    act_model_name: str,
    parser_model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    mem_len: int,
    p_threshold: int,
    is_leetcode: bool = False,
    no_utility: bool = False,
    cot: bool = False,
    rag_data_path: str = '',
    rag_num: int = 5,
    rag_embed_cache_dir: str = '',
) -> None:
    gen = generator_factory(language)
    pe_model = model_factory(pe_model_name)
    ue_model = model_factory(ue_model_name)
    act_model = model_factory(act_model_name)
    parser_model = model_factory(parser_model_name)

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

    # Track total token usage for models
    model_usage = {
        'gpt-35-turbo-0301': {'prompt_tokens': 0, 'completion_tokens': 0},
        'gpt-4': {'prompt_tokens': 0, 'completion_tokens': 0},
        'gpt4-turbo-128k': {'prompt_tokens': 0, 'completion_tokens': 0},
        'gpt-4-turbo-preview': {'prompt_tokens': 0, 'completion_tokens': 0}
    }

    for i, item in enumerate_resume(tqdm.tqdm(dataset[107 + 39 + 15:]), log_path):
        # try:
        cur_pass = 0
        complete = False
        acc_reward = 0
        privacy_reflections = []
        utility_reflections = []
        rewritings = []
        people = item["people"] if language == 'wiki' else {item['feature']: item['personality'][item['feature']]}

        if item['label'] == 'Medician':
            item['label'] = 'Physician'

        if cot:
            detection_i = gen.detect(item["text"] if language == 'wiki' else item['response'].replace('\n', ''), act_model)

        while cur_pass < pass_at_k and not complete:
            privacy_reflections.append(f"pass: {cur_pass}")
            utility_reflections.append(f"pass: {cur_pass}")
            rewritings.append(f"pass: {cur_pass}")

            # first attempt
            cur_rewriting = gen.rewrite(item["text"] if language == 'wiki' else item['response'].replace('\n', ''), item['label'], people, act_model, parser_model, "simple", cot=cot,
                                        detection_result=detection_i['raw_response'] if cot else None,
                                        temperature=0.0)
            rewritings.append(cur_rewriting)

            privacy_evaluation = gen.privacy_reflex(pe_model, rewritings[-1]['Anonymized text'], people, p_threshold,
                                                    no_utility, None)
            privacy_score = privacy_evaluation["Confirmation"]
            privacy_feedback = privacy_evaluation["Advice"]
            privacy_reflections.append(privacy_evaluation)

            if not no_utility:
                utility_evaluation = gen.utility_reflex(item['text'] if language == 'wiki' else item['response'].replace('\n', ''),
                                                        ue_model, rewritings[-1]['Anonymized text'],
                                                        item['label'] if language == 'wiki' else item['personality']['occupation'],
                                                        privacy_score)
                utility_score = utility_evaluation["Confirmation"]
                utility_feedback = utility_evaluation["Advice"]
                utility_reflections.append(utility_evaluation)
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
                # apply self-reflection in the next attempt
                if no_utility:
                    prev_rewriting = cur_rewriting['raw_response']
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
                    input_text=item["text"] if language == 'wiki' else item['response'].replace('\n', ''),
                    label=item['label'],
                    people=people,
                    act_model=act_model,
                    parser_model=parser_model,
                    cot=cot,
                    strategy="reflexion",
                    prev_rewriting=prev_rewriting,
                    reflection_privacy=privacy_feedback,
                    reflection_utility=utility_feedback,
                    privacy_score=privacy_score,
                    utility_score=utility_score,
                    detection_result=None,
                    p_threshold=p_threshold if language == 'wiki' else 7,
                    no_utility=no_utility
                )
                rewritings.append(cur_rewriting)

                # get self-reflection
                text_tobe_evaluated = cur_rewriting['Anonymized text']
                privacy_evaluation = gen.privacy_reflex(pe_model, text_tobe_evaluated, people, p_threshold, no_utility,
                                                        None)
                privacy_score = privacy_evaluation["Confirmation"]
                privacy_feedback = privacy_evaluation["Advice"]
                privacy_reflections.append(privacy_evaluation)

                if not no_utility:
                    utility_evaluation = gen.utility_reflex(item['text'] if language == 'wiki' else item['response'].replace('\n', ''),
                                                            ue_model, text_tobe_evaluated,
                                                            item['label'] if language == 'wiki' else item['personality']['occupation'],
                                                            privacy_score)
                    utility_score = utility_evaluation["Confirmation"]
                    utility_feedback = utility_evaluation["Advice"]
                    utility_reflections.append(utility_evaluation)
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
        if cot:
            item["detection_result"] = detection_i
        write_jsonl(log_path, [item], append=True)
        
        # Print current usage statistics
        print(f"\n=== Token Usage for Example {i} ===")
        # Still use the original model print methods for compatibility
        act_model.print_usage()
        pe_model.print_usage()
        ue_model.print_usage()
        parser_model.print_usage()
        
        # Print per-example detailed stats
        print("\nDetailed Token Usage:")
        model_list = [act_model, pe_model, ue_model, parser_model]
        for model in model_list:
            if model.name in model_usage:
                print(f"{model.name:<20} - Prompt: {model.prompt_tokens} tokens, Completion: {model.completion_tokens} tokens")
                model_usage[model.name]['prompt_tokens'] += model.prompt_tokens
                model_usage[model.name]['completion_tokens'] += model.completion_tokens
        
        print(f"log path: {log_path}\n")

        # except Exception as e:
        #     act_model.print_usage()
        #     pe_model.print_usage()
        #     ue_model.print_usage()
        #     parser_model.print_usage()
        #     write_jsonl(log_path, [{'status': 'Failed'}], append=True)
        #     print(f"{e}\n{i}-th example failed")
    
    # Print verbose summary of accumulated token usage at the end
    print(f"\n=== FINAL TOKEN USAGE SUMMARY ({time.ctime()}) ===")
    print(f"{'Model Name':<20} {'Prompt Tokens':<15} {'Completion Tokens':<15} {'Total Tokens':<15}")
    print("-" * 65)
    total_prompt = 0
    total_completion = 0
    for model_name, tokens in model_usage.items():
        if tokens['prompt_tokens'] > 0 or tokens['completion_tokens'] > 0:
            total = tokens['prompt_tokens'] + tokens['completion_tokens']
            print(f"{model_name:<20} {tokens['prompt_tokens']:<15} {tokens['completion_tokens']:<15} {total:<15}")
            total_prompt += tokens['prompt_tokens']
            total_completion += tokens['completion_tokens']
    
    print("-" * 65)
    print(f"{'TOTAL':<20} {total_prompt:<15} {total_completion:<15} {total_prompt + total_completion:<15}")
    print("=" * 65)
