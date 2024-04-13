import tqdm

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List


def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    no_utility: bool = False
) -> None:
    gen = generator_factory(language)
    model = model_factory(model_name)
    completion_tokens = 0
    prompt_tokens = 0

    for i, item in enumerate_resume(tqdm.tqdm(dataset[1:2]), log_path):
        # try:
        cur_pass = 0
        complete = False
        privacy_reflections = []
        utility_reflections = []
        rewritings = []
        detection_i = gen.detect(item["text"], model)
        people = item["people"]
        s_entity = detection_i["Sensitive entities"]
        completion_tokens += detection_i['usage']['completion_tokens']
        prompt_tokens += detection_i['usage']['prompt_tokens']

        while cur_pass < pass_at_k and not complete:
            privacy_reflections.append(f"pass: {cur_pass}")
            utility_reflections.append(f"pass: {cur_pass}")
            rewritings.append(f"pass: {cur_pass}")

            # first attempt
            cur_rewriting = gen.rewrite(item["text"], model, "simple", detection_result=detection_i['raw_response'],
                                        temperature=0.4)
            rewritings.append(cur_rewriting)
            completion_tokens += cur_rewriting['usage']['completion_tokens']
            prompt_tokens += cur_rewriting['usage']['prompt_tokens']

            privacy_evaluation = gen.privacy_reflex(model, rewritings[-1]['Generalized text'], people)
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
                utility_evaluation = gen.utility_reflex(item['text'], model, rewritings[-1]['Generalized text'], item['label'])
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
                item["rewritings"] = rewritings
                item["privacy_reflections"] = privacy_reflections
                item["utility_reflections"] = utility_reflections
                item["detection_result"] = detection_i
                item["complete"] = 'True'
                write_jsonl(log_path, [item], append=True)
                print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")
                print(f"log path: {log_path}\n")
                complete = True
                break

            cur_iter = 1
            complete = False
            while cur_iter <= max_iters:
                # prev_rewriting_feedback_str = ""
                # prev_rewriting_feedback_str += "Anonymized text:\n" + cur_rewriting['Generalized text'] + '\n'
                # if privacy_score == 'Yes':
                #     prev_rewriting_feedback_str += "Privacy feedback:\n" + privacy_feedback + '\n'
                # if utility_score == 'No':
                #     prev_rewriting_feedback_str += "Utility feedback:\n" + utility_feedback + '\n'

                # apply self-reflection in the next attempt
                if cur_iter == 1:
                    prev_rewriting = cur_rewriting['raw_text']
                else:
                    if "raw_text_utility" in cur_rewriting.keys():
                        prev_rewriting = cur_rewriting['raw_text_utility']
                    else:
                        prev_rewriting = cur_rewriting['raw_text_privacy']
                cur_rewriting = gen.rewrite(
                    input_text=item["text"],
                    model=model,
                    strategy="reflexion",
                    prev_rewriting=prev_rewriting,
                    reflection_privacy=privacy_feedback,
                    reflection_utility=utility_feedback,
                    privacy_score=privacy_score,
                    utility_score=utility_score,
                    detection_result=', '.join(s_entity),
                )
                rewritings.append(cur_rewriting)
                if "usage_privacy" in cur_rewriting.keys():
                    completion_tokens += cur_rewriting['usage_privacy']['completion_tokens']
                    prompt_tokens += cur_rewriting['usage_privacy']['prompt_tokens']
                if "usage_utility" in cur_rewriting.keys():
                    completion_tokens += cur_rewriting['usage_utility']['completion_tokens']
                    prompt_tokens += cur_rewriting['usage_utility']['prompt_tokens']


                # get self-reflection
                if 'Generalized text' in cur_rewriting.keys():
                    text_tobe_evaluated = cur_rewriting['Generalized text']
                else:
                    text_tobe_evaluated = cur_rewriting['Specialized text']
                privacy_evaluation = gen.privacy_reflex(model, text_tobe_evaluated, people)
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
                    utility_evaluation = gen.utility_reflex(item['text'], model, text_tobe_evaluated, item['label'])
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
        item["detection_result"] = detection_i
        write_jsonl(log_path, [item], append=True)
        print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")
        print(f"log path: {log_path}\n")
        # except:
        #     print(f"{i}-th example failed")