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
    is_leetcode: bool = False
) -> None:
    gen = generator_factory(language)
    model = model_factory(model_name)
    completion_tokens = 0
    prompt_tokens = 0

    for i, item in enumerate_resume(tqdm.tqdm(dataset), log_path):
        privacy_reflections = []
        utility_reflections = []
        rewritings = []
        detection_i = gen.detect(item["text"], model)
        people = detection_i["People"] if len(detection_i["People"].split(', ')) == 0 \
            else detection_i["People"].split(', ')[0]
        s_entity = detection_i["Sensitive entities"]
        completion_tokens += detection_i['usage']['completion_tokens']
        prompt_tokens += detection_i['usage']['prompt_tokens']

        # first attempt
        cur_rewriting = gen.rewrite(item["text"], model, "simple", detection_result=detection_i['raw_response'])
        rewritings.append(cur_rewriting['Generalized text'])
        completion_tokens += cur_rewriting['usage']['completion_tokens']
        prompt_tokens += cur_rewriting['usage']['prompt_tokens']

        privacy_evaluation = gen.privacy_reflex(model, rewritings[-1], people)
        privacy_score = privacy_evaluation["Confirmation"]
        privacy_feedback = privacy_evaluation["Advice"]
        privacy_reflections.append(privacy_evaluation)
        completion_tokens += privacy_evaluation['usage_1']['completion_tokens']
        prompt_tokens += privacy_evaluation['usage_1']['prompt_tokens']
        if "usage_2" in privacy_evaluation.keys():
            completion_tokens += privacy_evaluation['usage_2']['completion_tokens']
            prompt_tokens += privacy_evaluation['usage_2']['prompt_tokens']

        utility_evaluation = gen.utility_reflex(item['text'], model, rewritings[-1], item['label'])
        utility_score = utility_evaluation["Confirmation"]
        utility_feedback = utility_evaluation["Advice"]
        utility_reflections.append(utility_evaluation)
        completion_tokens += utility_evaluation['usage_1']['completion_tokens']
        prompt_tokens += utility_evaluation['usage_1']['prompt_tokens']
        if "usage_2" in utility_evaluation.keys():
            completion_tokens += utility_evaluation['usage_2']['completion_tokens']
            prompt_tokens += utility_evaluation['usage_2']['prompt_tokens']

        # if solved, exit early
        if privacy_score == 'No' and utility_score == 'Yes':
            item["rewritings"] = rewritings
            item["privacy_reflections"] = privacy_reflections
            item["utility_reflections"] = utility_reflections
            item["detection_result"] = detection_i
            item["complete"] = 'True'
            write_jsonl(log_path, [item], append=True)
            print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")
            continue

        cur_iter = 1
        while cur_iter <= max_iters:
            prev_rewriting_feedback_str = ""
            prev_rewriting_feedback_str += "Anonymized text:\n" + cur_rewriting['Generalized text'] + '\n'
            if privacy_score == 'Yes':
                prev_rewriting_feedback_str += "Privacy feedback:\n" + privacy_feedback + '\n'
            if utility_score == 'No':
                prev_rewriting_feedback_str += "Utility feedback:\n" + utility_feedback + '\n'

            # apply self-reflection in the next attempt
            cur_rewriting = gen.rewrite(
                input_text=item["text"],
                model=model,
                strategy="reflexion",
                prev_rewriting=prev_rewriting_feedback_str,
                reflection_privacy=privacy_feedback,
                reflection_utility=utility_feedback,
                privacy_score=privacy_score,
                utility_score=utility_score,
                detection_result=', '.join(s_entity),
            )
            rewritings.append(cur_rewriting['Generalized text'])
            completion_tokens += cur_rewriting['usage']['completion_tokens']
            prompt_tokens += cur_rewriting['usage']['prompt_tokens']


            # get self-reflection
            privacy_evaluation = gen.privacy_reflex(model, rewritings[-1], people)
            privacy_score = privacy_evaluation["Confirmation"]
            privacy_feedback = privacy_evaluation["Advice"]
            privacy_reflections.append(privacy_evaluation)
            completion_tokens += privacy_evaluation['usage_1']['completion_tokens']
            prompt_tokens += privacy_evaluation['usage_1']['prompt_tokens']
            if "usage_2" in privacy_evaluation.keys():
                completion_tokens += privacy_evaluation['usage_2']['completion_tokens']
                prompt_tokens += privacy_evaluation['usage_2']['prompt_tokens']

            utility_evaluation = gen.utility_reflex(item['text'], model, rewritings[-1], item['label'])
            utility_score = utility_evaluation["Confirmation"]
            utility_feedback = utility_evaluation["Advice"]
            utility_reflections.append(utility_evaluation)
            completion_tokens += utility_evaluation['usage_1']['completion_tokens']
            prompt_tokens += utility_evaluation['usage_1']['prompt_tokens']
            if "usage_2" in utility_evaluation.keys():
                completion_tokens += utility_evaluation['usage_2']['completion_tokens']
                prompt_tokens += utility_evaluation['usage_2']['prompt_tokens']
            # if solved, check if it passes the real tests, exit early
            if privacy_score == 'No' and utility_score == 'Yes':
                break
                item["complete"] = 'True'

            cur_iter += 1

        item["rewritings"] = rewritings
        item["privacy_reflections"] = privacy_reflections
        item["utility_reflections"] = utility_reflections
        item["complete"] = 'False'
        item["detection_result"] = detection_i
        write_jsonl(log_path, [item], append=True)
        print(f"Prompt tokens number: {prompt_tokens}, Completion tokens number: {completion_tokens}. \n")