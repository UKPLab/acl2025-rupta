from programming_runs.generators.model import ModelBase, Message
import random
import json
from sentence_transformers import SentenceTransformer

from typing import Union, List, Optional, Callable


def generic_generate_func_impl(
    func_sig: str,
    model: ModelBase,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    reflexion_chat_instruction: str,
    reflexion_few_shot: str,
    simple_chat_instruction: str,
    reflexion_completion_instruction: str,
    simple_completion_instruction: str,
    code_block_instruction: str,
    parse_code_block: Callable[[str], str],
    add_code_block: Callable[[str], str]
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "reflexion":
            message = f"{reflexion_few_shot}\n[previous impl]:\n{add_code_block(prev_func_impl)}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            prompt = f"{reflexion_chat_instruction}\n{code_block_instruction}"
            # func_bodies is a really bad name, as it can also be just 1 string
            print_messages(prompt, message)
            messages = [
                Message(
                    role="system",
                    content=prompt,
                ),
                Message(
                    role="user", # TODO: check this
                    content=reflexion_few_shot,
                ),
                Message(
                    role="assistant",
                    content=add_code_block(prev_func_impl),
                ),
                Message(
                    role="user",
                    content=f"[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:",
                ),
                Message(
                    role="assistant",
                    content=self_reflection,
                ),
                Message(
                    role="user",
                    content=f"[improved impl]:\n{func_sig}",
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
        else:
            system_prompt = f"{simple_chat_instruction}\n{code_block_instruction}"
            print_messages(system_prompt, func_sig)
            messages = [
                Message(
                    role="system",
                    content=f"{simple_chat_instruction}\n{code_block_instruction}",
                ),
                Message(
                    role="user",
                    content=func_sig,
                ),
            ]
            func_bodies = model.generate_chat(messages=messages, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{reflexion_completion_instruction}\n{add_code_block(prev_func_impl)}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{simple_completion_instruction}\n{func_sig}\n{code_block_instruction}"
            func_bodies = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        func_body_str = parse_code_block(func_bodies)
        print_generated_func_body(func_body_str)
        return func_body_str

    else:
        func_bodies = [parse_code_block(func_body) for func_body in func_bodies]
        print_generated_func_body("\n\n".join(func_bodies))
        return func_bodies


def generic_generate_internal_tests(
        func_sig: str,
        model: ModelBase,
        max_num_tests: int,
        test_generation_few_shot: str,
        test_generation_chat_instruction: str,
        test_generation_completion_instruction: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """Generates tests for a function."""
    if model.is_chat:
        if is_react:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[think]:"
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
            print(f'React test generation output: {output}')
        else:
            messages = [
                Message(
                    role="system",
                    content=test_generation_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f"{test_generation_few_shot}\n\n[func signature]:\n{func_sig}\n\n[unit tests]:",
                )
            ]
            output = model.generate_chat(messages=messages, max_tokens=1024)
    else:
        prompt = f'{test_generation_completion_instruction}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = model.generate(prompt, max_tokens=1024)
    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    return sample_n_random(valid_tests, max_num_tests)


def generic_detection(
        input_text: str,
        model: ModelBase,
        whole_task_instruction: str,
        general_task_instruction: str,
        detection_chat_instruction: str,
        detection_completion_instruction: str,
):
    """Generates tests for a function."""
    if model.is_chat:
        messages = [
            Message(
                role="system",
                content=general_task_instruction,
            ),
            Message(
                role="user",
                content=f"{whole_task_instruction}\n\n{detection_chat_instruction}\nThe person description text is here:\n{input_text}"
            )
        ]
        output, usage, finish_reason = model.generate_chat(messages=messages)
    else:
        prompt = f'{whole_task_instruction}\n{detection_completion_instruction}\n\n[description text]:\n{input_text}'
        output, usage, finish_reason = model.generate(prompt)

    output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
    output_dict['raw_response'] = output
    output_dict['usage'] = {}
    output_dict['usage']['prompt_tokens'] = usage.prompt_tokens
    output_dict['usage']['completion_tokens'] = usage.completion_tokens
    output_dict['finish_reason'] = finish_reason
    return output_dict

def generic_rewriting(
    input_text: str,
    model: ModelBase,
    strategy: str,
    prev_rewriting,
    reflection_privacy,
    reflection_utility,
    privacy_score,
    utility_score,
    detection_result,
    num_comps,
    temperature,
    no_utility,
    whole_task_instruction: str,
    general_system_instruction: str,
    detection_result_prefix: str,
    refelection_prev_re_instruction: str,
    reflection_privacy_instruction: str,
    refelection_utility_instruction: str,
    simple_rewriting_instruction: str,
    reflection_privacy_rewriting_instruction: str,
    reflection_utility_rewriting_instruction: str,
    reinforcement_learning_instruction: str
):
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_rewriting is None or reflection_privacy is None or reflection_utility is None):
        raise ValueError(
            f"Invalid arguments: "
            f"given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model.is_chat:
        if strategy == "simple":
            messages = [
                Message(
                    role="system",
                    content=general_system_instruction,
                ),
                Message(
                    role="user", # TODO: check this
                    content=f"{whole_task_instruction}\n\n{detection_result_prefix}\nThe person description text is here:\n{input_text}",
                ),
                Message(
                    role="assistant",  # TODO: check this
                    content=f"{detection_result}",
                ),
                Message(
                    role="user",
                    content=simple_rewriting_instruction,
                )
            ]
            output, usage, finish_reason = model.generate_chat(messages=messages, num_comps=num_comps,
                                                               temperature=temperature)
            output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
            output_dict['raw_text'] = output
            output_dict['usage'] = {}
            output_dict['usage']['prompt_tokens'] = usage.prompt_tokens
            output_dict['usage']['completion_tokens'] = usage.completion_tokens
            output_dict['finish_reason'] = finish_reason
        else:
            if not no_utility:
                if privacy_score == 'Yes':
                    # prev_rewriting += f"Suggestion: These entities {reflection_privacy} could be further generalized to improve the privacy score."
                    prev_rewriting += f"Suggestion: Entities that can be used to infer the person identity could be further generalized to improve the privacy score."
                else:
                    # prev_rewriting += f"Suggestion: These entities {reflection_utility} could be replaced with the original specific entities in the original biography to improve the utility score. You should also avoid specifying entities that could decrease the privacy score."
                    prev_rewriting += f"Suggestion: Entities that are about the description of the occupation but can not be used to infer the person identity could be specified to improve the utility score. "
                messages = [
                    Message(
                        role="system",
                        content=general_system_instruction,
                    ),
                    Message(
                        role='user',
                        content=reinforcement_learning_instruction + f"\n\nThe original biography is {input_text}\n\n{prev_rewriting}"
                    )
                ]
                output, usage, finish_reason = model.generate_chat(messages=messages, num_comps=num_comps,
                                                                         temperature=temperature)
                output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
                output_dict['raw_text'] = output
                output_dict['usage'] = {}
                output_dict['usage']['prompt_tokens'] = usage.prompt_tokens
                output_dict['usage']['completion_tokens'] = usage.completion_tokens
                output_dict['finish_reason'] = finish_reason
            else:
                messages = [
                    Message(
                        role="system",
                        content=general_system_instruction,
                    ),
                    Message(
                        role="user",  # TODO: check this
                        content=f"{whole_task_instruction}\n\n{detection_result_prefix}\nThe person description text is here:\n{input_text}",
                    ),
                    Message(
                        role="assistant",  # TODO: check this
                        content=f"{detection_result}",
                    ),
                    Message(
                        role="user",
                        content=f"{simple_rewriting_instruction}",
                    ),
                    Message(
                        role="assistant",
                        content=prev_rewriting
                    )
                ]
                if privacy_score == 'Yes':
                    messages.append(
                        Message(
                            role="user",
                            content=f"{reflection_privacy_rewriting_instruction}\nThe entity list is here:\n{reflection_privacy}",
                        )
                    )
                    output, usage, finish_reason = model.generate_chat(messages=messages, num_comps=num_comps,
                                                                             temperature=temperature)
                    output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
                    output_dict['raw_text'] = output
                    output_dict['usage'] = {}
                    output_dict['usage']['prompt_tokens'] = usage.prompt_tokens
                    output_dict['usage']['completion_tokens'] = usage.completion_tokens
                    output_dict['finish_reason'] = finish_reason
    else:
        if strategy == "simple":
            prompt = (f"{whole_task_instruction}\n[description text]:\n{input_text}\n{detection_result_prefix}"
                      f"\n{detection_result}\n{simple_rewriting_instruction}")
            output, usage, finish_reason = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = (f"{whole_task_instruction}\n[description text]:\n{input_text}\n{detection_result_prefix}"
                      f"\n{detection_result}\n{simple_rewriting_instruction}"
                      f"{refelection_prev_re_instruction}\n{prev_rewriting}\n{reflection_privacy_instruction}"
                      f"\n{reflection_privacy}\n{refelection_utility_instruction}"
                      f"\n{reflection_utility}\n{reflection_rewriting_instruction}"
                      )
            output, usage, finish_reason = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    return output_dict

def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: ModelBase,
        self_reflection_chat_instruction: str,
        self_reflection_completion_instruction: str,
        add_code_block: Callable[[str], str],
        self_reflection_few_shot: Optional[str] = None,
) -> str:
    if model.is_chat:
        if self_reflection_few_shot is not None:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'{self_reflection_few_shot}\n\n[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
            print(f'Self reflection output: {reflection}')
        else:
            messages = [
                Message(
                    role="system",
                    content=self_reflection_chat_instruction,
                ),
                Message(
                    role="user",
                    content=f'[function impl]:\n{add_code_block(func)}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:',
                )
            ]
            reflection = model.generate_chat(messages=messages)
    else:
        reflection = model.generate(
            f'{self_reflection_completion_instruction}\n{add_code_block(func)}\n\n{feedback}\n\nExplanation:')
    return reflection  # type: ignore

def generic_privacy_reflection(
        model: ModelBase,
        curr_rewriting: str,
        people: str,
        no_utility: bool,
        general_system_instruction: str,
        privacy_reflection_chat_instruction_1: str,
        privacy_reflection_completion_instruction_1: str,
        privacy_reflection_chat_instruction_2: str,
        privacy_reflection_completion_instruction_2: str
):
    if model.is_chat:
        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{privacy_reflection_chat_instruction_1}\nThe biography text is here:\n{curr_rewriting}',
            )
        ]
        output, usage_1, finish_reason_1 = model.generate_chat(messages=messages)
        candidate = json.loads(output.replace('\n', ''))["Candidates"].split(', ')
        emb_model = SentenceTransformer("all-mpnet-base-v2")
        candidate_emb = emb_model.encode(candidate)
        people_emb = emb_model.encode(people)
        sim_score = candidate_emb.dot(people_emb)
        if True in (sim_score > 0.75):
            if no_utility:
                messages.append(
                    Message(
                        role="assistant",
                        content=output
                    )
                )
                messages.append(
                    Message(
                        role="user",
                        content=f"The person described by the text is {people}. {privacy_reflection_completion_instruction_2}"
                    )
                )
                output, usage_2, finish_reason_2 = model.generate_chat(messages=messages)
                try:
                    output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
                except json.decoder.JSONDecodeError:
                    messages.append(Message(role="assistant", content=output))
                    messages.append(Message(role="user",
                                            content="Your response is not exactly in the JSON structure that I provide, "
                                                    "put your response in the JSON structure."))
                    output, usage_3, finish_reason_3 = model.generate_chat(messages=messages)
                    output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
                    output_dict['usage_3'] = {}
                    output_dict['usage_3']['prompt_tokens'] = usage_3.prompt_tokens
                    output_dict['usage_3']['completion_tokens'] = usage_3.completion_tokens
                output_dict['usage_2'] = {}
                output_dict['usage_1'] = {}
                output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
                output_dict['usage_2']['prompt_tokens'] = usage_2.prompt_tokens
                output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
                output_dict['usage_2']['completion_tokens'] = usage_2.completion_tokens
                output_dict['finish_reason_1'] = finish_reason_1
                output_dict['finish_reason_2'] = finish_reason_2
                output_dict['candidates'] = candidate
                output_dict['rank'] = int(sim_score.argmax()) + 1
            else:
                output_dict = {}
                output_dict["Confirmation"] = "Yes"
                output_dict["Advice"] = ""
                output_dict['usage_1'] = {}
                output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
                output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
                output_dict['candidates'] = candidate
                output_dict['rank'] = int(sim_score.argmax()) + 1
        else:
            output_dict = {}
            output_dict["Confirmation"] = "No"
            output_dict["Advice"] = ""
            output_dict['usage_1'] = {}
            output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
            output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
            output_dict['candidates'] = candidate
            output_dict['rank'] = sim_score.size + 1
    else:
        output = model.generate(
            f'{privacy_reflection_completion_instruction_1}\n[Description text]:\n{curr_rewriting}\n[Person name]:\n{people}')

    return output_dict  # type: ignore

def generic_utility_reflection(
        input_text: str,
        model: ModelBase,
        label: str,
        privacy_score: str,
        curr_rewriting: str,
        general_system_instruction: str,
        utility_reflection_chat_instruction_1: str,
        utility_reflection_completion_instruction_1: str,
        utility_reflection_chat_instruction_2: str,
        utility_reflection_completion_instruction_2: str
):
    if model.is_chat:
        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{utility_reflection_chat_instruction_1}\nThe anonymized biography is here:\n{curr_rewriting}',
            )
        ]
        output, usage_1, finish_reason_1 = model.generate_chat(messages=messages)
        temp = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
        occupation = temp['Occupation']
        confidence_score = int(temp["Confidence Score"])
        if occupation != label or (occupation == label and confidence_score < 100):
            # messages.append(
            #     Message(
            #         role="assistant",
            #         content=output
            #     )
            # )
            # if occupation != label:
            #     messages.append(
            #         Message(
            #             role="user",
            #             content=f'The true occupation of the person is {label}. Your classification is wrong.'
            #                     f' {utility_reflection_completion_instruction_2}'
            #                     # f'The original biography is here:\n{input_text}'
            #         )
            #     )
            # else:
            #     messages.append(
            #         Message(
            #             role="user",
            #             content=f'The true occupation of the person is {label}. '
            #                     f'Your confidence of making this classification is no high enough. '
            #                     f'{utility_reflection_completion_instruction_2}'
            #                     # f'The original biography is here:\n{input_text}'
            #         )
            #     )
            # output, usage_2, finish_reason_2 = model.generate_chat(messages=messages)
            # try:
            #     output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
            # except json.decoder.JSONDecodeError:
            #     messages.append(Message(role="assistant", content=output))
            #     messages.append(Message(role="user",
            #                             content="Your response is not exactly in the JSON structure that I provide, "
            #                                     "put your response in the JSON structure."))
            #     output, usage_3, finish_reason_3 = model.generate_chat(messages=messages)
            #     output_dict = json.loads(output.replace('\n', '').replace("```json", '').replace("```", ""))
            #     output_dict['usage_3'] = {}
            #     output_dict['usage_3']['prompt_tokens'] = usage_3.prompt_tokens
            #     output_dict['usage_3']['completion_tokens'] = usage_3.completion_tokens
            # output_dict['usage_1'] = {}
            # output_dict['usage_2'] = {}
            # output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
            # output_dict['usage_2']['prompt_tokens'] = usage_2.prompt_tokens
            # output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
            # output_dict['usage_2']['completion_tokens'] = usage_2.completion_tokens
            # output_dict['finish_reason_1'] = finish_reason_1
            # output_dict['finish_reason_2'] = finish_reason_2
            output_dict = {}
            output_dict["Confirmation"] = "No"
            output_dict["Advice"] = ""
            output_dict['usage_1'] = {}
            output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
            output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
            output_dict['occupation'] = occupation
            output_dict['Confidence Score'] = confidence_score
        else:
            output_dict = {}
            output_dict["Confirmation"] = "Yes"
            output_dict["Advice"] = ""
            output_dict['usage_1'] = {}
            output_dict['usage_1']['prompt_tokens'] = usage_1.prompt_tokens
            output_dict['usage_1']['completion_tokens'] = usage_1.completion_tokens
            output_dict['occupation'] = occupation
            output_dict['Confidence Score'] = confidence_score
    else:
        output = model.generate(
            f'{utility_reflection_completion_instruction_1}\n[Original text]:\n{input_text}\n[Anonymized text]:\n{curr_rewriting}\n[Classification label]{label}')

    return output_dict  # type: ignore


def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)

def print_messages(system_message_text: str, user_message_text: str) -> None:
    print(f"""----------------------- SYSTEM MESSAGE -----------------------)
{system_message_text}
----------------------------------------------
----------------------- USER MESSAGE -----------------------
{user_message_text}
----------------------------------------------
""", flush=True)

def print_generated_func_body(func_body_str: str) -> None:
    print(f"""--------------------- GENERATED FUNC BODY ---------------------
{func_body_str}
------------------------------------------""")
