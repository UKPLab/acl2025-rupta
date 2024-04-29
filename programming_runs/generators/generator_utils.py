import copy

from programming_runs.generators.model import ModelBase, Message
import random
import json
from sentence_transformers import SentenceTransformer
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, RetryOutputParser, OutputFixingParser


from typing import Union, List, Optional, Callable


def parse_fixing(general_system_instruction, format_instructions, output_parser, output_dict, parser_model, key_list):
    fixing_messages = [
        Message(
            role="system",
            content=general_system_instruction,
        ),
        Message(
            role="user",
            content=f"{format_instructions}\n\nBut I got '{output_dict['raw_response']}', help me to fix"
                    f" it to fit the given json format",
        )
    ]
    fixing_dict = parser_model.generate_chat(messages=fixing_messages,
                                             format_instructions=format_instructions,
                                             parser=output_parser)
    for k in key_list:
        output_dict[k] = copy.deepcopy(fixing_dict[k])

    return output_dict


def generic_detection(
        input_text: str,
        model: ModelBase,
        whole_task_instruction: str,
        general_task_instruction: str,
        detection_chat_instruction: str,
        detection_completion_instruction: str,
):
    if model.is_chat:
        response_schemas = [
            ResponseSchema(name="People", description="name of the detected people separated by ', '"),
            ResponseSchema(
                name="Sensitive entities",
                description="the list of detected sensitive entities where every two entities are separated by ', '",
            ),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        messages = [
            Message(
                role="system",
                content=general_task_instruction,
            ),
            Message(
                role="user",
                content=f"{whole_task_instruction}\n\n{detection_chat_instruction.format(format_instructions_1=format_instructions)}\nThe person description text is here:\n{input_text}"
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
    else:
        prompt = f'{whole_task_instruction}\n{detection_completion_instruction}\n\n[description text]:\n{input_text}'
        output_dict, usage, finish_reason = model.generate(prompt)

    return output_dict


def generic_rewriting(
    input_text: str,
    model: ModelBase,
    parser_model: ModelBase,
    strategy: str,
    cot: bool,
    prev_rewriting,
    reflection_privacy,
    reflection_utility,
    privacy_score,
    utility_score,
    detection_result,
    num_comps,
    temperature,
    no_utility,
    p_threshold,
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
            response_schemas_2 = [
                ResponseSchema(name="Anonymized text", description="your anonymization result")
            ]
            output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas_2)
            format_instructions_2 = output_parser_2.get_format_instructions()

            messages = [
                Message(
                    role="system",
                    content=general_system_instruction,
                )
            ]
            if cot:
                response_schemas_1 = [
                    ResponseSchema(name="People", description="name of the detected people separated by ', '"),
                    ResponseSchema(
                        name="Sensitive entities",
                        description="the list of detected sensitive entities where every two entities are separated by ', '",
                    ),
                ]
                output_parser_1 = StructuredOutputParser.from_response_schemas(response_schemas_1)
                format_instructions_1 = output_parser_1.get_format_instructions()
                messages.extend(
                    [
                        Message(
                            role="user",
                            content=f"{whole_task_instruction}\n\n{detection_result_prefix.format(format_instructions_1=format_instructions_1)}\nThe person description text is here:\n{input_text}",
                        ),
                        Message(
                            role="assistant",
                            content=f"{detection_result}",
                        ),
                        Message(
                            role="user",
                            content=simple_rewriting_instruction.format(format_instructions_2=format_instructions_2)
                        )
                    ]
                )
            else:
                messages.append(
                    Message(
                        role="user",
                        content=simple_rewriting_instruction.format(format_instructions_2=format_instructions_2) +
                                f"\n\nThe biography is here: {input_text}",
                    )
                )
            output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions_2,
                                              parser=output_parser_2)
            if output_dict['parse_success'] is False:
                output_dict = parse_fixing(general_system_instruction, format_instructions_2, output_parser_2,
                                           output_dict, parser_model, [d.name for d in response_schemas_2])
        else:
            if not no_utility:
                if privacy_score == 'Yes':
                    # prev_rewriting += f"Suggestion: These entities {reflection_privacy} could be further generalized to improve the privacy score."
                    prev_rewriting += f"Suggestion: Entities that can be used to infer the person identity could be further generalized to improve the privacy score. You can refer the following detected sensitive entity list: {reflection_privacy}"
                else:
                    # prev_rewriting += f"Suggestion: These entities {reflection_utility} could be replaced with the original specific entities in the original biography to improve the utility score. You should also avoid specifying entities that could decrease the privacy score."
                    prev_rewriting += f"Suggestion: Entities that are about the description of the occupation but can not be used to infer the person identity could be specified to improve the utility score. "
                response_schemas = [
                    ResponseSchema(name="Anonymized text", description="your editing result")
                ]
                output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
                format_instructions = output_parser.get_format_instructions()

                messages = [
                    Message(
                        role="system",
                        content=general_system_instruction,
                    ),
                    Message(
                        role='user',
                        content=reinforcement_learning_instruction.format(p_threshold=p_threshold, format_instructions=format_instructions) + f"\n\nThe original biography is {input_text}\n\n{prev_rewriting}"
                    )
                ]
                output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                                  parser=output_parser)
                if output_dict['parse_success'] is False:
                    output_dict = parse_fixing(general_system_instruction, format_instructions, output_parser,
                                               output_dict, parser_model, [d.name for d in response_schemas])

            else:
                response_schemas_2 = [
                    ResponseSchema(name="Anonymized text", description="your anonymization result")
                ]
                output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas_2)
                format_instructions_2 = output_parser_2.get_format_instructions()
                response_schemas_3 = [
                    ResponseSchema(name="Anonymized text", description="your further anonymization result")
                ]
                output_parser_3 = StructuredOutputParser.from_response_schemas(response_schemas_3)
                format_instructions_3 = output_parser_3.get_format_instructions()

                messages = [
                    Message(
                        role="system",
                        content=general_system_instruction,
                    )
                ]
                if cot:
                    response_schemas_1 = [
                        ResponseSchema(name="People", description="name of the detected people separated by ', '"),
                        ResponseSchema(
                            name="Sensitive entities",
                            description="the list of detected sensitive entities where every two entities are separated by ', '",
                        ),
                    ]
                    output_parser_1 = StructuredOutputParser.from_response_schemas(response_schemas_1)
                    format_instructions_1 = output_parser_1.get_format_instructions()
                    messages.extend(
                        [
                            Message(
                                role="user",  # TODO: check this
                                content=f"{whole_task_instruction}\n\n{detection_result_prefix.format(format_instructions_1=format_instructions_1)}\nThe person description text is here:\n{input_text}",
                            ),
                            Message(
                                role="assistant",  # TODO: check this
                                content=f"{detection_result}",
                            ),
                            Message(
                                role="user",
                                content=f"{simple_rewriting_instruction.format(format_instructions_2=format_instructions_2)}",
                            )
                        ]
                    )
                else:
                    messages.extend(
                        [
                            Message(
                                role="user",
                                content=f"{simple_rewriting_instruction.format(format_instructions_2=format_instructions_2)}" +
                                f"\n\nThe biography is here: {input_text}",
                            ),
                            Message(
                                role="assistant",
                                content=prev_rewriting
                            )
                        ]
                    )
                assert privacy_score == 'Yes'
                messages.append(
                    Message(
                        role="user",
                        content=f"{reflection_privacy_rewriting_instruction.format(format_instructions_3=format_instructions_3)}\nThe entity list is here:\n{reflection_privacy}",
                    )
                )
                output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions_3,
                                                  parser=output_parser_3)
                if output_dict['parse_success'] is False:
                    output_dict = parse_fixing(general_system_instruction, format_instructions_3, output_parser_3,
                                               output_dict, parser_model, [d.name for d in response_schemas_3])

    else:
        if strategy == "simple":
            prompt = (f"{whole_task_instruction}\n[description text]:\n{input_text}\n{detection_result_prefix}"
                      f"\n{detection_result}\n{simple_rewriting_instruction}")
            output_dict, usage, finish_reason = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = (f"{whole_task_instruction}\n[description text]:\n{input_text}\n{detection_result_prefix}"
                      f"\n{detection_result}\n{simple_rewriting_instruction}"
                      f"{refelection_prev_re_instruction}\n{prev_rewriting}\n{reflection_privacy_instruction}"
                      f"\n{reflection_privacy}\n{refelection_utility_instruction}"
                      f"\n{reflection_utility}\n"
                      )
            output_dict, usage, finish_reason = model.generate(
                prompt, num_comps=num_comps, temperature=temperature)

    return output_dict


def generic_privacy_reflection(
        model: ModelBase,
        retriever,
        curr_rewriting: str,
        people: str,
        p_threshold: int,
        no_utility: bool,
        general_system_instruction: str,
        privacy_reflection_chat_instruction_1: str,
        privacy_reflection_completion_instruction_1: str,
        privacy_reflection_chat_instruction_2: str,
        privacy_reflection_completion_instruction_2: str
):
    if model.is_chat:
        response_schemas = [
            ResponseSchema(name="Candidates", description=f"the sorted list of name of {p_threshold} celebrity "
                                                          "candidates where every two names are separated by \', \'")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions_1 = output_parser.get_format_instructions()

        # retrieved_docs = retriever.invoke(curr_rewriting)
        # retrieved_docs_str = ""
        # for d in retrieved_docs:
        #     retrieved_docs_str += f"{d.page_content}\n"

        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{privacy_reflection_chat_instruction_1.format(format_instructions_1=format_instructions_1, p_threshold=p_threshold)}\n\nThe anonymized biography text is here:\n{curr_rewriting}'
                        # f'The retrieved context is here:{retrieved_docs_str}',
            )
        ]
        output_dict_1 = model.generate_chat(messages=messages, format_instructions=format_instructions_1,
                                            parser=output_parser)
        candidate = output_dict_1["Candidates"].split(', ')
        emb_model = SentenceTransformer("all-mpnet-base-v2")
        candidate_emb = emb_model.encode(candidate)
        people_emb = emb_model.encode(people)
        sim_score = candidate_emb.dot(people_emb)
        if True in (sim_score > 0.75):
            response_schemas = [
                ResponseSchema(name="Confirmation", description="\"Yes or No\""),
                ResponseSchema(name="Advice", description="\"the list of detected sensitive entities where"
                                                          " every two entities are separated by \', \'\"}")
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions_2 = output_parser.get_format_instructions()

            messages.append(
                Message(
                    role="assistant",
                    content=output_dict_1['raw_response']
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=f"The person described by the text is {people}. {privacy_reflection_completion_instruction_2.format(format_instructions_2=format_instructions_2)}"
                )
            )
            output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions_2,
                                              parser=output_parser)
            output_dict['candidate'] = candidate
            output_dict['rank'] = int(sim_score.argmax()) + 1
            output_dict['finish_reason_1'] = output_dict_1['finish_reason']
            if 'retry_finish_reason' in output_dict_1.keys():
                output_dict['retry_finish_reason_1'] = output_dict_1['retry_finish_reason']
        else:
            output_dict = output_dict_1
            output_dict["Confirmation"] = "No"
            output_dict["Advice"] = ""
            output_dict['candidate'] = candidate
            output_dict['rank'] = sim_score.size + 1
    else:
        output_dict = model.generate(
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
        response_schemas = [
            ResponseSchema(name="Occupation", description="the occupation you choose"),
            ResponseSchema(name="Confidence Score", description="the confidence score of making your choice")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{utility_reflection_chat_instruction_1.format(format_instructions=format_instructions)}\nThe anonymized biography is here:\n{curr_rewriting}',
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
        occupation = output_dict['Occupation']
        confidence_score = int(output_dict["Confidence Score"])
        if occupation != label or (occupation == label and confidence_score < 100):
            output_dict["Confirmation"] = "No"
        else:
            output_dict["Confirmation"] = "Yes"
        output_dict["Advice"] = ""
    else:
        output_dict = model.generate(
            f'{utility_reflection_completion_instruction_1}\n[Original text]:\n{input_text}\n[Anonymized text]:\n{curr_rewriting}\n[Classification label]{label}')

    return output_dict  # type: ignore


def generic_privacy_confidence_evaluation(
        model: ModelBase,
        curr_rewriting: str,
        people: str,
        general_system_instruction: str,
        privacy_confidence_evaluation_instruction: str,
):
    if model.is_chat:
        response_schemas = [
            ResponseSchema(name="Confidence Score", description="the confidence score of making your choice"),
            ResponseSchema(name="Reason", description="the reason for your scoring")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{privacy_confidence_evaluation_instruction.format(format_instructions=format_instructions)}\n\nThe anonymized text is here:\n{curr_rewriting}\nThe possible celebrity is {people}',
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
    else:
        output_dict = model.generate(
            f'{privacy_confidence_evaluation_instruction}\n[Description text]:\n{curr_rewriting}\n[Person name]:\n{people}')

    return output_dict  # type: ignore


def generic_privacy_selection_evaluation(
        model: ModelBase,
        curr_rewriting: str,
        people: str,
        candidate_list: str,
        general_system_instruction: str,
        candidate_generation_instruction: str,
        privacy_selection_evaluation_instruction: str,
):
    if model.is_chat:
        gn_flag = False
        if candidate_list is None:
            gn_flag = True
            response_schemas = [
                ResponseSchema(name="Similar celebrities", description="the list of the most similar celebrities where "
                                                                       "every two celebrities are separated by a comma")
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()

            messages = [
                Message(
                    role="system",
                    content=general_system_instruction,
                ),
                Message(
                    role="user",
                    content=f'{candidate_generation_instruction.format(format_instructions=format_instructions)}\n\nThe anonymized text is here:\n{curr_rewriting}\nThe described celebrity is {people}',
                )
            ]

            output_dict_1 = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                                parser=output_parser)
            candidate_list = copy.deepcopy(output_dict_1["Similar celebrities"])
            candidate_list = candidate_list.split(', ')
            candidate_list.append(people)
            random.shuffle(candidate_list)
            candidate_list = ', '.join(candidate_list)
        response_schemas = [
            ResponseSchema(name="People", description="the name of the most possible celebrity")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()

        messages = [
            Message(
                role="system",
                content=general_system_instruction,
            ),
            Message(
                role="user",
                content=f'{privacy_selection_evaluation_instruction.format(format_instructions=format_instructions)}\n\nThe anonymized text is here:\n{curr_rewriting}\nThe candidate list is here:\n{candidate_list}',
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
        emb_model = SentenceTransformer("all-mpnet-base-v2")
        candidate_emb = emb_model.encode(output_dict['People'])
        people_emb = emb_model.encode(people)
        sim_score = candidate_emb.dot(people_emb)
        if sim_score > 0.75:
            output_dict['success'] = True
        else:
            output_dict['success'] = False
        if gn_flag:
            output_dict['candidate_list'] = candidate_list
            output_dict['finish_reason_1'] = output_dict_1['finish_reason']
            if 'retry_finish_reason' in output_dict_1.keys():
                output_dict['retry_finish_reason_1'] = output_dict_1['retry_finish_reason']
    else:
        output_dict = model.generate(
            f'{privacy_selection_evaluation_instruction}\n[Description text]:\n{curr_rewriting}\n[Person name]:\n{people}')

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
