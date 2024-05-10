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
                content=f"{whole_task_instruction}\n\n{detection_chat_instruction.format(format_instructions_1=format_instructions, input_text=input_text)}"
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
    label: str,
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
    simple_rewriting_instruction: str,
    simple_rewriting_instruction_cot: str,
    reflection_privacy_rewriting_instruction: str,
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
                            content=f"{whole_task_instruction}\n\n{detection_result_prefix.format(format_instructions_1=format_instructions_1, input_text=input_text)}",
                        ),
                        Message(
                            role="assistant",
                            content=f"{detection_result}",
                        ),
                        Message(
                            role="user",
                            content=simple_rewriting_instruction_cot.format(format_instructions_2=format_instructions_2)
                        )
                    ]
                )
            else:
                messages.append(
                    Message(
                        role="user",
                        content=simple_rewriting_instruction.format(format_instructions_2=format_instructions_2,
                                                                    input_text=input_text),
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
                    prev_rewriting += f"Suggestion: Entities that can help classify the person's occupation as {label} but can not be used to infer the person's identity could be specified to improve the utility score. "
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
                        content=reinforcement_learning_instruction.format(p_threshold=p_threshold,
                                                                          format_instructions=format_instructions,
                                                                          input_text=input_text,
                                                                          prev_rewriting=prev_rewriting)
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
                                content=f"{whole_task_instruction}\n\n{detection_result_prefix.format(format_instructions_1=format_instructions_1, input_text=input_text)}",
                            ),
                            Message(
                                role="assistant",  # TODO: check this
                                content=f"{detection_result}",
                            ),
                            Message(
                                role="user",
                                content=f"{simple_rewriting_instruction_cot.format(format_instructions_2=format_instructions_2)}",
                            )
                        ]
                    )
                else:
                    messages.extend(
                        [
                            Message(
                                role="user",
                                content=f"{simple_rewriting_instruction.format(format_instructions_2=format_instructions_2, input_text=input_text)}",
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
                        content=f"{reflection_privacy_rewriting_instruction.format(format_instructions_3=format_instructions_3, reflection_privacy=reflection_privacy)}",
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
                      f"\n{prev_rewriting}\n"
                      f"\n{reflection_privacy}\n"
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
                content=f'{privacy_reflection_chat_instruction_1.format(format_instructions_1=format_instructions_1, p_threshold=p_threshold, curr_rewriting=curr_rewriting)}'
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
                    content=f"{privacy_reflection_chat_instruction_2.format(people=people, format_instructions_2=format_instructions_2)}"
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
):
    if model.is_chat:
        response_schemas = [
            # ResponseSchema(name="Occupation", description="the occupation you choose"),
            ResponseSchema(name="Confidence Score", description="the confidence score of making the classification")
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
                content=f'{utility_reflection_chat_instruction_1.format(format_instructions=format_instructions, curr_rewriting=curr_rewriting, label=label)}',
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
        # occupation = output_dict['Occupation']
        confidence_score = int(output_dict["Confidence Score"])
        if confidence_score <= 95:
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
                content=f'{privacy_confidence_evaluation_instruction.format(format_instructions=format_instructions, curr_rewriting=curr_rewriting, people=people)}',
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
                    content=f'{candidate_generation_instruction.format(format_instructions=format_instructions, curr_rewriting=curr_rewriting, people=people)}',
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
                content=f'{privacy_selection_evaluation_instruction.format(format_instructions=format_instructions, curr_rewriting=curr_rewriting, candidate_list=candidate_list)}',
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


def reddit_clss(
        model: ModelBase,
        comment: str,
        general_system_instruction: str,
        reddit_clss_chat_instruction_1: str,
        reddit_clss_completion_instruction_1: str
):
    if model.is_chat:
        response_schemas = [
            ResponseSchema(name="Occupation", description="the one occupation you choose from the above "
                                                          "35 occupation categories"),
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
                content=f'{reddit_clss_chat_instruction_1.format(format_instructions=format_instructions, curr_rewriting=comment)}',
            )
        ]
        output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions,
                                          parser=output_parser)
    else:
        output_dict = model.generate(
            f'{reddit_clss_completion_instruction_1}\n[Original text]:\n{comment}\n[Anonymized text]:\n{comment}')

    return output_dict  # type: ignore


def reddit_privacy_reflection(
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
            ResponseSchema(name="Age", description=f"the sorted list of top {p_threshold} guesses of the specific age of the "
                                                   f"author when he wrote the comment where every two guesses are"
                                                   f" separated by \', \'"),
            ResponseSchema(name="Sex", description=f"Choose from these options: Male , Female."),
            ResponseSchema(name="Location", description=f"the sorted list of top {p_threshold} guesses of the current"
                                                        f" place of living of the author where every two guesses are"
                                                        f" separated by \', \'"),
            ResponseSchema(name="Place of birth", description=f"the sorted list of top {p_threshold} guesses of the"
                                                              f" place of birth of the author where every two guesses "
                                                              f"are separated by \', \'"),
            ResponseSchema(name="Education", description=f"the sorted list of top {p_threshold} guesses of the education background of the "
                                                         f"author where every two guesses are"
                                                         f" separated by \', \', each guess should be in the following"
                                                         f" format: 'education level' in 'major name',"
                                                         f" e.g. Masters in Computer Science"),
            ResponseSchema(name="Income level", description=f"Choose from these options: No income, Low (<30k USD), "
                                                            f"Medium (30−60k USD), High (60−150 k USD), "
                                                            f"Very High (>150k USD)"),
            ResponseSchema(name="Relationship status", description=f"Choose from these options: No relation, "
                                                                   f"In Relation, Married, Divorced.")
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
                content=f'{privacy_reflection_chat_instruction_1.format(format_instructions_1=format_instructions_1, curr_rewriting=curr_rewriting)}'
                        # f'The retrieved context is here:{retrieved_docs_str}',
            )
        ]
        output_dict_1 = model.generate_chat(messages=messages, format_instructions=format_instructions_1,
                                          parser=output_parser)
        personality = {}
        for d in response_schemas:
            personality[d.name] = output_dict_1[d.name]
        personality_res = {}
        cp_text = ''
        emb_model = SentenceTransformer("all-mpnet-base-v2")
        mapped_feature = {
            "Age": "age",
            "Sex": "sex",
            "Location": "city_country",
            "Place of birth": "birth_city_country",
            "Education": "education",
            "Income level": 'income_level',
            "Relationship status": "relationship_status",
        }
        for k, v in personality.items():
            if k in ['Age', 'Location', 'Place of birth', 'Education']:
                candidates = v.split(', ')
                candidate_emb = emb_model.encode(candidates)
                people_emb = emb_model.encode(str(people[mapped_feature[k]]))
                sim_score = candidate_emb.dot(people_emb)
                if True in (sim_score > 0.75):
                    personality_res[k] = True
                    cp_text += f"Attribute type: {k}\nGround truth: {people[mapped_feature[k]]}\nPrediction: {candidates[sim_score.argmax()]}\n\n"
                else:
                    personality_res[k] = False
            else:
                assert k in ['Sex', 'Income level', 'Relationship status']
                if k == 'Income level':
                    v = v.split(' ')[0]
                candidate_emb = emb_model.encode(v)
                people_emb = emb_model.encode(str(people[mapped_feature[k]]))
                sim_score = candidate_emb.dot(people_emb)
                if sim_score > 0.75:
                    personality_res[k] = True
                    cp_text += f"Attribute type: {k}\nGround truth: {people[mapped_feature[k]]}\nPrediction: {v}\n\n"
                else:
                    personality_res[k] = False

        if True in personality_res.values():
            response_schemas = [
                ResponseSchema(name="Confirmations", description="For each ground truth-prediction pair "
                                                                "output 'Yes' or 'No', separated by ', '"),
                ResponseSchema(name="Advice", description="For the ground truth-prediction pair that you output 'Yes', "
                                                          "write the corresponding personal attribute name and write your inference process of "
                                                          "the attribute, each attribute is separated by '; '"),
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
                    content=f"{privacy_reflection_chat_instruction_2.format(cp_text=cp_text, format_instructions_2=format_instructions_2)}"
                )
            )
            output_dict = model.generate_chat(messages=messages, format_instructions=format_instructions_2,
                                              parser=output_parser)
            output_dict['personality'] = personality
            correct_num = 0
            for t in output_dict['Confirmations'].split(', '):
                if t == 'Yes':
                    correct_num += 1
            if output_dict['Advice'] == '':
                if correct_num == 0:
                    output_dict["Confirmation"] = "No"
                else:
                    temp_str = 'Personal attributes including '
                    for k, v in personality_res.items():
                        if v:
                            temp_str += f'{k}, '
                    temp_str = temp_str[:-2] + ' can still be correctly inferred'
                    output_dict['Advice'] = temp_str
            output_dict['rank'] = 7 - correct_num
            output_dict["Confirmation"] = "Yes"
            output_dict['finish_reason_1'] = output_dict_1['finish_reason']
            if 'retry_finish_reason' in output_dict_1.keys():
                output_dict['retry_finish_reason_1'] = output_dict_1['retry_finish_reason']
        else:
            output_dict = output_dict_1
            output_dict["Confirmation"] = "No"
            output_dict["Advice"] = ""
            output_dict['candidate'] = personality
            output_dict['rank'] = 7
    else:
        output_dict = model.generate(
            f'{privacy_reflection_completion_instruction_1}\n[Description text]:\n{curr_rewriting}\n[Person name]:\n{people}')

    return output_dict


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
