from programming_runs.generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import (generic_detection, generic_rewriting,
                              generic_privacy_reflection, generic_utility_reflection,
                              generic_privacy_selection_evaluation, generic_privacy_confidence_evaluation,
                              reddit_clss, reddit_privacy_reflection)

from typing import Optional, List, Union
import ast
import re
import os
from programming_runs.reddit_prompt import *


class RDReWriter(Generator):

    def detect(self, input_text: str, model: ModelBase):
        return generic_detection(
            input_text=input_text,
            model=model,
            whole_task_instruction=WHOLE_TASK_INSTRUCTION,
            general_task_instruction=GENERAL_SYSTEM_INSTRUCTION,
            detection_chat_instruction=DETECTION_INSTRUCTION,
            detection_completion_instruction=DETECTION_INSTRUCTION,
        )

    def rewrite(
            self,
            input_text: str,
            act_model: ModelBase,
            parser_model: ModelBase,
            strategy: str,
            cot: bool = False,
            prev_rewriting: Optional[str] = None,
            reflection_privacy: Optional[str] = None,
            reflection_utility: Optional[str] = None,
            privacy_score: Optional[int] = None,
            utility_score: Optional[int] = None,
            detection_result: Optional[str] = None,
            num_comps: int = 1,
            temperature: float = 0.0,
            p_threshold: int = 10,
            no_utility: bool = False
    ):
        return generic_rewriting(
            input_text=input_text,
            model=act_model,
            parser_model=parser_model,
            strategy=strategy,
            cot=cot,
            prev_rewriting=prev_rewriting,
            reflection_privacy=reflection_privacy,
            reflection_utility=reflection_utility,
            privacy_score=privacy_score,
            utility_score=utility_score,
            detection_result=detection_result,
            num_comps=num_comps,
            temperature=temperature,
            no_utility=no_utility,
            p_threshold=p_threshold,
            whole_task_instruction=WHOLE_TASK_INSTRUCTION,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            detection_result_prefix=DETECTION_INSTRUCTION,
            simple_rewriting_instruction=SIMPLE_REWRITING_INSTRUCTION,
            simple_rewriting_instruction_cot=SIMPLE_REWRITING_INSTRUCTION_COT,
            reflection_privacy_rewriting_instruction=REFELECTION_PRIVACY_REWRITING_INSTRUCTION,
            reinforcement_learning_instruction=REINFORCEMENT_INSTRUCTION
        )

    def privacy_reflex(self, model: ModelBase, rewriting, people, p_threshold, no_utility, retriever):
        return reddit_privacy_reflection(
            model=model,
            retriever=retriever,
            curr_rewriting=rewriting,
            people=people,
            p_threshold=p_threshold,
            no_utility=no_utility,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            privacy_reflection_chat_instruction_1=PRIVACY_REFLECTION_INSTRUCTION_1,
            privacy_reflection_completion_instruction_1=PRIVACY_REFLECTION_INSTRUCTION_1,
            privacy_reflection_chat_instruction_2=PRIVACY_REFLECTION_INSTRUCTION_2,
            privacy_reflection_completion_instruction_2=PRIVACY_REFLECTION_INSTRUCTION_2
        )

    def utility_reflex(self, input_text: str, model: ModelBase, rewriting, label, privacy_score):
        return generic_utility_reflection(
            input_text=input_text,
            model=model,
            label=label,
            privacy_score=privacy_score,
            curr_rewriting=rewriting,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            utility_reflection_chat_instruction_1=UTILITY_REFLECTION_INSTRUCTION_1,
            utility_reflection_completion_instruction_1=UTILITY_REFLECTION_INSTRUCTION_1,
        )

    def privacy_confidence_evaluation(self, model: ModelBase, rewriting, people):
        return generic_privacy_confidence_evaluation(
            model=model,
            curr_rewriting=rewriting,
            people=people,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            privacy_confidence_evaluation_instruction=PRIVACY_EVALUATION_CONFIDENCE_INSTRUCTION
        )

    def privacy_selection_evaluation(self, model: ModelBase, rewriting, people, candidate_list):
        return generic_privacy_selection_evaluation(
            model=model,
            curr_rewriting=rewriting,
            people=people,
            candidate_list=candidate_list,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            candidate_generation_instruction=PRIVACY_EVALUATION_SELECTION_INSTRUCTION_1,
            privacy_selection_evaluation_instruction=PRIVACY_EVALUATION_SELECTION_INSTRUCTION_2
        )

    def clssification(self, model: ModelBase, comment):
        return reddit_clss(
            model=model,
            comment=comment,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            reddit_clss_chat_instruction_1=OCC_CLSS_INSTRUCTION,
            reddit_clss_completion_instruction_1=OCC_CLSS_INSTRUCTION
        )


DUMMY_FUNC_SIG = "def func():"
DUMMY_FUNC_CALL = "func()"


def handle_first_line_indent(func_body: str) -> str:
    if func_body.startswith("    "):
        return func_body
    split = func_body.splitlines()
    return f"    {split[0]}\n" + "\n".join(split[1:])


def handle_entire_body_indent(func_body: str) -> str:
    split = func_body.splitlines()
    res = "\n".join(["    " + line for line in split])
    return res


def fix_turbo_response(func_body: str) -> str:
    return fix_markdown(remove_unindented_signatures(func_body))


def fix_markdown(func_body: str) -> str:
    return re.sub("`{3}", "", func_body)


def remove_unindented_signatures(code: str) -> str:
    regex = r"^def\s+\w+\s*\("

    before_signature = []
    after_signature = []
    signature_found = False

    for line in code.split("\n"):
        if re.match(regex, line):
            signature_found = True
            continue

        if signature_found:
            after_signature.append(line)
        else:
            if not line.startswith("    ") and line.strip():
                line = "    " + line
            before_signature.append(line)

    return "\n".join(before_signature + after_signature)


def py_fix_indentation(func_body: str) -> str:
    func_body = fix_turbo_response(func_body)
    """
    3 cases:
        1. good syntax
        2. first line not good
        3. entire body not good
    """

    def parse_indent_rec(f_body: str, cur_state: int) -> str:
        f_body = fix_markdown(f_body)
        if cur_state > 1:
            return f_body
        code = f'{DUMMY_FUNC_SIG}\n{f_body}\n{DUMMY_FUNC_CALL}'
        try:
            exec(code)
            return f_body
        except (IndentationError, SyntaxError):
            p_func = handle_first_line_indent if cur_state == 0 else handle_entire_body_indent
            return parse_indent_rec(p_func(func_body), cur_state + 1)
        except Exception:
            return f_body

    return parse_indent_rec(func_body, 0)


def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False
