from programming_runs.generators.model import ModelBase, message_to_str
from .generator_types import Generator
from .generator_utils import (generic_detection, generic_rewriting,
                              generic_privacy_reflection, generic_utility_reflection)

from typing import Optional, List, Union
import ast
import re

DETECTION_INSTRUCTION = (
    "Let's first complete the personally identifiable information detection task, given the person description text, you need to answer the name of the"
    " person that is mainly talked about by the text, then identify every entities "
    "(including words, dates, and numerals) in the provided person description that"
    " can be connected to the person being talked about, such that if these tokens "
    "are removed the result text is anonymised, namely people can not know who is described in the text. "
    "Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure:\n{\"People\": \"name of the detected people separated by \', \'\", \"Sensitive entities\": \" the list of detected sensitive entities where every two entities are separated by \', \'\"}")
GENERAL_SYSTEM_INSTRUCTION = ("You are an efficient assistant. "
                              "Only respond with the given JSON template filled with your answers.")
WHOLE_TASK_INSTRUCTION = ("You are a text anonymization annotator. "
                          "I require your assistance in anonymizing the given "
                          "description text to facilitate sharing it "
                          "with other people while preserving its utility. \n"
                          "Let's think step by step, to achieve this goal, "
                          "first, you need to detect all the sensitive entities. "
                          "Second, you need to rewrite the text by replacing sensitive entities with more general "
                          "entities to make "
                          "them less specific and insert or remove tokens necessary to ensure the final "
                          "output text is fluent and natural. The style and format of the output text should "
                          "be consistent with the input text."
                          # "The level of generalization is constrained by the downstream task,"
                          # " occupation classification, that means your anonymization "
                          # "should not change the occupation label."
                          )
DETECTION_RESULT_PREFIX = "All the detected sensitive entities are here:\n[Sensitive entities]:\n]"
REFELECTION_PREV_RE_INSTRUCTION = ("Here is the list of your previous rewriting results "
                                   "and the corresponding evaluation score")
REFELECTION_PRIVACY_INSTRUCTION = "Here is the feedback about how to improve the privacy level"
REFELECTION_UTILITY_INSTRUCTION = "Here is the feedback about how to improve the utility level"
REFELECTION_PRIVACY_REWRITING_INSTRUCTION = (
    "Here is the list of the sensitive entities that still can be used to correctly infer"
    "the identity of the people being talked about by your anonymized text."
    " Please further generalize the sensitive entities in your anonymized text"
    # " about "
    # "how to further improve the privacy safety of it. Please refer to"
    # " the feedback"
    " to give me a better anonymized text with a higher level of safety. "
    "Note that you are only allowed to decrease the specificity by "
    "replacing sensitive"
    " words or entities with more general one and should not delete the"
    " content of the text. Only tokens necessary to make the sentence natural and fluent can be added. "
    "Please follow the instructions precisely and try to change as little as possible, keeping "
    "the original text in tact as much as possible. "
    "Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure:\n{\"Anonymized text\": \"the generalized text\"}"
    )
REFELECTION_UTILITY_REWRITING_INSTRUCTION = ("Here is the list of confused entities that can do harm to the performance of the downstream"
                                             " occupation classification task on the anonymized text. Please specialize"
                                             " these entities referring to the original person description text and"
                                             # "feedback about "
                                             # "how to improve the utility level. Please refer to the feedback to"
                                             " give me a better anonymized text with higher utility. "
                                             "Note that you are only allowed to increase the utility of the anonymized"
                                             " text by replacing "
                                             "the confused entities with more specific entities in the "
                                             "original description text. "
                                             # "Entities that can be used to directly identify specifically "
                                             # "who is talked by "
                                             # "the text should not be recovered from the original text. "
                                             "Fill in your response into the corresponding position enclosed in \"\" "
                                             "in the following JSON structure and respond me"
                                             " with the filled JSON structure:\n{\"Anonymized text\": "
                                             "\"the result text with better utility\"}"
                                             )
SIMPLE_REWRITING_INSTRUCTION = (
    "Let's now complete the second task. "
    "You are an expert anonymizer that carefully choose some of the detected sensitive entities of the"
    " text and then generalize them to make the text anonymous, namely the text can be suitable to describe not only a specific one person but many similar "
    "person. You follow the instructions and format precisely and you try to change as "
    "little as possible, keep the original text in tact as much as possible. Only generalize information and do not"
    " invent new information. \nExample: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> "
    "'my wife and I' is not. \nExample: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, "
    "but 'my husband and I have a dog' -> 'my partner and I have a cat' is not. \nExample: 'He is a basketball player'"
    " -> 'He is an athlete' is valid but 'He is an artist' or 'She is an athlete', 'They are athlete' is not.\n"
    # "You are an expert anonymizer that carefully adapts small parts of the description text to make it anonymous."
    # " Given a paragraph of description text, please generalize all your detected entities that can be used to infer the person being talked about to make sure people can not easily guess who is talked by the description text."
    "Names can be replaced with common pronoun like 'A person', or other "
    "sensitive entities can be replaced with more general counterparts. "
    # "Note that the generalization should not affect the occupation classification of the text."
    "Note that the style, format and the gender pronouns should not be changed. Especially pay attention to the "
    "consistent use of \"he\", \"she\" and \"They\". You are only allowed to adapt small parts of the description"
    " text to make it anonymous."
    " Additionally, you can insert as little as possible tokens to ensure the final "
    "output text is fluent and natural while preserving anonymity. Removing content is not allowed, only decrease"
    "the specificity by generalizing."
    # "meaning and coherence of the text."
    # " Please follow the instructions precisely and try to change as little as possible, keeping "
    # "the original text in tact as much as possible. Only generalize your detected sensitive information and do not invent new information. "
    "Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure:\n{\"Anonymized text\": \"your anonymization result\"}")
PRIVACY_REFLECTION_INSTRUCTION_1 = (
    # "You are very good at evaluating the level of association between the given celebrity and a paragraph of biography."
    # " Given the name of the celebrity and the biography, you are supposed to give a confidence score confirming "
    # "that the given celebrity is the people described by the biography. The score should be on a continuous scale from "
    # "0 to 100 that starts with \"Completely irrelevant\", goes through \"Partial descriptions matching\", then \"Most "
    # "descriptions matching\", up to \"Exactly matching\". Besides, you should detect and respond with a list of"
    # " entities appearing in the biography that you used to evaluate the confidence score and could be generalized to "
    # "decrease the association confidence score. "
    # "Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and respond "
    # "me with the filled JSON structure:\n{\"Confidence score\": \"the association confidenc escore\", \"Entities\": "
    # "\"the list of entities where every two entities are separated by \', \'\"}"
    "You are very knowledgeable about celebrities and can identify them from the given anonymized biography."
    " When asked to identify individuals, disregard any missing information and context, like placeholder '<PERSON>' or 'ORGANIZATION', "
    "respond only with the names of {p_threshold} celebrity candidates most likely being talked by"
    " the provided anonymized biography text and sort the names from highest to lowest probability."
    " Now identify the individual talked by the following text and "
    "fill in your response into the corresponding position enclosed in \"\" in the following JSON structure "
    "and respond me with the filled JSON structure:\n{js}"
    # "You are an experienced pricurrent or finalvacy risk evaluator. Given a paragraph of description text and the name of"
    # " a person described by the text, you should first play a role of attacker to try to guess who is described by the text."
    # " Then you need to compare your guess result with the given person name to see whether you guess correctly or not. "
    # "Now give me your guess result."
)
# " you should first assess whether people can correctly guess the identity of the person described by the txt. "
PRIVACY_REFLECTION_INSTRUCTION_2 = ("If the name is within your output name list, respond with \'Yes\' and"
                                    # "with me a confidence score (0-100) of associating the person with the description text. "
                                    " you need to detect and then respond me with the sensitive entities (including words, dates, and numerals) in the biography that "
                                    "you just used to infer the person's identity and should be generalized to decrease the confidence score of associating the person with the biography text. Otherwise, just give me the answer \'No\'."
                                    # "You are very knowledgeable about celebrities. Given the description text"
                                    #                               " and the name of a person, please respond with the confidence score "
                                    #                               "about associating the description text with the given person, 0 represents fully "
                                    #                               "irrelevant and 100 represents fully relevant, then give me feedback about why you give this score and "
                                    #                               # "which part is the clues that you use to infer the identity of the person and should be further generalized to decrease the confidence score"
                                    " Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure"
                                    # ":\n{\"Confidence score\": \"your confidence score about associating\","
                                    # " \"Advice\": \"your advice about how to further generalize the text\"}"
                                    ":\n{\"Confirmation\": \"Yes or No\","
                                    " \"Advice\": \"the list of detected sensitive entities where every two entities are separated by \', \'\"}"
                                    )
UTILITY_REFLECTION_INSTRUCTION_1 = (
    "You are an expert at classifying people's occupation according to their anonymized biography. Given the anonymized biography of"
    " a person and the set of optional occupation, choose the most likely occupation of the person from the optional "
    "occupation set according to the anonymized biography and give me your confidence score of making this classification on a continuous scale from 0 to 100.\nThe optional occupation set is here:\n"
    "Chef, Classical Music Artist, Table Tennis Player, Entomologist, Lacrosse Player, Astronaut, Medician,"
    " Fashion Designer, Horse Trainer, Ambassador, Photographer, Engineer, Formula One Racer, Comedian, Martial Artist, "
    "Chess Player, Painter, Soccer Player, Tennis Player, Architect, Cyclist, Basketball Player, Congressman, Baseball Player. \n"
    "Fill in your response into "
    "the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure"
    ":\n{\"Occupation\": \"the occupation you choose\", \"Confidence Score\": \"the confidence score of making your choice\": \"\"}"
)
UTILITY_REFLECTION_INSTRUCTION_2 = (
    # "If your answer is incorrect or your confidence score of making a correct classification is not higher than 90, respond with \'No\' and "
                                    # "and refer to the "
                                    "You need to detect and then respond me with the confused entities in the occupation description part of the biography that interfere with you from "
                                    "making a correct classification with high confidence and should be specialized to make you can correctly"
                                    "classify the occupation. Only respond the a few entities most relevant to the occupation description. "
                                    # "following original biography and give me advice about which span of the original"
                                    # " text mainly contains the occupation information rather than person identity information and should not be anonymized "
                                    # "and should be recovered to the anonymized text to keep the utility of the "
                                    # "anonymized text while keeping the identity of the person anonymous at the "
                                    # "same time."
                                    # " Otherwise, just respond with \'Yes\'. "
                                    "Fill in your response into "
                                    "the corresponding position enclosed in \"\" in the following JSON structure and respond me with the filled JSON structure"
                                    # ":\n{\"Confirmation\": \"Yes or No\", "
                                    ":\n{\"Advice\": \"the list of confused entities where every two entities are separated by \', \'\"}")

REINFORCEMENT_INSTRUCTION = (
    "Let's play a game to maximize accumulated rewards by editing biography texts. "
    "Here’s how it works: You'll be provided with a paragraph of biography text alongside its editing history. "
    "Each edit is associated with a privacy score, indicating the level of personal privacy maintained, and a utility "
    "score, reflecting its usefulness for a specific task.\n\n"
    "Your task is to edit the latest entry in the editing history. "
    "You can only alter the text by replacing some entities with more general entities to improve the privacy score, or more specific entities in the"
    " original biography to improve the utility score. \n"
    "Here are the rewarding rules:\n"
    "- If the privacy score is {p_threshold} or less, the reward is equivalent to the privacy score.\n"
    "- If the privacy score exceeds 10, the reward is equivalent to the utility score.\n"
    "The editing goal is to get a higher reward than before while keeping changes to the "
    "text minimal, ensuring the original context and structure are preserved as much as possible. "
    "You are also permitted to make minor adjustments to the text to maintain fluency and naturalness.\n"
    "For example:\n"
    "- Generalize: 'my husband and I' can be changed to 'my partner and I'. Incorrect generalizations like changing "
    "'my husband and I' to 'my wife and I' are not allowed.\n"
    "- Specify: You may replace generic terms with specific ones from the original text to increase utility, "
    "without changing unrelated details.\n\n"
    "Your ultimate goal is to maximize the total accumulated reward through strategic edits. I will provide editing "
    "suggestions to guide your editing towards achieving higher rewards. "
    "Fill in your response into the corresponding position enclosed in \"\" in the following JSON structure and "
    "respond me with the filled JSON structure:\n{js}")

    # "Let's play a maximizing accumulated reward game. I have a paragraph of biography text. "
    #                          "You will be given a list of editing history of the biography text with each edited text corresponding to a privacy score, which manifests the level of privacy"
    #                          " safety, and a utility score, which manifests the level of utility for a downstream "
    #                          "task. The reward of every editing result depends on these two scores: if privacy score is less than or equal to 10, the reward is equal to the privacy score, else the reward is equal to the utility score."
    #                          " You are supposed to edit the last one of editing history to get a higher reward so as to maximize the accumulated reward. You are only allowed to edit the text by replacing some entities with more general"
    #                          " entities, to get a higher privacy reward, or by replacing some entities with more "
    #                          "specific entities in the original biography, to get a higher utility reward. "
    #                          "Your final goal is to maximize the reward accumulating across all rounds of your editing."
    #                          " I will give you editing suggestions to help you make correct "
    #                          "editing actions to get higher reward."
    #                          )
                             # "In the following, there is a paragraph of biography text, "
                             # "and an anonymized version of the biography text, and the privacy and utility reward score of the "
                             # "anonymized biography text. You can select and then replace some entities in the anonymized"
                             # " biography text with more general entities to get a higher privacy reward. You can also"
                             # " select and replace some entities in the anonymized with more specific entities referring "
                             # "to the original biography to get a higher utility reward. "
                             # ", your task it to try to get as much as possible the accumulative reward.")
# "You are a utility evaluator of the anonymized text. Given the original text, "
# "the anonymized text and the classification label of the original text,"
# "you should first determine whether a people can still classify the"
# " anonymized text as the given classification label."
# "If your answer is \'No\', then compare the anonymized text and the original text "
# "and give me advice about which part of the original text should not be anonymized "
# "and should be recovered to the anonymized text to keep the utility of the anonymized"
# " text while keep the identity of the person anonymous at the same time. Otherwise, just respond with \'Yes\'."
# "Fill in your response into the corresponding position enclosed in \"\" in the following template "
# ":\n\n{\"Confirmation\": \"Yes or No\", "
# "\"Advice\": \"your advice for keeping the utility if your answer is No\"}"
# " your task is to first give me the confidence score (0-100) of classifying the"
# " anonymized text as the given classification label. "
# "Then, compare the anonymized text and the original text and give me advice about"
# " which part of the original text should not be anonymized and should be recovered to the anonymized text to "
# "improve the confidence score of classifying the anonymized text as the given "
# "classification label without leaking the person identity mentioned in the original text."
# "Fill in your response into the corresponding position enclosed in \"\" in the following template "
# ":\n{\"Confidence score\": \"your confidence score about classifying\", "
# "\"Advice\": \"your advice for improve the confidence score\"}"


class ReWriter(Generator):

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
            model: ModelBase,
            strategy: str,
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
    ) -> Union[str, List[str]]:
        return generic_rewriting(
            input_text=input_text,
            model=model,
            strategy=strategy,
            prev_rewriting=prev_rewriting,
            reflection_privacy=reflection_privacy,
            reflection_utility=reflection_utility,
            privacy_score=privacy_score,
            utility_score=utility_score,
            detection_result=detection_result,
            num_comps=num_comps,
            temperature=temperature,
            no_utility=no_utility,
            whole_task_instruction=WHOLE_TASK_INSTRUCTION,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            detection_result_prefix=DETECTION_INSTRUCTION,
            refelection_prev_re_instruction=REFELECTION_PREV_RE_INSTRUCTION,
            reflection_privacy_instruction=REFELECTION_PRIVACY_INSTRUCTION,
            refelection_utility_instruction=REFELECTION_UTILITY_INSTRUCTION,
            simple_rewriting_instruction=SIMPLE_REWRITING_INSTRUCTION,
            reflection_privacy_rewriting_instruction=REFELECTION_PRIVACY_REWRITING_INSTRUCTION,
            reflection_utility_rewriting_instruction=REFELECTION_UTILITY_REWRITING_INSTRUCTION,
            reinforcement_learning_instruction=REINFORCEMENT_INSTRUCTION.format(p_threshold=p_threshold,
                                                                                js="{\"Anonymized text\": "
                                                                                   "\"your editing result\"}")
        )

    def privacy_reflex(self, model: ModelBase, rewriting, people, p_threshold, no_utility):
        return generic_privacy_reflection(
            model=model,
            curr_rewriting=rewriting,
            people=people,
            no_utility=no_utility,
            general_system_instruction=GENERAL_SYSTEM_INSTRUCTION,
            privacy_reflection_chat_instruction_1=PRIVACY_REFLECTION_INSTRUCTION_1.format(p_threshold=p_threshold, js=
                                                                                          "{\"Candidates\": \"the sorted list "
                                                                                          "of name of celebrity candidates where every two names are separated by \', \'\"}"),
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
            utility_reflection_chat_instruction_2=UTILITY_REFLECTION_INSTRUCTION_2,
            utility_reflection_completion_instruction_2=UTILITY_REFLECTION_INSTRUCTION_2
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
