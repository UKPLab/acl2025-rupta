from typing import List, Optional, Union
from abc import abstractmethod, ABC

from programming_runs.generators.model import ModelBase


class Generator:
    @abstractmethod
    def self_reflection(self, func: str, feedback: str, model: ModelBase) -> str:
        ...

    @abstractmethod
    def func_impl(
        self,
        func_sig: str,
        model: ModelBase,
        strategy: str,
        prev_func_impl: Optional[str] = None,
        feedback: Optional[str] = None,
        self_reflection: Optional[str] = None,
        num_comps: int = 1,
        temperature: float = 0.0,
    ) -> Union[str, List[str]]:
        ...

    @abstractmethod
    def internal_tests(
            self,
            func_sig: str,
            model: ModelBase,
            max_num_tests: int = 5
    ) -> List[str]:
        ...

    @abstractmethod
    def detect(self, input_text: str, model: ModelBase):
        ...

    @abstractmethod
    def rewrite(
            self,
            input_text: str,
            label: str,
            people,
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
        ...

    @abstractmethod
    def privacy_reflex(self, model: ModelBase, rewriting, people, p_threshold, no_utility, retriever):
        ...

    @abstractmethod
    def utility_reflex(self, input_text: str, model: ModelBase, rewriting, label, privacy_score):
        ...

    @abstractmethod
    def privacy_confidence_evaluation(self, model: ModelBase, rewriting, people):
        ...

    @abstractmethod
    def privacy_selection_evaluation(self, model: ModelBase, rewriting, original_text, people, candidate_list):
        ...