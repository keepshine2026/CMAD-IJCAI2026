from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptSet(ABC):
    """
    Abstract base class for a set of prompts.
    """
    @staticmethod
    @abstractmethod
    def get_role() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_constraint() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_format() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_adversarial_answer_prompt(question) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_query_prompt(question) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_file_analysis_prompt(query, file) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_websearch_prompt(query) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_distill_websearch_prompt(query, results) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_reflect_prompt(question, answer) -> str:
        raise NotImplementedError

    @staticmethod
    def get_react_prompt(question, solutions, feedback) -> str:
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    # def get_self_consistency(materials: Dict[str, Any]) -> str:
    #     """ TODO """

    # @staticmethod
    # @abstractmethod
    # def get_select_best(materials: Dict[str, Any]) -> str:
    #     """ TODO """

    @staticmethod
    @abstractmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_decision_constraint() -> str:
        raise NotImplementedError
        
    @staticmethod
    @abstractmethod
    def get_decision_role() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_decision_few_shot() -> str:
        raise NotImplementedError
