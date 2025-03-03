# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import argparse
from enum import Enum
from typing import Any, Dict, List

MAX_KEY_LENGTH = 256
MAX_JSON_LENGTH = 4096


class ArgumentAction(Enum):
    STORE = 'store'
    STORE_TRUE = 'store_true'
    STORE_FALSE = 'store_false'
    STORE_CONST = 'store_const'
    APPEND = 'append'
    COUNT = 'count'


class ArgumentValidator:
    context = None

    def __init__(self, *args, allow_none: bool = False, **kwargs):
        self.allow_none = allow_none
        self.validation_pipeline = []
        self.create_validation_pipeline()

    def validate(self, value: Any) -> None:
        if value is None and self.allow_none:
            return
        for method in self.validation_pipeline:
            method(value)

    def add_validation_method(self, method, position: int = None, target_method=None):
        if position is not None:
            self.validation_pipeline.insert(position, method)
        elif target_method and target_method in self.validation_pipeline:
            target_index = self.validation_pipeline.index(target_method)
            self.validation_pipeline.insert(target_index + 1, method)
        else:
            self.validation_pipeline.append(method)

    def delete_validation_method(self, method=None, position: int = None):
        if position is not None:
            if 0 <= position < len(self.validation_pipeline):
                self.validation_pipeline.pop(position)
        elif method and method in self.validation_pipeline:
            self.validation_pipeline.remove(method)

    def _create_validation_pipeline(self, *methods):
        self.validation_pipeline.clear()
        self.validation_pipeline.extend(methods)


class StringArgumentValidator(ArgumentValidator):
    def __init__(self, min_length: int = 0, max_length: int = float('inf'), allow_none: bool = False):
        super().__init__(allow_none=allow_none)
        self.min_length = min_length
        self.max_length = max_length

    @staticmethod
    def validate_type(value: str) -> None:
        if not isinstance(value, str):
            raise argparse.ArgumentTypeError("Value must be a string")

    def validate_length(self, value: str) -> None:
        if not (self.min_length <= len(value) <= self.max_length):
            raise argparse.ArgumentTypeError(f"String length must be between {self.min_length} and {self.max_length}")

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type, self.validate_length)


class NumberArgumentValidator(ArgumentValidator):
    def __init__(self, min_value: float = float('-inf'), max_value: float = float('inf'),
                 special_values: List[float] = None, allow_none: bool = False):
        super().__init__(allow_none=allow_none)
        self.min_value = min_value
        self.max_value = max_value
        self.special_values = special_values if special_values is not None else []

    @staticmethod
    def validate_type(value: Any) -> None:
        if not isinstance(value, (int, float)):
            raise argparse.ArgumentTypeError("Value must be a number")

    def validate_range(self, value: Any) -> None:
        if value in self.special_values:
            return
        if self.min_value is not None and self.max_value is not None and not (
                self.min_value <= value <= self.max_value):
            raise argparse.ArgumentTypeError(f"Number must be between {self.min_value} and {self.max_value}")
        if self.min_value is not None and value < self.min_value:
            raise argparse.ArgumentTypeError(f"Number must be greater than or equal to {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise argparse.ArgumentTypeError(f"Number must be lower than or equal to {self.max_value}")

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type, self.validate_range)


class BooleanArgumentValidator(ArgumentValidator):
    @staticmethod
    def validate_type(value: Any) -> None:
        if not isinstance(value, bool):
            raise argparse.ArgumentTypeError("Value must be a boolean")

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type)


class ListArgumentValidator(ArgumentValidator):
    def __init__(self, element_validator: ArgumentValidator, max_length: int = float('inf'), allow_none: bool = False):
        super().__init__(allow_none=allow_none)
        self.element_validator = element_validator
        self.max_length = max_length

    @staticmethod
    def validate_type(value: list) -> None:
        if not isinstance(value, list):
            raise argparse.ArgumentTypeError("Value must be a list")

    def validate_length(self, value: list) -> None:
        if len(value) > self.max_length:
            raise argparse.ArgumentTypeError(f"List length must be lower than or equal to {self.max_length}")

    def validate_elements(self, value: list) -> None:
        for element in value:
            self.element_validator.validate(element)

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type, self.validate_length, self.validate_elements)


class DictionaryArgumentValidator(ArgumentValidator):
    def __init__(self, element_validator_dict: Dict[Any, ArgumentValidator], allow_none: bool = False):
        super().__init__(allow_none=allow_none)
        self.element_validator_dict = element_validator_dict

    @staticmethod
    def validate_type(value: dict) -> None:
        if not isinstance(value, dict):
            raise argparse.ArgumentTypeError("Value must be a dict")

    def validate_elements(self, value: dict) -> None:
        for ele_key, ele_value in value.items():
            if ele_key in self.element_validator_dict:
                self.element_validator_dict.get(ele_key).validate(ele_value)

    def create_validation_pipeline(self):
        super()._create_validation_pipeline(self.validate_type, self.validate_elements)


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.argument_validators = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, validator: ArgumentValidator | Dict[Any, ArgumentValidator] = None, 
                     **kwargs) -> argparse.Action:
        arguments = super().add_argument(*args, **kwargs)
        if validator is not None:
            self.argument_validators.update({arguments.dest: validator})
        return arguments

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        args_all = super().parse_args(args, namespace)
        ArgumentValidator.context = vars(args_all)
        for arg, value in vars(args_all).items():
            if arg in self.argument_validators:
                validator = self.argument_validators[arg]
                type_of_value = type(value)
                try:
                    if isinstance(validator, dict):
                        if type_of_value is list:
                            type_of_value_to_validate = type(value[0])
                        else:
                            type_of_value_to_validate = type_of_value
                        if type_of_value_to_validate in validator:
                            validator[type_of_value_to_validate].validate(value)
                        else:
                            raise argparse.ArgumentTypeError(f"Validation failed for argument '{arg}': \
                                type {type_of_value_to_validate} not supported")
                    else:
                        validator.validate(value)
                except argparse.ArgumentTypeError as e:
                    raise argparse.ArgumentTypeError(f"Validation failed for argument '{arg}': {e}")
        return args_all

    def update_argument(self, old_name: str, new_name: str = None, **kwargs) -> None:
        old_name = old_name.lstrip('-')
        if new_name:
            kwargs.update({'dest': new_name.lstrip('-')})
        for action in self._actions:
            if action.dest == old_name:
                for key, value in kwargs.items():
                    setattr(action, key, value)