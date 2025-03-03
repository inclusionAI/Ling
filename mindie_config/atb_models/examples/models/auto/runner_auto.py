# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
from collections import OrderedDict


class AutoRunnerClass:
    _runner_mapping = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` "
        )

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model_name = kwargs.pop("model_name", None)
        runner_class = cls._runner_mapping[model_name]
        kwargs["model_path"] = model_path
        return runner_class(
            **kwargs
        )

    @classmethod
    def register(cls, model_name, runner_class):
        """
        Register a new runner for this model name.

        Args:
            model_class ([`clip`]):
                The model to register.
            runner_class ([`MultiModalRunner`]):
                The runner to register.
        """
        cls._runner_mapping.register(model_name, runner_class)


class _LazyAutoMapping(OrderedDict):
    """
    Args:
        - runner_mapping: The map model name to runner class
    """

    def __init__(self, runner_mapping):
        super().__init__()
        self._runner_mapping = runner_mapping
        self.register_runner = {}
        self._modules = {}

    def __len__(self):
        common_keys = set(self._runner_mapping.keys())
        return len(common_keys) + len(self.register_runner)

    def __getitem__(self, key):
        if key in self.register_runner:
            return self.register_runner[key]
        if key in self._runner_mapping:
            runner_path = self._runner_mapping[key]
            return self.load_runner_from_module(key, runner_path)
        raise KeyError(key)

    def load_runner_from_module(self, model_name, runner_path):
        if runner_path is None:
            raise ValueError(f"runner of {model_name} is None")
        runner_dir_name = "examples"
        if len(runner_path) == 1:
            if runner_path[0] == "PARunner":
                module_path = f"{runner_dir_name}.run_pa"
            elif runner_path[0] == "FARunner":
                module_path = f"{runner_dir_name}.run_fa"
        else:
            module_path = f"{runner_dir_name}.models.{model_name}.{runner_path[0]}"
        module = importlib.import_module(module_path)

        try:
            return getattr(module, runner_path[-1])
        except ValueError as e:
            raise ValueError(f"Could not find {runner_path[-1]} neither in {module}!") from e

    def keys(self):
        mapping_keys = []
        for key, name in self._runner_mapping.items():
            if self.load_runner_from_module(key, name):
                mapping_keys.append(key)
        return mapping_keys + list(self.register_runner.keys())

    def values(self):
        mapping_values = []
        for key, name in self._runner_mapping.items():
            if self.load_runner_from_module(key, name):
                mapping_values.append(key)
        return mapping_values + list(self.register_runner.values())

    def items(self):
        mapping_items = []
        for key, name in self._runner_mapping.items():
            if self.load_runner_from_module(key, name):
                mapping_items.append((key, name))
        return mapping_items + list(self.register_runner.items())

    def register(self, key, value):
        """
        Register a new runner in this mapping.
        """
        if key in self._runner_mapping.keys():
            raise ValueError(f"'{key}' is already used.")

        self.register_runner[key] = value


RUNNER_MAPPING_NAMES = OrderedDict(
    [
        # Base runner mapping
        ("clip", ["run", "MultiModalRunner"]),
        ("llava", ["llava", "MultiModalPARunner"]),
        ("qwen_vl", ["run_pa", "PARunner"]),
        ("llama", ["PARunner"]),
    ]
)

RUNNER_MAPPING = _LazyAutoMapping(RUNNER_MAPPING_NAMES)


class AutoRunner(AutoRunnerClass):
    _runner_mapping = RUNNER_MAPPING
