# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from dataclasses import dataclass
from atb_llm.models.base.router import BaseRouter
from atb_llm.models.bunny.qwen2.modeling_bunny import BunnyConfig as BunnyConfigQwen2
from atb_llm.models.bunny.minicpm.modeling_bunny import BunnyConfig as BunnyConfigMiniCPM
from ..base.model_utils import safe_get_tokenizer_from_pretrained


@dataclass
class BunnyRouter(BaseRouter):
    @property
    def model_version(self):
        """
        次级模型名称，比如v2_13b
        :return:
        """
        if self.config_dict['model_type'] == "bunny-qwen2":  # 只有13b才是40层，同时兼容 v1 v2
            model_ver = "qwen2"
        else:
            model_ver = "minicpm"
        return model_ver

    def get_config(self):
        if self.config_dict['model_type'] == "bunny-qwen2":
            bunny_config = BunnyConfigQwen2
        else:
            bunny_config = BunnyConfigMiniCPM

        config = bunny_config.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code
        )
        self.checkout_config(config)
        return config

    def get_tokenizer(self):
        return safe_get_tokenizer_from_pretrained(
            self.model_name_or_path,
            padding_side="left",
            trust_remote_code=False
        )

    def checkout_config(self, config):
        super().check_config(config)
        attribute_ranges = {
            'attention_dropout': (0, 2147483647),
            'max_window_layers': (1, 2147483647),
            'num_key_value_heads': (1, 2147483647),
            'rope_theta': (1, 2147483647),
            'sliding_window': (1, 2147483647)
        }
        for attr, (min_val, max_val) in attribute_ranges.items():
            if not hasattr(config, attr):
                continue
            value = getattr(config, attr)
            if value < min_val or value > max_val:
                raise ValueError(f"self._config.{attr} must be between {min_val} and {max_val}")