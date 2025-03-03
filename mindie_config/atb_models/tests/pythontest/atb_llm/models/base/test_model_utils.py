# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import unittest
from unittest.mock import patch, Mock

from transformers import AutoTokenizer

# 待测函数
from atb_llm.models.base.model_utils import (
    EXTRA_EXP_INFO,
    filter_urls_from_error,
    safe_get_tokenizer_from_pretrained,
    safe_get_model_from_pretrained,
    safe_get_auto_model_from_pretrained,
    safe_get_auto_model_for_sequence_classification_from_pretrained,
    safe_get_config_from_pretrained,
    safe_get_config_dict,
    safe_from_pretrained,
    safe_open_clip_from_pretrained
)


class TestModelUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mocked_standardize_path = patch('atb_llm.utils.file_utils.standardize_path')
        cls.mocked_check_path_permission = patch('atb_llm.utils.file_utils.check_path_permission')
        cls.mocked_standardize_path.start()
        cls.mocked_check_path_permission.start()

    @classmethod
    def tearDownClass(cls):
        cls.mocked_standardize_path.stop()
        cls.mocked_check_path_permission.stop()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_filter_urls_from_error_case_exp_with_empty_args_result_success(self):
        error_message = Exception()

        filtered_error = filter_urls_from_error(error_message)

        self.assertEqual(filtered_error.args, ())
    
    def test_filter_urls_from_error_case_exp_with_no_url_result_success(self):
        arg = "Load tokenizer faild. Please check tokenizer files in model path."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        self.assertEqual(filtered_error.args, (arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_domain_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://huggingface.co for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_domain_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://huggingface.co:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_ipv4_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://255.200.100.0 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_ipv4_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://255.200.100.0:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_http_and_ipv6_url_result_success(self):
        arg = "Load tokenizer faild. Please visit http://[0000:1234:5678:90ab:cdef:90AB:CDEF:0000] " \
            + "for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit http*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_https_and_ipv6_url_result_success(self):
        arg = "Load tokenizer faild. Please visit https://[1234::5678]:1234 for more information."
        error_message = Exception(arg)
        
        filtered_error = filter_urls_from_error(error_message)

        filtered_arg = "Load tokenizer faild. Please visit https*** for more information."
        self.assertEqual(filtered_error.args, (filtered_arg,))
    
    def test_filter_urls_from_error_case_exp_with_multi_args_and_urls_result_success(self):
        arg0 = "Load tokenizer faild. Please check tokenizer files in model path."
        arg1 = "Load tokenizer faild. Please visit http://huggingface.co/abcd for more information."
        arg2 = "Load tokenizer faild. Please visit https://huggingface.co:1234/efgh for more information."
        arg3 = "Load tokenizer faild. Please visit http://255.200.100.0/ijkl for more information."
        arg4 = "Load tokenizer faild. Please visit https://255.200.100.0:1234/mnop for more information."
        arg5 = "Load tokenizer faild. Please visit http://[0000:1234:5678:90ab:cdef:90AB:CDEF:0000]/qrst " \
               + "for more information."
        arg6 = "Load tokenizer faild. Please visit https://[1234::5678]:1234/uvwx for more information."
        error_message = Exception(arg0, arg1, arg2, arg3, arg4, arg5, arg6)

        filtered_error = filter_urls_from_error(error_message)

        filtered_args = (
            "Load tokenizer faild. Please check tokenizer files in model path.",
            "Load tokenizer faild. Please visit http***/abcd for more information.",
            "Load tokenizer faild. Please visit https***/efgh for more information.",
            "Load tokenizer faild. Please visit http***/ijkl for more information.",
            "Load tokenizer faild. Please visit https***/mnop for more information.",
            "Load tokenizer faild. Please visit http***/qrst for more information.",
            "Load tokenizer faild. Please visit https***/uvwx for more information."
        )
        self.assertEqual(filtered_error.args, filtered_args)
    
    def validate_param_local_files_only(self, *args, **kwargs):
        if 'local_files_only' not in kwargs or kwargs['local_files_only'] is not True:
            return False
        return True
    
    def validate_param_local_files_only_return_two_value(self, *args, **kwargs):
        if 'local_files_only' not in kwargs or kwargs['local_files_only'] is not True:
            return False, False
        return True, True
    
    def test_safe_get_tokenizer_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            tokenizer = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertTrue(tokenizer)
    
    def test_safe_get_tokenizer_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_tokenizer_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_tokenizer_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_tokenizer_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_tokenizer_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_model_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_model_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_model_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_model_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_auto_model_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModel.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_auto_model_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            model = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertTrue(model)
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_for_sequence_classification_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_auto_model_for_sequence_classification_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_auto_model_for_sequence_classification_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_auto_model_for_sequence_classification_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_from_pretrained_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            config = safe_get_config_from_pretrained(model_path)
        
        self.assertTrue(config)
    
    def test_safe_get_config_from_pretrained_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_config_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_config_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_from_pretrained_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoConfig.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_config_from_pretrained(model_path)
        
        self.assertIn(
            f"{safe_get_config_from_pretrained.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_dict_case_success(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only_return_two_value
            config = safe_get_config_dict(model_path)
        
        self.assertTrue(config)
    
    def test_safe_get_config_dict_case_raise_environment_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_get_config_dict(model_path)
        
        self.assertIn(
            f"{safe_get_config_dict.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_get_config_dict_case_raise_value_error(self):
        model_path = "/home/data/llama2-7b"

        with patch("transformers.configuration_utils.PretrainedConfig.get_config_dict") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_get_config_dict(model_path)
        
        self.assertIn(
            f"{safe_get_config_dict.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_from_pretrained_case_success(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = self.validate_param_local_files_only
            tokenizer = safe_from_pretrained(target_cls, model_path)
        
        self.assertTrue(tokenizer)
    
    def test_safe_from_pretrained_case_raise_environment_error(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = EnvironmentError
            with self.assertRaises(EnvironmentError) as context:
                _ = safe_from_pretrained(target_cls, model_path)
        
        self.assertIn(
            f"Get instance from {target_cls.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_from_pretrained_case_raise_value_error(self):
        target_cls = AutoTokenizer
        model_path = "/home/data/llama2-7b"

        with patch("transformers.AutoTokenizer.from_pretrained") as mocked_from_pretrained:
            mocked_from_pretrained.side_effect = ValueError
            with self.assertRaises(ValueError) as context:
                _ = safe_from_pretrained(target_cls, model_path)
        
        self.assertIn(
            f"Get instance from {target_cls.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_open_clip_from_pretrained_case_model_name_illegal_raise_value_error(self):
        open_clip_method = Mock(__name__='open_clip_method')
        model_name = 'hf-hub:llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(ValueError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            "Model name should not start with hf-hub: to avoid internet connection.", str(context.exception.__cause__))
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))

    def test_safe_open_clip_from_pretrained_case_raise_environment_error(self):
        open_clip_method = Mock(side_effect=EnvironmentError, __name__='open_clip_method')
        model_name = 'llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(EnvironmentError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))
    
    def test_safe_open_clip_from_pretrained_case_raise_value_error(self):
        open_clip_method = Mock(side_effect=ValueError, __name__='open_clip_method')
        model_name = 'llama'
        model_path = "/home/data/llama2-7b"

        with self.assertRaises(ValueError) as context:
            safe_open_clip_from_pretrained(open_clip_method, model_name, model_path)
        
        self.assertIn(
            f"Get instance from {open_clip_method.__name__} failed. " \
            + EXTRA_EXP_INFO, str(context.exception))


if __name__ == '__main__':
    unittest.main()
