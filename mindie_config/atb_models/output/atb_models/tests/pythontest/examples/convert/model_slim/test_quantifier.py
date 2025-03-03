# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import unittest
from unittest.mock import patch
import argparse

from ddt import ddt

from examples.convert.model_slim.quantifier import parse_arguments, CPU


@ddt
class TestQuantifier(unittest.TestCase):
    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory'])
    def test_default_value(self):
        args = parse_arguments()
        self.assertIsNone(args.part_file_size)
        self.assertEqual(args.w_bit, 8)
        self.assertEqual(args.a_bit, 8)
        self.assertIsNone(args.disable_names)
        self.assertEqual(args.device_type, CPU)
        self.assertEqual(args.fraction, 0.01)
        self.assertEqual(args.act_method, 1)
        self.assertFalse(args.co_sparse, False)
        self.assertEqual(args.anti_method, "")
        self.assertEqual(args.disable_level, "L0")
        self.assertFalse(args.do_smooth, False)
        self.assertFalse(args.use_sigma, False)
        self.assertFalse(args.use_reduce_quant, False)
        self.assertEqual(args.tp_size, 1)
        self.assertEqual(args.sigma_factor, 3.0)
        self.assertFalse(args.is_lowbit, False)
        self.assertTrue(args.mm_tensor, True)
        self.assertTrue(args.w_sym, True)
        self.assertFalse(args.use_kvcache_quant, False)
        self.assertFalse(args.use_fa_quant, False)
        self.assertEqual(args.fa_amp, 0)
        self.assertTrue(args.open_outlier, True)
        self.assertEqual(args.group_size, 64)
        self.assertFalse(args.is_dynamic, False)
        self.assertEqual(args.input_ids_name, 'input_ids')
        self.assertEqual(args.attention_mask_name, 'attention_mask')
        self.assertEqual(args.tokenizer_args, '{}')
        self.assertTrue(args.disable_last_linear, True)
        self.assertIsNone(args.model_name)

    @patch('sys.argv', ['test_script.py', '--model_path', '', '--save_directory', 'fake_save_directory'])
    def test_model_path_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'A' * 4097, '--save_directory', 'fake_save_directory'])
    def test_model_path_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', ''])
    def test_save_directory_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'A' * 4097])
    def test_save_directory_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--input_ids_name', ''])
    def test_input_ids_name_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--input_ids_name', 'A' * 257])
    def test_input_ids_name_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--attention_mask_name', ''])
    def test_attention_mask_name_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--attention_mask_name', 'A' * 257])
    def test_attention_mask_name_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--tokenizer_args', 'A'])
    def test_tokenizer_args_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--tokenizer_args', 'A' * 4097])
    def test_tokenizer_args_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()
    
    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--model_name', ''])
    def test_model_name_too_short(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--model_name', 'A' * 257])
    def test_model_name_too_long(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_arguments()

    @patch('sys.argv', ['test_script.py', '--model_path', 'fake_model_path', '--save_directory', 'fake_save_directory',
                        '--model_name', 'A' * 256, '--input_ids_name', 'A', '--attention_mask_name', 'B',
                        '--tokenizer_args', '{"pad_token_id": 0}'])
    def test_valid_param(self):
        args = parse_arguments()
        self.assertEqual(args.model_path, 'fake_model_path')
        self.assertEqual(args.save_directory, 'fake_save_directory')
        self.assertEqual(args.model_name, 'A' * 256)
        self.assertEqual(args.input_ids_name, 'A')
        self.assertEqual(args.attention_mask_name, 'B')
        self.assertEqual(args.tokenizer_args, '{"pad_token_id": 0}')


if __name__ == '__main__':
    unittest.main()
