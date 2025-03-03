# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import stat
import unittest
from unittest.mock import patch

# 待测函数
from atb_llm.utils.file_utils import (
    safe_open,
    standardize_path,
    check_path_is_none,
    is_path_exists,
    check_path_is_link,
    check_path_length_lt,
    check_file_size_lt,
    check_owner,
    check_other_write_permission,
    check_path_permission,
    check_file_safety,
    safe_listdir,
    safe_chmod,
    has_owner_write_permission,
    safe_readlines
)


class TestFileUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))

        cls.file_path = os.path.join(cls.test_dir, "test.txt")
        with safe_open(cls.file_path, 'w', encoding='utf-8', permission_mode=0o640) as fw:
            fw.write("This is a test file." + "\n")
        
        cls.symbolic_link = os.path.join(cls.test_dir, "link_to_test.txt")
        if os.path.lexists(cls.symbolic_link):
            os.unlink(cls.symbolic_link)
        os.symlink(cls.file_path, cls.symbolic_link)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.file_path)
        os.unlink(cls.symbolic_link)

    def setUp(self):
        with safe_open(self.file_path, 'w', encoding='utf-8', permission_mode=0o640) as fw:
            fw.write("This is a test file." + "\n")

    def tearDown(self):
        pass

    def test_safe_open_case_read_file_result_success(self):
        file_path = self.file_path
        mode = 'r'
        encoding = 'utf-8'
        permission_mode = 0o640
        is_exist_ok = True

        file_context = []
        with safe_open(file_path, mode, encoding, permission_mode, is_exist_ok) as fr:
            for line in fr.readlines():
                file_context.append(line)
        
        self.assertEqual(len(file_context), 1)

    def test_safe_open_case_append_file_result_success(self):
        file_path = self.file_path
        mode = 'a'
        encoding = 'utf-8'
        permission_mode = 0o640
        is_exist_ok = True

        max_path_length = 1024
        max_file_size = 1024 * 1024
        check_link = True

        file_context = []
        with safe_open(file_path, mode, encoding, permission_mode, is_exist_ok, \
                       max_path_length=max_path_length, max_file_size=max_file_size, check_link=check_link) as fa:
            fa.write("Test safe_open success." + "\n")
        mode = 'r'
        with safe_open(file_path, mode, encoding, permission_mode, is_exist_ok, \
                       max_path_length=max_path_length, max_file_size=max_file_size, check_link=check_link) as fr:
            for line in fr.readlines():
                file_context.append(line)
        
        self.assertEqual(len(file_context), 2)

    def test_standardize_path_case_input_relative_path_result_success(self):
        path = os.path.basename(self.file_path)
        max_path_length = 1024
        check_link = True

        origin_working_dir = os.getcwd()
        os.chdir(self.test_dir)
        real_path = standardize_path(path, max_path_length, check_link)
        os.chdir(origin_working_dir)

        self.assertEqual(real_path, self.file_path)
    
    def test_standardize_path_case_input_symbolic_link_result_success(self):
        path = self.symbolic_link
        max_path_length = 1024
        check_link = False

        real_path = standardize_path(path, max_path_length, check_link)

        self.assertEqual(real_path, self.file_path)
    
    def test_standardize_path_case_input_real_path_result_success(self):
        path = self.file_path
        max_path_length = 1024
        check_link = True

        real_path = standardize_path(path, max_path_length, check_link)

        self.assertEqual(real_path, self.file_path)
    
    def test_is_path_exists_case_exists_result_true(self):
        path = self.file_path

        result = is_path_exists(path)

        self.assertTrue(result)
    
    def test_is_path_exists_case_not_exists_result_false(self):
        path = os.path.join(self.test_dir, "a_not_exists_file.txt")

        result = is_path_exists(path)

        self.assertFalse(result)

    def test_check_path_is_none_case_input_none_result_raise_type_error(self):
        file_path = None

        with self.assertRaises(TypeError) as context:
            check_path_is_none(file_path)
        
        self.assertIn("The file path should not be None.", str(context.exception))
    
    def test_check_path_is_none_case_input_not_none_result_success(self):
        file_path = self.file_path

        check_path_is_none(file_path)
    
    def test_check_path_is_link_case_not_symbolic_link_result_success(self):
        path = self.file_path

        check_path_is_link(path)
    
    def test_check_path_is_link_case_is_symbolic_link_result_raise_value_error(self):
        path = self.symbolic_link

        with self.assertRaises(ValueError) as context:
            check_path_is_link(path)
        
        self.assertIn("The path should not be a symbolic link file.", str(context.exception))
    
    def test_check_path_length_lt_case_less_than_max_path_length_result_success(self):
        path = self.file_path
        max_path_length = 1024

        check_path_length_lt(path, max_path_length)
    
    def test_check_path_length_lt_case_greater_than_max_path_length_result_raise_value_error(self):
        path = self.file_path
        max_path_length = 1

        with self.assertRaises(ValueError) as context:
            check_path_length_lt(path, max_path_length)
        
        self.assertIn("The length of path should not be greater than", str(context.exception))
    
    def test_check_file_size_lt_case_less_than_max_file_size_result_success(self):
        path = self.file_path
        max_file_size = 1024 * 1024

        check_file_size_lt(path, max_file_size)
    
    def test_check_file_size_lt_case_greater_than_max_file_size_result_raise_value_error(self):
        path = self.file_path
        max_file_size = 1

        with self.assertRaises(ValueError) as context:
            check_file_size_lt(path, max_file_size)
        
        self.assertIn("The size of file should not be greater than", str(context.exception))
    
    def test_check_owner_case_the_cur_user_is_root_result_success(self):
        path = self.file_path

        with patch("os.geteuid") as mocked_geteuid, \
             patch("os.getgid") as mocked_getgid, \
             patch("os.stat") as mocked_stat:
            path_stat = os.stat_result((0, 0, 0, 0, 1001, 1001, 0, 0, 0, 0))
            mocked_geteuid.return_value = 0
            mocked_getgid.return_value = 0
            mocked_stat.return_value = path_stat
            check_owner(path)
    
    def test_check_owner_case_the_cur_user_is_path_owner_result_success(self):
        path = self.file_path

        with patch("os.geteuid") as mocked_geteuid, \
             patch("os.getgid") as mocked_getgid, \
             patch("os.stat") as mocked_stat:
            path_stat = os.stat_result((0, 0, 0, 0, 1001, 1001, 0, 0, 0, 0))
            mocked_geteuid.return_value = 1001
            mocked_getgid.return_value = 1002
            mocked_stat.return_value = path_stat
            check_owner(path)
    
    def test_check_owner_case_the_cur_user_and_path_owner_in_same_user_group_result_success(self):
        path = self.file_path

        with patch("os.geteuid") as mocked_geteuid, \
             patch("os.getgid") as mocked_getgid, \
             patch("os.stat") as mocked_stat:
            path_stat = os.stat_result((0, 0, 0, 0, 1001, 1001, 0, 0, 0, 0))
            mocked_geteuid.return_value = 1002
            mocked_getgid.return_value = 1001
            mocked_stat.return_value = path_stat
            check_owner(path)
    
    def test_check_owner_case_the_other_user_result_raise_permission_error(self):
        path = self.file_path

        with patch("os.geteuid") as mocked_geteuid, \
             patch("os.getgid") as mocked_getgid, \
             patch("os.stat") as mocked_stat:
            path_stat = os.stat_result((0, 0, 0, 0, 1001, 1001, 0, 0, 0, 0))
            mocked_geteuid.return_value = 1002
            mocked_getgid.return_value = 1002
            mocked_stat.return_value = path_stat
            with self.assertRaises(PermissionError) as context:
                check_owner(path)
            
        self.assertIn("The current user does not have permission to access the path:", str(context.exception))
    
    def test_check_other_write_permission_case_not_have_other_write_permission_result_success(self):
        file_path = self.file_path

        check_other_write_permission(file_path)
    
    def test_check_other_write_permission_case_have_other_write_permission_result_raise_permission_error(self):
        file_path = self.file_path

        os.chmod(file_path, 0o642)
        with self.assertRaises(PermissionError) as context:
            check_other_write_permission(file_path)
        
        os.chmod(file_path, 0o640)
        self.assertIn("The file should not be writable by others who are neither the owner nor in the group.", \
                      str(context.exception))
    
    def test_check_path_permission_case_do_check_result_success(self):
        file_path = self.file_path

        check_path_permission(file_path)
    
    def test_check_file_safety_case_file_exists_result_success(self):
        file_path = self.file_path
        mode = 'r'
        is_exist_ok = True
        max_file_size = 1024
        is_check_file_size = True

        check_file_safety(file_path, mode, is_exist_ok, max_file_size, is_check_file_size)
    
    def test_check_file_safety_case_file_exists_but_not_is_exists_ok_result_raise_file_exists_error(self):
        file_path = self.file_path
        mode = 'r'
        is_exist_ok = False
        max_file_size = 1024
        is_check_file_size = True

        with self.assertRaises(FileExistsError) as context:
            check_file_safety(file_path, mode, is_exist_ok, max_file_size, is_check_file_size)
        
        self.assertIn("The file is expected not to exist, but it already does.", str(context.exception))
    
    def test_check_file_safety_case_file_not_exists_result_success(self):
        file_path = os.path.join(self.test_dir, "a_not_exists_file.txt")
        mode = 'w'
        is_exist_ok = True
        max_file_size = 1024
        is_check_file_size = True

        check_file_safety(file_path, mode, is_exist_ok, max_file_size, is_check_file_size)
    
    def test_check_file_safety_case_file_not_exists_but_mode_is_read_result_raise_file_not_found_error(self):
        file_path = os.path.join(self.test_dir, "a_not_exists_file.txt")
        mode = 'r'
        is_exist_ok = True
        max_file_size = 1024
        is_check_file_size = True

        with self.assertRaises(FileNotFoundError) as context:
            check_file_safety(file_path, mode, is_exist_ok, max_file_size, is_check_file_size)
        
        self.assertIn("The file is expected to exist, but it does not.", str(context.exception))
    
    def test_safe_listdir_case_file_num_not_exceeds_result_success(self):
        file_path = self.test_dir
        max_file_num = 1024

        file_list = safe_listdir(file_path, max_file_num)

        self.assertListEqual(file_list, os.listdir(file_path))
    
    def test_safe_listdir_case_file_num_exceeds_result_raise_value_error(self):
        file_path = self.test_dir
        max_file_num = 1

        with self.assertRaises(ValueError) as context:
            _ = safe_listdir(file_path, max_file_num)

        self.assertIn("The file num in dir is", str(context.exception))
        self.assertIn("which exceeds the limit", str(context.exception))
    
    def test_safe_chmod_case_chmod_ok_result_success(self):
        file_path = self.file_path
        permission_mode = 0o600

        safe_chmod(file_path, permission_mode)
        file_stat = os.stat(file_path)
        mode = stat.S_IMODE(file_stat.st_mode)
        os.chmod(file_path, 0o640)

        self.assertEqual(mode, permission_mode)
    
    def test_has_owner_write_permission_case_has_result_true(self):
        file_path = self.file_path
        
        has_permission = has_owner_write_permission(file_path)

        self.assertTrue(has_permission)
    
    def test_has_owner_write_permission_case_not_has_result_false(self):
        file_path = self.file_path
        
        os.chmod(file_path, 0o440)
        has_permission = has_owner_write_permission(file_path)
        os.chmod(file_path, 0o640)

        self.assertFalse(has_permission)
    
    def test_safe_readlines_case_line_num_not_exceeds_result_success(self):
        max_line_num = 1024 * 1024

        with safe_open(self.file_path, 'r', encoding='utf-8', permission_mode=0o640) as fr:
            lines = safe_readlines(fr, max_line_num)
        with safe_open(self.file_path, 'r', encoding='utf-8', permission_mode=0o640) as fr:
            lines_golden = fr.readlines()
        
        self.assertListEqual(lines, lines_golden)
    
    def test_safe_readlines_case_line_num_exceeds_result_raise_value_error(self):
        max_line_num = 0

        with safe_open(self.file_path, 'r', encoding='utf-8', permission_mode=0o640) as fr:
            with self.assertRaises(ValueError) as context:
                _ = safe_readlines(fr, max_line_num)
        
        self.assertIn("The file line num is", str(context.exception))
        self.assertIn("which exceeds the limit", str(context.exception))


if __name__ == '__main__':
    unittest.main()
