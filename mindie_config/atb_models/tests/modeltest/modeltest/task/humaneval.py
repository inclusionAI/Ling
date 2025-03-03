# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
import json
import io
import contextlib
import tempfile
import signal
import multiprocessing
import itertools
import importlib.util
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from modeltest.metric.pass_k import PassKMetric
from atb_llm.utils.log.logging import logger
from atb_llm.utils.file_utils import safe_open
from .precision_task import PrecisionTask


class HumanEvalPrecisionTask(PrecisionTask):
    def __init__(self, task_config):
        super().__init__(task_config)
        self.entry = ""
        self.all_results = []  # New list
    
    def prepare_data(self, metric):
        datasets_input = []
        for sub_dataset_name in tqdm(self.task_config.subject_mapping.keys()):
            self.entry = os.path.join(
                        self.local_dataset_folder,
                        self.task_config.local_dataset_path,
                        sub_dataset_name)
            if isinstance(metric, PassKMetric):
                metric.correct_num_list.append(0)  # Initialize correct count
            dataset = []
            with safe_open(self.entry, encoding='utf-8') as f:
                for idx_in_dataset, line in enumerate(f):
                    line_json = json.loads(line)
                    line_json['idx_in_dataset'] = idx_in_dataset
                    dataset.append(line_json)
            datasets_input.append(dataset)
        return datasets_input

    def build_queries(self, sub_dataset_idx, batched_data, model_config):
        return [query['prompt'].strip() for query in batched_data]
    
    def result_judge(self, metric, generate_token_lists, _, sub_dataset_idx, batched_data):
        samples = []
        task_ids = []
        logger.info(f"Processing {len(generate_token_lists)} generated texts.")
        for idx, completion in enumerate(generate_token_lists):
            task_id = batched_data[idx]['task_id']
            response_cleaned_up = self.cleanup_code(completion)
            sample = {
                'task_id': task_id,
                'completion': response_cleaned_up,
            }
            task_ids.append(task_id)
            samples.append(sample)
            metric.csv_debug['test_result'].append(response_cleaned_up)
            metric.csv_debug['golden_result'].append([])

        self.evaluate_functional_correctness(
            samples=samples,
            task_ids=task_ids,
            n_workers=4,
            timeout=3.0,
            problem_file=self.entry
        )

        k_value = self.task_config.metric.get('k', 1)
        logger.info(f"Using k value: {k_value}")
        
        if len(self.all_results) == metric.case_num_list[-1]:
            pass_at_k, passed_all = self.calculate_pass_k(ks=(k_value,))
            metric.csv_debug['pass'] = passed_all
            correct_count = sum(passed_all)
            metric.correct_num_list[sub_dataset_idx] += correct_count
            metric.correct_num += correct_count
            logger.info(f"Functional correctness results: {pass_at_k}")
            result_file = os.path.join(metric.result_dir, 'humaneval_infer_results.jsonl')
            with safe_open(result_file, 'w', encoding='utf-8') as f:
                for result in self.all_results:
                    f.write(json.dumps(result) + '\n')
            logger.info(f"All inference results saved to {result_file}")

    def evaluate_functional_correctness(
        self, 
        samples: List[Dict], 
        task_ids: list, 
        n_workers: int = 4, 
        timeout: float = 3.0, 
        problem_file: str = "./data/humaneval/human-eval.jsonl"
    ):
        problems = self.read_problems(problem_file, task_ids)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            logger.info("Reading samples...")
            for sample in samples:
                task_id = sample["task_id"]
                completion = sample["completion"]
                args = (problems[task_id], completion, timeout, completion_id[task_id])
                future = executor.submit(self.check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
            if len(completion_id) != len(problems):
                raise ValueError(
                    "length of completion_id should be equal to length of problems, "
                    "Some problems are not attempted."
                )
            logger.info("Running test suites...")
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                self.all_results.append(result)
                logger.info(f"Collected result for task_id: {result['task_id']}")

    def check_correctness(
        self, 
        problem: Dict, 
        completion: str, 
        timeout: float, 
        completion_id: Optional[int] = None
    ) -> Dict:

        def unsafe_execute():
            with create_tempdir() as tmp_dir:
                reliability_guard()
                completion_file = os.path.join(tmp_dir, "completion_code.py")
                with safe_open(completion_file, 'w', encoding='utf-8') as f:
                    f.write(problem["prompt"] + completion + "\n")
                    f.write(problem["test"] + "\n")
                    f.write(f"check({problem['entry_point']})")
                exec_globals = {}
                try:
                    with swallow_io():
                        with time_limit(timeout):
                            spec = importlib.util.spec_from_file_location("completion_code", completion_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                        result.append("passed")
                except TimeoutException:
                    result.append("timed out")
                except BaseException as e:
                    result.append(f"failed: {e}")

        manager = multiprocessing.Manager()
        result = manager.list()
        p = multiprocessing.Process(target=unsafe_execute)
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()
        if not result:
            result.append("timed out")

        return dict(
            task_id=problem["task_id"],
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
            completion=completion
        )

    def calculate_pass_k(self, ks: Tuple[int, ...] = (1,)):
        total, correct, passed_all = [], [], []
        sorted_results = sorted(self.all_results, key=lambda x: x['task_id'])
        
        for result in sorted_results:
            passed_saved = [result["passed"]]
            passed_all.extend(passed_saved)
            total.append(len(passed_saved))
            correct.append(sum(passed_saved))

        total = np.array(total)
        correct = np.array(correct)

        pass_at_k = {}
        for k in ks:
            if k is not None and (total >= k).all():
                pass_at_k[f"pass@{k}"] = self.estimate_pass_at_k(total, correct, k).mean()

        return pass_at_k, passed_all

    def estimate_pass_at_k(
        self, 
        num_samples: Union[int, List[int], np.ndarray], 
        num_correct: Union[List[int], np.ndarray], 
        k: int
    ) -> np.ndarray:

        def estimator(n: int, c: int, k: int) -> float:
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
        if isinstance(num_samples, int):
            num_samples_it = itertools.repeat(num_samples, len(num_correct))
        else:
            if len(num_samples) != len(num_correct):
                raise ValueError("length of num_samples should be equal to length of num_correct")
            num_samples_it = iter(num_samples)
        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

    def read_problems(self, evalset_file: str, task_ids: list) -> Dict[str, Dict]:
        return {task["task_id"]: task for task in self.stream_jsonl(evalset_file) if task["task_id"] in task_ids}

    def stream_jsonl(self, filename: str) -> Iterable[Dict]:
        with safe_open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

    def cleanup_code(self, code: str) -> str:
        code_splits = code.split("\n")
        is_empty_line = False
        ind_empty_line = None
        for i, line in enumerate(code_splits):
            if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                is_empty_line = True
                ind_empty_line = i
                break
        if is_empty_line:
            code = "\n".join(code_splits[:ind_empty_line])
        else:
            end_words = ["\ndef", "\nclass", "\n#", "\nassert", '\n"""', "\nprint", "\nif", "\n\n\n"]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
        return code


class TimeoutException(Exception):
    pass


# Reliability Guard and Timeout Management
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        yield dirname


@contextlib.contextmanager
def swallow_io():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    import platform
    import resource
    if maximum_memory_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    import builtins
    builtins.exit = None
    builtins.quit = None