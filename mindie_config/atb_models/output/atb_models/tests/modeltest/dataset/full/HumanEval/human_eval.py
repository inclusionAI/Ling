# MIT License

# Copyright (c) 2023 Anton Bacaj

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict, Counter
from typing import List, Union, Iterable, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import os
import sys
import json
import contextlib
import tempfile
import platform
import faulthandler
import signal
import multiprocessing
import itertools
import shutil
import importlib.util
import tqdm
import numpy as np

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log.logging import logger


def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    with safe_open(filename, "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                yield json.loads(line)


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class TimeoutException(Exception):
    pass


class RedirectStdin:
    def __init__(self, new_target):
        self._new_target = new_target
        self._old_target = None

    def __enter__(self):
        self._old_target = sys.stdin
        sys.stdin = self._new_target
        return self._new_target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdin = self._old_target


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    original_unlink = os.unlink
    original_remove = os.remove
    
    def get_safe_unlink():
        return getattr(shutil, '_orig_unlink', original_unlink)
    
    def get_safe_remove():
        return getattr(shutil, '_orig_remove', original_remove)

    try:
        os.unlink = get_safe_unlink()
        os.remove = get_safe_remove()
        with tempfile.TemporaryDirectory() as dirname:
            yield dirname
    finally:
        os.unlink = original_unlink
        os.remove = original_remove


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with RedirectStdin(stream):
                yield


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


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    os.unlink = None 

    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys as sys_local
    sys_local.modules['ipdb'] = None
    sys_local.modules['joblib'] = None
    sys_local.modules['resource'] = None
    sys_local.modules['psutil'] = None
    sys_local.modules['tkinter'] = None


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """

    def unsafe_execute():
        with create_tempdir() as tmp_dir:
            completion_file = os.path.join(tmp_dir, "completion_code.py")
            with safe_open(completion_file, 'w') as f:
                f.write(problem["prompt"] + completion + "\n")
                f.write(problem["test"] + "\n")
                f.write(f"check({problem['entry_point']})")

            exec_globals = {}
            try:
                with time_limit(timeout):
                    if not os.path.exists(completion_file):
                        raise FileNotFoundError(f"Completion file {completion_file} not found.")
                    spec = importlib.util.spec_from_file_location("completion_code", completion_file)
                    if spec is None:
                        raise ImportError(f"Could not load spec from {completion_file}")
                    module = importlib.util.module_from_spec(spec)
                    if module is None:
                        raise ImportError(f"Could not load module from {completion_file}")
                    spec.loader.exec_module(module)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except FileNotFoundError as fnf_error:
                result.append(f"failed: File not found error - {fnf_error}")
            except ImportError as imp_error:
                result.append(f"failed: Import error - {imp_error}")
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
    )


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        if len(num_samples) != len(num_correct):
            raise ValueError("length of num_samples should be equal to length of num_correct")
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    with safe_open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode('utf-8'))


def evaluate_functional_correctness(
    sample_file: str,
    k: Tuple[int, int, int] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "../data/humaneval/human-eval.jsonl",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        logger.info("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) != len(problems):
            raise ValueError(
                "length of completion_id should be equal to length of problems, "
                "Some problems are not attempted."
            )

        logger.info("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct, passed_all = [], [], []
    results = {k: results[k] for k in sorted(results)}
    for result in results.values():
        passed_saved = [r[1]["passed"] for r in result]
        passed_all.extend(passed_saved)
        total.append(len(passed_saved))
        correct.append(sum(passed_saved))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file.split('infer.jsonl')[0] + "result.jsonl"
    logger.info("Writing results to %s.", out_file)
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))
    with safe_open(out_file, "ab") as fp:
        fp.write((json.dumps(pass_at_k) + "\n").encode('utf-8'))

    return pass_at_k, passed_all