# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import logging
import json
import argparse

from atb_llm.utils.file_utils import safe_open


class CalculateAcc(object):
    def __init__(self, eval_npu, eval_gpu, sampling_answer) -> None:
        self.eval_npu = eval_npu
        self.eval_gpu = eval_gpu
        self.sampling_answer = sampling_answer
        self.calculate()

    def check_answer(self, out_put, answer_list, position=-1):
        for answer in answer_list:
            if answer in out_put:
                return answer[position]
        return False

    def result_load(self, gpu_result):
        short_num = 0
        result_map = {}
        for img_id in gpu_result:
            item = gpu_result[img_id]
            out_put = item['output_sequence'][0]
            for _ in range(10):
                out_put = out_put.replace('  ', ' ')
            if '</s>' in out_put:
                out_put = out_put.replace('</s>', '')
            if '<|im_end|>' in out_put:
                out_put = out_put.replace('<|im_end|>', '')
            answer_list1 = ['A', 'B', 'C', 'D', 'E']
            answer_list2 = ['is A', 'is B', 'is C', 'is D', 'is E']
            answer_list3 = [' A.', ' B.', ' C.', ' D.', ' E.']
            answer = False
            if len(out_put) < 5:
                short_num += 1
                if self.check_answer(out_put, answer_list1):
                    answer = self.check_answer(out_put, answer_list1)
            elif self.check_answer(out_put, answer_list2):
                answer = self.check_answer(out_put, answer_list2)
            elif self.check_answer(out_put, answer_list3):
                answer = self.check_answer(out_put, answer_list3, position=1)
            if answer:
                result_map[img_id] = answer
        return result_map

    def npu_result_load(self):
        with safe_open(self.eval_npu, 'r') as f:
            npu_result = json.load(f)
            npu_result = self.result_load(npu_result)
        return npu_result

    def gpu_result_load(self):
        with safe_open(self.eval_gpu, 'r') as f:
            gpu_result = json.load(f)
            gpu_result = self.result_load(gpu_result)
        return gpu_result

    def correct_result_load(self):
        answer_map = {}
        with safe_open(self.sampling_answer, 'r') as f:
            sampling_answer_map = json.load(f)
        for video in sampling_answer_map:
            for video_id in sampling_answer_map[video]:
                answer = sampling_answer_map[video][video_id]["answer"]
                answer = self.check_answer(answer, ['A', 'B', 'C', 'D', 'E'])
                answer_map[video_id] = answer
        return answer_map

    def calculate(self):
        gpu_result = self.gpu_result_load()
        npu_result = self.npu_result_load()
        answer_map = self.correct_result_load()
        gpu_total_num, npu_total_num, gpu_right_num, npu_right_num = 0, 0, 0, 0
        for video_id in answer_map:
            if video_id not in gpu_result or video_id not in npu_result:
                continue
            answer = answer_map[video_id]
            if video_id in gpu_result:
                gpu_total_num += 1
                if gpu_result[video_id] == answer:
                    gpu_right_num += 1
            if video_id in npu_result:
                npu_total_num += 1
                if npu_result[video_id] == answer:
                    npu_right_num += 1
        logging.info("gpu %.4f ", gpu_right_num / gpu_total_num)
        logging.info("npu %.4f ", npu_right_num / npu_total_num)


parser = argparse.ArgumentParser()
parser.add_argument('--eval_npu_path', type=str, default='eval_npu.json')
parser.add_argument('--eval_gpu_path', type=str, default='eval_gpu.json')
parser.add_argument('--sampling_answer_path', type=str, default='sampling_answer.json')

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
calculate = CalculateAcc(args.eval_npu_path, args.eval_gpu_path, args.sampling_answer_path)
