# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import csv
import json
import argparse

import requests

from atb_llm.utils.file_utils import safe_open


MULTI_AUDIO_ROOT = "./examples/models/qwen2_audio/precision/multi_audio_data"
DATA_FIRST, DATA_SECOND = 'data_first.csv', 'data_second.csv'


class MultiAudioTest:
    def __init__(self, port, data_path, save_path):
        self.port = "http://127.0.0.1:" + str(port)
        self.data_path = data_path
        self.save_path = save_path
        self.method_list = self.get_method_list()
        self.multi_root = MULTI_AUDIO_ROOT
    
    @staticmethod
    def response_check(response):
        if response.status_code != 200:
            raise ValueError(f"Error response code: {response.status_code}")

    def get_method_list(self):
        method_list = [self.openai_requests, self.vllm_requests, self.tgi_requests]
        method_list += [self.trition_generate_requests]
        return method_list

    def get_audio_list(self, audio_path):
        audio_list = []
        with safe_open(audio_path, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                audio = os.path.join(self.data_path, row[0])
                if os.path.exists(audio):
                    audio_list.append(audio)
                else:
                    raise ValueError(f"{audio}不存在，请检查文件.")
        return audio_list
    
    def openai_requests(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        request_data = {
            "model": "qwen2_audio",
            "messages": messages,
            "max_tokens": 512,
            "do_sample": False,
            "stream": False
        }
        request_data = json.dumps(request_data)

        headers = {"Content-Type": "application/json"}
        port_body = "/v1/chat/completions"
        url = self.port + port_body
        response = requests.post(url=url, data=request_data, headers=headers, timeout=60)
        self.response_check(response)

        result = response.json()
        generate_text = result["choices"][0]["message"]["content"]
        return generate_text

    def trition_generate_requests(self, prompt):
        request_data = {
            "text_input": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens":256
            }
        }
        request_data = json.dumps(request_data)
        headers = {"Content-Type": "application/json"}
        port_body = "/v2/models/qwen2_audio/generate"
        url = self.port + port_body
        response = requests.post(url=url, data=request_data, headers=headers, timeout=60)
        self.response_check(response)

        result = response.json()
        generate_text = result["text_output"]
        return generate_text

    def tgi_requests(self, prompt):
        request_data = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "max_new_tokens":256
            }
        }
        request_data = json.dumps(request_data)

        headers = {"Content-Type": "application/json"}
        port_body = "/generate"
        url = self.port + port_body
        response = requests.post(url=url, data=request_data, headers=headers, timeout=60)
        self.response_check(response)

        result = response.json()
        generate_text = result["generated_text"]
        generate_text = generate_text.replace("<|im_end|>", "")
        return generate_text
    
    def vllm_requests(self, prompt):
        request_data = {
            "prompt": prompt,
            "max_tokens": 2500,
            "max_new_tokens":256,
            "do_sample": False,
            "stream": False
        }
        request_data = json.dumps(request_data)

        headers = {"Content-Type": "application/json"}
        port_body = "/generate"
        url = self.port + port_body
        response = requests.post(url=url, data=request_data, headers=headers, timeout=60)
        self.response_check(response)

        result = response.json()
        result = str(result)
        result = result.split(']')
        generate_text = result[-2][:-1].strip()
        generate_text = generate_text.replace(r"\'", "\'")
        return generate_text

    def process_requires(self, req_mode, prompt):
        mode_list = ['OpenAI', 'VLLM', 'TGI', 'TritionGenerate']
        if req_mode in mode_list:
            mode_idx = mode_list.index(req_mode)
            return self.method_list[mode_idx](prompt)
        else:
            raise ValueError("暂不支持")


    def create_requests(self, req_mode):
        data_first = os.path.join(self.multi_root, DATA_FIRST)
        audio_list_first = self.get_audio_list(data_first)
        data_second = os.path.join(self.multi_root, DATA_SECOND)
        audio_list_second = self.get_audio_list(data_second)
        file = safe_open(self.save_path, mode='w', encoding='utf-8')
        filenames = ["first_audio", "second_audio", "answer"]
        writer = csv.DictWriter(file, fieldnames=filenames)
        writer.writeheader()
        for idx, audio_first in enumerate(audio_list_first):
            audio_second = audio_list_second[idx]

            prompt_text1 = "What did the speaker say in the first audio? "
            prompt_text2 = "And what did the speaker say in the second audio?"
            file_type = "type"
            audio_url = "audio_url"
            prompt = [
                {file_type: audio_url, audio_url: f"{audio_first}"},
                {file_type: audio_url, audio_url: f"{audio_second}"},
                {file_type: "text", "text": prompt_text1 + prompt_text2}
            ]
            generate_text = self.process_requires(req_mode, prompt)
            answer_dict = {"first_audio": audio_first, "second_audio": audio_second, "answer": generate_text}
            writer.writerow(answer_dict)
            
            
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--port", required=True, type=int, help="port.")
    parser.add_argument("--req_method", required=True, type=str, help="req_method")
    parser.add_argument("--audio_path", required=True, type=str, help="audio path.")
    parser.add_argument("--save_path", required=True, type=str, help="save_path.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = args.save_path
    audio_path = args.audio_path
    port = args.port
    req_method = args.req_method
    multi_audio_test = MultiAudioTest(port=port, data_path=audio_path, save_path=save_path)
    multi_audio_test.create_requests(req_method)


if __name__ == "__main__":
    main()
