# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from io import BytesIO
import argparse
import os
import json
import logging
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from atb_llm.utils.file_utils import safe_open, safe_listdir, standardize_path, check_file_safety
from atb_llm.models.base.model_utils import safe_from_pretrained


MAX_AUDIO_NUM = 30000


def get_init(gpu_idx, model_path):
    model = safe_from_pretrained(Qwen2AudioForConditionalGeneration, model_path)
    model = model.to(f'cuda:{gpu_idx}')
    processor = safe_from_pretrained(AutoProcessor, model_path)

    prompt_head = "<|audio_bos|><|AUDIO|><|audio_eos|>In this audio, what kind of sound can you hear? "
    prompt_body1 = "A: Laughter, B: Sigh, C: Cough, D: Throat clearing, E: Sneeze, F: Sniff, "
    prompt_body2 = "Please select the one closest to the correct answer. ASSISTANT:"
    prompt = prompt_head + prompt_body1 + prompt_body2
    return model, processor, prompt


def load_audio(audio_path):
    with safe_open(audio_path, 'rb') as file:
        content = file.read()
    return content


def run_single(audio_path, gpu_idx, model, processor, prompt):
    audio, sr = librosa.load(BytesIO(load_audio(audio_path)), sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=prompt, audios=audio, return_tensors="pt")
    inputs = inputs.to(f'cuda:{gpu_idx}')

    generated_ids = model.generate(**inputs, max_length=256)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def run_batch(model_path, audio_dir, gpu_idx, predict_path):
    model, processor, prompt = get_init(gpu_idx, model_path)
    audio_dir = standardize_path(audio_dir)
    check_file_safety(audio_dir)
    audio_list = safe_listdir(audio_dir)
    if len(audio_list) > MAX_AUDIO_NUM:
        audio_list = audio_list[:MAX_AUDIO_NUM]
    res_map = {}
    for idx, audio in enumerate(audio_list):
        try:
            audio_path = os.path.join(audio_dir, audio)
            response = run_single(audio_path, gpu_idx, model, processor, prompt)
            res_map[audio] = response
        except Exception as e:
            logging.info('error audio data idx: %d, %s', idx, e)
    with safe_open(predict_path, 'w', encoding='utf-8') as f:
        json.dump(res_map, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", required=True, type=str, help="model path.")
    parser.add_argument("--audio_path", required=True, type=str, help="audio path.")
    parser.add_argument("--gpu_idx", required=True, type=int, help="gpu pidx.")
    parser.add_argument("--predict_path", required=True, type=str, help="predict json path.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    audio_path = args.audio_path
    gpu_idx = args.gpu_idx
    predict_path = args.predict_path
    run_batch(model_path, audio_path, gpu_idx, predict_path)


if __name__ == "__main__":
    main()
