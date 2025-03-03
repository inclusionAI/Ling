# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import argparse

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log import logger


def mood_map2():
    choose_map = {
        "A": "laughter", 
        "B": "sigh", 
        "C": "cough", 
        "D": "throatclearing", 
        "E": "sneeze", 
        "F": "sniff"
    }
    map_reverse = {}
    for key in choose_map:
        map_reverse[choose_map[key].lower()] = key
    return choose_map, map_reverse


def mood_map():
    choose_map = {
        "A": "Laughter", 
        "B": "Sigh", 
        "C": "Cough", 
        "D": "Throat clearing", 
        "E": "Sneeze", 
        "F": "Sniff"
    }
    map_reverse = {}
    for key in choose_map:
        map_reverse[choose_map[key].lower()] = key
    return choose_map, map_reverse


def process_predict(predict_result):
    choose_map, _ = mood_map()
    for key in predict_result:
        for choose in choose_map:
            choose_ = choose + '.'
            
            if choose_map[choose].lower() in predict_result[key].lower():
                predict_result[key] = choose
                break
            if choose_ in predict_result[key]:
                predict_result[key] = choose
                break
            if choose in predict_result[key]:
                predict_result[key] = choose
                break

    return predict_result


def compare(predict_result):
    _, map_reverse = mood_map2()
    total_num, right_num = 0, 0

    for audio_id in predict_result:
        for key in map_reverse:
            if key in audio_id:
                pre = predict_result[audio_id]
                label = map_reverse[key]
                if pre == label:
                    right_num += 1
                total_num += 1
    if total_num > 0:
        logger.info("Right: %d, Total: %d, Accuracy: %f", right_num, total_num, right_num / total_num)
    else:
        logger.info("check predict_result data.")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--predict_path",
                        required=True,
                        help="predict_path path.")
    return parser.parse_args()


def main():
    args = parse_args()
    predict_path = args.predict_path
    with safe_open(predict_path, 'r', encoding='utf-8') as f:
        predict_result = json.load(f)
    predict_result = process_predict(predict_result)
    compare(predict_result)


if __name__ == "__main__":
    main()
