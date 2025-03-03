# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os 


def select_data(eval_qa_root):
    dataset_qajson = {
    "Ucfcrime": f"{eval_qa_root}/Eval_QA/Ucfcrime_QA_new.json",
    "Youcook2": f"{eval_qa_root}/Eval_QA/Youcook2_QA_new.json",
    "TVQA": f"{eval_qa_root}/Eval_QA/TVQA_QA_new.json",
    "MSVD": f"{eval_qa_root}/Eval_QA/MSVD_QA_new.json",
    "MSRVTT": f"{eval_qa_root}/Eval_QA/MSRVTT_QA_new.json",
    "Driving-decision-making": f"{eval_qa_root}/Eval_QA/Driving-decision-making_QA_new.json",
    "NBA": f"{eval_qa_root}/Eval_QA/NBA_QA_new.json",
    "SQA3D": f"{eval_qa_root}/Eval_QA/SQA3D_QA_new.json",
    "Driving-exam": f"{eval_qa_root}/Eval_QA/Driving-exam_QA_new.json",
    "MV": f"{eval_qa_root}/Eval_QA/MV_QA_new.json",
    "MOT": f"{eval_qa_root}/Eval_QA/MOT_QA_new.json",
    "ActivityNet": f"{eval_qa_root}/Eval_QA/ActivityNet_QA_new.json",
    "TGIF": f"{eval_qa_root}/Eval_QA/TGIF_QA_new.json",
    "sampling_data":f"{eval_qa_root}/Eval_QA/sampling_datasets_QA_new.json"
    }
    return dataset_qajson


def process_qs(item, eval_video_root, question):
    question += item['question']
    qs_choice = item['choices']
    if len(qs_choice) == 6:
        question += f"Choices: A.{qs_choice['A']} B.{qs_choice['B']} C.{qs_choice['C']} \
D.{qs_choice['D']} E.{qs_choice['E']} F.{qs_choice['F']} \
\n Among the six options A, B, C, D, E, F above, the one closest to the correct answer is:"
    elif len(qs_choice) == 5:
        question += f" A.{qs_choice['A']} B.{qs_choice['B']} C.{qs_choice['C']} \
D.{qs_choice['D']} E.{qs_choice['E']} \
\n Among the five options A, B, C, D, E above, the one closest to the correct answer is: "
    elif len(qs_choice) == 4:
        question += f" A.{qs_choice['A']} B.{qs_choice['B']} C.{qs_choice['C']} \
D.{qs_choice['D']} \n Among the four options A, B, C, D above, the one closest to the correct answer is:"
    elif len(qs_choice) == 3:
        question += f" A.{qs_choice['A']} B.{qs_choice['B']} C.{qs_choice['C']} \
\n Among the three options A, B, C above, the one closest to the correct answer is: "
    elif len(qs_choice) == 2:
        question += f" A.{qs_choice['A']} B.{qs_choice['B']} \
\n Among the two options A, B above, the one closest to the correct answer is: "
    vid_rela_path = item['vid_path']
    vid_path = os.path.join(eval_video_root, vid_rela_path)
    return question, vid_path
