# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import argparse
import pandas as pd
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log.logging import logger


def accumulate_res(mode, res_lst):
    if not res_lst:
        raise ValueError("empty res_lst")
    last_slash_index = res_lst[0].rfind('/')
    res_folder_path = res_lst[0][:last_slash_index]
    origin_filename = res_lst[0][last_slash_index + 1:]
    df_list = []
    header_saved = False
    for file_path in res_list:
        csv_df = pd.read_csv(file_path, header=None)
        if not header_saved:
            header_saved = True
            df_list.append(csv_df)
        else:
            df_list.append(csv_df.iloc[1:])
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.loc[1:, [9, 10]] = None

    parts = origin_filename.split("_", 6)
    if mode == "round":
        res_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[6]}"
        res_filename = f"{res_filename[:res_filename.rfind('_')]}_round_result.csv"

    elif mode == "final":  
        non_first_token_tput_avg = final_df.iloc[1:, 7].apply(pd.to_numeric, errors='coerce').mean()
        e2e_tput_avg = final_df.iloc[1:, 8].apply(pd.to_numeric, errors='coerce').mean()
        final_df.loc[final_df.index[-1], 9] = non_first_token_tput_avg
        final_df.loc[final_df.index[-1], 10] = e2e_tput_avg
        res_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[6]}"
        res_filename = f"{res_filename[:res_filename.rfind('_')]}_final_result.csv"
    else:
        raise ValueError("incorrect accumulate res mode")
    res_path = os.path.join(res_folder_path, res_filename)
    with safe_open(res_path, mode='w', permission_mode=0o644) as f:
        final_df.to_csv(f, index=False, header=False)
    logger.info("maxbs %s result file saved in: %s", mode, res_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Model test utils arguments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['round', 'final'],
        help="Specify which mode to test"
    )
    parser.add_argument(
        "--res_list",
        type=str,
        help="Specify the res_list to accumulate"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res_list = args.res_list.split(",") if args.res_list else None
    accumulate_res(args.mode, res_list)
