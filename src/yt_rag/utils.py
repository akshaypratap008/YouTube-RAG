import pickle
import os
import sys
from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from pathlib import Path
import pandas as pd
from datetime import datetime

def save_object(obj, file_path:str):
    # saves the object as pickle file at file path
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"[INFO] Pickle file saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path:str):
    # load the pickle file
    try:
        with open(file_path, "rb") as f:
            loaded_file = pickle.load(f)
        logging.info(f"[INFO] Pickle file {file_path} loaded")
        return loaded_file
    except Exception as e:
        raise CustomException(e, sys)

def load_prompt(path: str)-> str:
    prompt_dir = "src/yt_rag/prompts"
    prompt_path = Path(prompt_dir) / f"{path}"
    return prompt_path.read_text(encoding="utf-8")
    

def save_eval_results(obj:pd.DataFrame, dir_path:str = "eval_results"):
    """Save Evaluation results as excel"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = Path(dir_path)
    dir.mkdir(exist_ok=True)
    file_path = dir / f"eval_{timestamp}.csv"
    obj.to_csv(file_path)

def convert_to_seconds(timestamp:str) -> int:
    """Converts timestamp (eg. '00:02:30') from str into seconds in int"""
    timestamps = timestamp.lstrip("(").rstrip(")").split(":")
    seconds = 0
    if len(timestamps) == 2:
        for i, timestamp in enumerate(timestamps):
            if i == 0:
                seconds += int(timestamp) * 60
            elif i == 1:
                seconds += int(timestamp)
    elif len(timestamps) == 3:
        for i, timestamp in enumerate(timestamps):
            if i == 0:
                seconds += int(timestamp) * 60 * 60
            elif i == 1:
                seconds += int(timestamp) * 60
            elif i == 2:
                seconds += int(timestamp)

    return seconds


