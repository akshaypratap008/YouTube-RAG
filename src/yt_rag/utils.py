import pickle
import os
import sys
from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from pathlib import Path
import pandas as pd
import time

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

def load_prompt(prompt_name: str)-> str:
    prompt_path = Path("src")/f"yt_rag/prompts/{prompt_name}.txt"
    return prompt_path.read_text(encoding="utf-8")

def save_eval_results(obj:pd.DataFrame, file_path:str = "eval_results"):
    """Save Evaluation results as excel"""
    current_time = time.time()
    file_path = Path(file_path/{current_time})
    obj.to_excel(file_path=file_path)

