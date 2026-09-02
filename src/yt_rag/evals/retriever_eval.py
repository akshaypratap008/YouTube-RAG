from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from pathlib import Path
import json
from src.yt_rag.components.search import RAGSearch
import pandas as pd
import numpy as np
from datetime import datetime
import sys

from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException

EVAL_SET_DIR = "rag_evaluation/evaluation_set"
THRESHOLD = 0.7
LLM_JUDGE_MODEL = "gpt-4.1-mini"

class RetrivalEval:
    """
    Evaluates the retrieval results using deepeval and saves them into a csv file in rag
    """

    def __init__(self, eval_set_dir:str = EVAL_SET_DIR, eval_set_name:str = None):
        logging.info(f'[INFO] {eval_set_name} Evaluation initiated for evaluation_set: {eval_set_name}')
        self.eval_set_dir = Path(eval_set_dir)
        self.eval_set_name = eval_set_name
        self._run_evals()


    def _run_evals(self):
        test_cases = []
        eval_set_file_path = self.eval_set_dir / f"{self.eval_set_name}"
        try:
            with open(eval_set_file_path, 'r') as f:
                data = json.load(f)
            logging.info(f'[INFO] {self.eval_set_name} : Evaluation set loaded from {eval_set_file_path}')
        except Exception as e:
            raise CustomException(e, sys)
        self.video_url = data.get("video_url")

        # add question, actial answer and retrieved context to the test cases list (all items are LLMTestCase object)
        qa_pairs = data.get("qa_pairs")
        for pair in qa_pairs:
            retrieval_context = RAGSearch(url = self.video_url).search(query = pair.get("question"))
            test_cases.append(
                LLMTestCase(
                    input = pair.get("question"),
                    expected_output = pair.get("answer"),
                    retrieval_context = retrieval_context,
                    actual_output = "(generator not evaluated in this run)"
                )
            )
        logging.info(f"[INFO] {self.eval_set_name} : Test cases created")

        # metrics => Recall(any chunk missed?), Precission(how much noise is retrieved?)
        metrics = [
            ContextualRecallMetric(threshold=THRESHOLD, model = LLM_JUDGE_MODEL, include_reason=True),
            ContextualPrecisionMetric(threshold=THRESHOLD, model = LLM_JUDGE_MODEL, include_reason=True)
        ]

        self.hyperparameters = {
            "retriever": "base_k5",
            "embedding_model": "text-embedding-3-large",
            "similarity_threshold": 0.60,
            "max_tokens": 300,
            "overlap": 2,
            "top_k": 5,
            "judge_model": LLM_JUDGE_MODEL,
            "eval_set": self.eval_set_name
        }

        try:
            # evaluate every metric for every case
            result = evaluate(
                test_cases=test_cases,
                metrics=metrics,
                hyperparameters = self.hyperparameters,
                cache_config = CacheConfig(
                    use_cache=False,
                    write_cache=False
                )
            )
            logging.info(f'[INFO] {self.eval_set_name} : Evaluation completed for evaluation set: {self.eval_set_name}')
            self.save_results(result = result)
            logging.info(f'[INFO] {self.eval_set_name} : Evaluation results saved')
            return result
        except Exception as e:
            raise CustomException(e, sys)

    def save_results(self, result):
        recall_scores = []
        precision_scores = []

        success_count = 0
        failure_count = 0

        for test_result in result.test_results:
            for metric in test_result.metrics_data:
                if metric.name == "Contextual Recall":
                    recall_scores.append(metric.score)
                elif metric.name == "Contextual Precision":
                    precision_scores.append(metric.score)
                if metric.success:
                    success_count += 1
                else :
                    failure_count += 1

        final_recall_score = np.mean(recall_scores)
        final_precision_score = np.mean(precision_scores)

    
        result_df = pd.DataFrame([{
            "date_time": datetime.now(),
            "eval_set": self.eval_set_name,
            "embedding_model": self.hyperparameters.get("embedding_model"),
            "chunk_similarity_threshold": self.hyperparameters.get("similarity_threshold"),
            "max_token": self.hyperparameters.get("max_tokens"),
            "top_k": self.hyperparameters.get("top_k"),
            "recall_score": final_recall_score,
            "precision_score": final_precision_score,
            "success" : success_count,
            "failure": failure_count
        }])

        try:
            # load csv file as df if exist
            file_path = Path("rag_evaluation/eval_results/deep_eval_results.csv")
            if file_path.exists():
                df = pd.read_csv(file_path)
                # concat result df and original df
                df = pd.concat([df, result_df], ignore_index=True)
                df.to_csv(file_path, index = False)

            else:
                # save file to eval_results folder
                result_df.to_csv(file_path, index = False)
        except Exception as e:
            raise CustomException(e, sys)

        

            





            
