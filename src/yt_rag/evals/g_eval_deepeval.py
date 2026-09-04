import json
from deepeval.test_case import LLMTestCase, SingleTurnParams
from src.yt_rag.components.search import RAGSearch
from deepeval.metrics import GEval
from deepeval.evaluate import evaluate
from deepeval.evaluate.types import EvaluationResult
import pandas as pd
from pathlib import Path
import sys
from deepeval.metrics.g_eval import Rubric

from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException

class GEvalPipeline:

    def __init__(self, eval_set_path:str):
        self.eval_set_path = eval_set_path
        
    def run(self):
        logging.info(f'[INFO] - g-eval evaluation started')
        with open(self.eval_set_path, 'r') as f:
            eval_set = json.load(f)
            goldens = eval_set['goldens']

        test_cases = []
        rag = RAGSearch(url = eval_set['video_url'])
        for g in goldens:
            input = g['input']
            result = rag.generate_response(context = rag.search(input), query = input)

            test_cases.append(
                LLMTestCase(
                    input=input,
                    actual_output=result,
                    expected_output= g['expected_output']
                )
            )

        correctness = GEval(
            name = "Correctness",
            evaluation_steps= [
                "Compare the actual output against the key facts in the expected output.",
                "Heavily penalize statements in the actual output that contradicts the expected output or are factually wrong.",
                "Reward the statements that match the expected output in meaning, regardless of wording.",
                "DO NOT penalize the actual output for ommiting information - only wrong statements count here."
            ],
            rubric=[
                Rubric(score_range=(9, 10), expected_outcome="All factual claims are accurate and consistent with the expected output. No contradictions or factual errors."),
                Rubric(score_range=(5, 8), expected_outcome="Mostly accurate but includes one minor factual inaccuracy or slightly imprecise statement."),
                Rubric(score_range=(0, 4), expected_outcome="Contains clear factual errors or statements that contradict the expected output.")
            ],
            evaluation_params= [
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT
            ]
        )

        clarity = GEval(
            name = "Clarity",
            evaluation_steps = [
                "Evaluate whether the actual output is clear and consise",
                "Assess whether complex ideas are presented in a way that's easy to follow."
            ],
            rubric=[
                Rubric(score_range=(9, 10), expected_outcome="Extremely clear, well‑structured, and easy to understand. Ideas flow logically with precise language."),
                Rubric(score_range=(5, 8), expected_outcome="Generally clear but may contain minor ambiguity, slight disorganization, or occasional awkward phrasing."),
                Rubric(score_range=(0, 4), expected_outcome="Unclear, confusing, poorly structured, or difficult to follow. Meaning is partially or fully obscured.")
            ],
            evaluation_params=[
                SingleTurnParams.ACTUAL_OUTPUT
            ]
        )

        metrics = [
            correctness,
            clarity
        ]

        results = evaluate(test_cases= test_cases, metrics=metrics)
        logging.info(f'[INFO] - evaluation results generated')
        self.save_results(results = results)
        return results
        

    def save_results(self, results:EvaluationResult):
        rows = []
        for test_result in results.test_results:
            row = {
                "test_name": test_result.name,
                "model_name": None,
                "success": test_result.success,
                "correctness_score": None,
                "correctness_success": None,
                "clarity_score": None,
                "clarity_success": None,
            }

            for metric_data in test_result.metrics_data:
                row["model_name"] = metric_data.evaluation_model

                if metric_data.name == "Correctness [GEval]":
                    row["correctness_score"] = metric_data.score
                    row["correctness_success"] = metric_data.success
                elif metric_data.name == "Clarity [GEval]":
                    row["clarity_score"] = metric_data.score
                    row["clarity_success"] = metric_data.success

            rows.append(row)

        results_df = pd.DataFrame(rows)

        try:
            file_path = Path("rag_evaluation/eval_results/g_eval_results.csv")
            if file_path.exists():
                df = pd.read_csv(file_path)
                # concat result df and original df
                df = pd.concat([df, results_df], ignore_index=True)
                df.to_csv(file_path, index = False)
            else:
                results_df.to_csv(file_path, index = False)
            logging.info(f'[INFO] - evaluation results saved as csv in {file_path}')
        except Exception as e:
            raise CustomException(e, sys)
        

