from pathlib import Path
import os
import sys
import json

from src.yt_rag.components.search import RAGSearch
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import CacheConfig
from src.yt_rag.logger import logging
from deepeval.evaluate.types import EvaluationResult
import pandas as pd
from src.yt_rag.exceptions import CustomException

GOLDENS_PATH = "rag_evaluation/evaluation_set/deep_eval_goldens"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7

# os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"


class RagEval:
    """This class contains components which can be used to evaluate the generator using deepeval library."""

    def __init__(self, goldens_path:str = GOLDENS_PATH, judge_model_name:str = JUDGE_MODEL, threshold:float = THRESHOLD):
        self.GOLDENS_PATH = goldens_path
        self.judge_model_name = judge_model_name
        self.threshold = threshold

        self.eval_set_names:list = os.listdir(Path(self.GOLDENS_PATH))
        # print(self.eval_set_names)

    def run(self, eval_set_names = None):
        if eval_set_names is None:
            eval_set_names = self.eval_set_names
        elif isinstance(eval_set_names, str):
            eval_set_names = [eval_set_names]
        for name in eval_set_names:
            test_cases = []

            # load json for every set
            eval_set_file_path = Path(self.GOLDENS_PATH, name)
            with open(eval_set_file_path, "r") as f:
                eval_set = json.load(f)
                logging.info(f'[INFO] - {name} - golden dataset json file loaded')
                # print(eval_set)

            video_url = eval_set['video_url']
            goldens = eval_set['goldens']

            # create test cases
            generator = RAGSearch(url = video_url, llm_model="gpt-4o")
            for g in goldens:
                input = g['input']

                # context is fetched from retriever
                context = generator.search(query = input)

                # answer is generated using the generator
                answer = generator.generate_response(context = context, query=input)

                print(
                    f"\nQuestion: {input}"
                    f"\nChunks: {len(context)}"
                    f"\nWords per chunk: {[len(chunk.split()) for chunk in context]}"
                    f"\nTotal context words: {sum(len(chunk.split()) for chunk in context)}"
                    f"\nAnswer words: {len(answer.split())}\n"
                )

                test_cases.append(
                    LLMTestCase(
                        input= input,
                        actual_output=answer,
                        retrieval_context=context,
                        # expected answer is not needed
                    )
                )

            # define metrics
            metrics = [
                ContextualRelevancyMetric(
                    threshold= self.threshold,
                    model = self.judge_model_name,
                    include_reason= False
                )
                # FaithfulnessMetric(
                #     threshold= self.threshold,
                #     model = self.judge_model_name,
                #     include_reason= False
                # ),
                # AnswerRelevancyMetric(
                #     threshold= self.threshold,
                #     model = self.judge_model_name,
                #     include_reason= False
                # )
            ]

            # evaluate
            results = evaluate(
                test_cases=test_cases,
                metrics = metrics,
                cache_config=CacheConfig(
                    use_cache=False,
                    write_cache=False
                )
            )
            logging.info(f'[INFO] - {name} - Evaluation complete for {name}')
            # self.save_results(results = results, eval_set_name=name)
            logging.info(f'[INFO] - {name} - Evaluation saved in csv')
            

    def save_results(self, results:EvaluationResult, eval_set_name:str):
        rows = []
        for test_result in results.test_results:
            row = {
                "eval_set_name": eval_set_name,
                "test_name": test_result.name,
                "model_name": None,
                "success": test_result.success,
                "faithfulness_score": None,
                "faithfulness_success": None,
                "answer_relevance_score": None,
                "answer_relevance_success": None,
                "contextual_relevancy_score": None,
                "contextual_relevancy_success": None
            }

            for metric_data in test_result.metrics_data:
                row["model_name"] = metric_data.evaluation_model

                if metric_data.name == "Faithfulness":
                    row["faithfulness_score"] = metric_data.score
                    row["faithfulness_success"] = metric_data.success
                elif metric_data.name == "Answer Relevancy":
                    row["answer_relevance_score"] = metric_data.score
                    row["answer_relevance_success"] = metric_data.success
                elif metric_data.name == "Contextual Relevancy":
                    row["contextual_relevancy_score"] = metric_data.score
                    row["contextual_relevancy_success"] = metric_data.success


            rows.append(row)

        results_df = pd.DataFrame(rows)

        try:
            file_path = Path("rag_evaluation/eval_results/full_rag_eval_deepeval.csv")
            if file_path.exists():
                df = pd.read_csv(file_path)
                # concat result df and original df
                df = pd.concat([df, results_df], ignore_index=True)
                df.to_csv(file_path, index = False)
            else:
                results_df.to_csv(file_path, index = False)
        except Exception as e:
            raise CustomException(e, sys)
        

        