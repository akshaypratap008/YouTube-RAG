from dotenv import load_dotenv
from pathlib import Path
import os
import sys
import json
from typing import List, Any, Dict
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith import traceable


from src.yt_rag.components.search import RAGSearch
from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from src.yt_rag.utils import load_prompt, convert_to_seconds
import time 

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "YouTube-RAG"

LLM_JUDGE_MODEL = "gpt-4o-mini"

class EvalPipeline:
    """
    Runs evaluation pipeline
    """

    def __init__(self, eval_set_dir:str = "rag_evaluation/evaluation_set"):
        """
        Initiate Evaluation pipeline
        """
        self.eval_set_dir = eval_set_dir
        self.eval_sets_names: list = os.listdir(Path(self.eval_set_dir))
        logging.info(f"[INFO] Evaluation started")

    def run_evaluations(self):
        """
        Visit each eval set -> load examples to dataset -> run evaluations.
        """
        try:
            client = Client()
            dataset = client.create_dataset(f"RAG-Full-Evaluation")
            logging.info(f"[INFO] - ")
            eval_dir = Path(self.eval_set_dir)
            for i, eval_set in enumerate(self.eval_sets_names):
                file_path = eval_dir / eval_set
                with open(file_path, "r") as f:
                    data = json.load(f)
                examples = data['qa_pairs'][0]
                client.create_example(
                    inputs = {
                        "question": examples['question'], 
                        "difficulty": examples['difficulty']
                    },
                    outputs= {
                        "answer": examples['answer'],
                        "timestamp": examples["timestamp"]
                    },
                    metadata= {"timestamp": convert_to_seconds(examples['timestamp'])},
                    dataset_id = dataset.id
                )
                logging.info(f"[INFO] Examples added to dataset for {eval_set}")

                def target(inputs:dict) -> dict:
                    return self.run_rag_pipeline(url = data['video_url'], query = inputs['question'])

                experiment_results = client.evaluate(
                    target, 
                    data = "RAG-Full-Evaluation",
                    evaluators= [self.correctness, self.groundedness, self.relevance, self.retrieval_relevance, self.timestamp_error],
                    metadata= {"version": "LCEL context, gpt-4-0125-preview"}
                )
                logging.info(f"[INFO] Evaluation completed for evaluation set - {i}")
        except Exception as e:
            raise CustomException(e, sys)

    @traceable()
    def run_rag_pipeline(self, url, query):
        rag = RAGSearch(url = url)
        start = time.time()
        relevant_chunks = rag.search(query = query)
        context = " ".join(relevant_chunks)
        response = rag.generate_response(context = context, query = query)
        timestamps = rag.get_video_timestamps()
        end = time.time()

        # check if timestamps have been retrieved
        if timestamps and isinstance(timestamps[0], (list, tuple)) and len(timestamps[0]) >= 1:
            timestamp = timestamps[0][0]            # assignes the start time for the most relevant chunk to timestamp variable
        else:
            timestamp = None

        return {
            "answer": response,
            "context": context,
            "timestamp": timestamp,
            "runtime_duration": end-start,
            "video_id": rag.video_id  
        }    


    def correctness(self, inputs: dict, outputs: dict, reference_outputs:dict) -> dict:
        """
        Evaluates how correct the answer is as compared to the ground truth answer. Grades 1-10 based on the similarity between the two. Also returns the bool value. 
        """
        class CorrectnessGrade(TypedDict):
            # define the correctness grade output
            grade : Annotated[int, "Grade from 1 to 10"]
            correct: Annotated[bool, "True if answer is correct otherwise False"]
            reasoning: Annotated[str, "Explain the reasoning for the score"]

        correctness_instruction = load_prompt(path="eval_prompts/correctness_prompt.txt")         # loads prompt from text files inside src.yt_rag.prompts.eval_prompts

        grader_llm = ChatOpenAI(model = LLM_JUDGE_MODEL).with_structured_output(CorrectnessGrade, method = "json_schema", strict=True)

        answers = f"""
QUESTION: {inputs['question']}
GROUND TRUTH_ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {outputs["answer"]}
"""
        results = grader_llm.invoke([
            {"role": "system", "content": correctness_instruction},
            {"role": "user", "content": answers}
        ])

        return {
            "key": "correctness",
            "score": results['grade'],
            "value": results['correct'],
            "comment": results['reasoning'],
            "metadata": {"dificulty": inputs.get('difficulty')}
        }
    
    def relevance(self, inputs: dict, outputs: dict) -> dict:
        """
        Evaluates how relevant the answer is to the question. Grades 1-10 on how much the answer is relevant to the quest. Also returns a bool value
        """
        class RelevanceGrade(TypedDict):
            grade: Annotated[int, "Grade from 1 to 10"]
            relevant: Annotated[bool, "True if answer addresses the question. False if answer doesn't address the question"]
            reasoning: Annotated[str, "Explain the reason for the score"]

        relevance_instructions = load_prompt("eval_prompts/relevance_prompt.txt")            # loads prompt from src.yt_rag.prompts.eval_prompts

        grader_llm = ChatOpenAI(model = LLM_JUDGE_MODEL).with_structured_output(RelevanceGrade, method = "json_schema", strict=True)

        answers = f"""
QUESTION: {inputs['question']}
STUDENT ANSWER: {outputs['answer']}
"""
        results =  grader_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answers}
        ])

        return {
            "key": "relevance",
            "score": results['grade'],
            "value": results["relevant"],
            "comment": results['reasoning'],
            "metadata": {"dificulty": inputs.get('difficulty')}
        }

    def groundedness(self, inputs:dict, outputs:dict) -> dict:
        """
        Evaluates the groundedness of the Answers. Grades between 1-10 on how much the answer relates to the retrieved chunks. 
        """
        class GroundednessGrade(TypedDict):
            grade: Annotated[int, "Grade from 1 to 10"]
            grounded: Annotated[bool, "True if the answer is grounded. False if the answer hallucinates and invents facts."]
            reasoning: Annotated[str, "Explain the reason for the score"]

        groundedness_instructions = load_prompt("eval_prompts/groundedness_prompt.txt")

        grader_llm = ChatOpenAI(model = LLM_JUDGE_MODEL).with_structured_output(GroundednessGrade, method = "json_schema", strict = True)

        answers = f"""
FACTS: {outputs['context']}
STUDENT ANSWER: {outputs['answer']}
"""
        results = grader_llm.invoke([
            {"role": "system", "content": groundedness_instructions},
            {"role": "user", "content": answers}
        ])

        return {
            "key": "groundedness",
            "score": results['grade'],
            "value": results['grounded'],
            "comment": results['reasoning'],
            "metadata": {"dificulty": inputs.get('difficulty')}
        }

    def retrieval_relevance(self, inputs: dict, outputs:dict) -> dict:
        """
        Evaluates how relevant the retrieved chunks are to the question
        """
        class RetrievedRelevanceGrade(TypedDict):
            grade: Annotated[int, "Grade between 1 to 10"]
            relevant: Annotated[bool, "True if the retrieved docs are relevant to the question, else false"]
            reasoning: Annotated[str, "Explain the reasoning behind the scores"]

        retrieved_relevance_instructions = load_prompt("eval_prompts/retrieval_relevance_prompt.txt")

        grader_llm = ChatOpenAI(model = LLM_JUDGE_MODEL).with_structured_output(RetrievedRelevanceGrade, method = "json_schema", strict = True)

        answers = f"""
FACTS : {outputs['context']}
QUESTION : {inputs['question']}
"""
        results = grader_llm.invoke([
            {"role": "system", "content": retrieved_relevance_instructions},
            {"role": "user", "content": answers}
        ])

        return {
            "key": "retrieval_relevance",
            "score": results['grade'],
            "value": results['relevant'],
            "comment": results['reasoning'],
            "metadata": {"dificulty": inputs.get('difficulty')}
        }

    def timestamp_error(self, inputs:dict, outputs:dict, reference_outputs:dict) -> dict:
        """
        Evaluates the error between the timestamps of the correct answer and the timestamp of the most relevant chunk retrieved by the rag
        """
        correct_ts = convert_to_seconds(reference_outputs["timestamp"])
        retrieved_ts = outputs['timestamp']

        if retrieved_ts is None:
            return {
                "key": "timestamp_error",
                "score": None,
                "value": None,
                "comment": "No timestamp were pulled for the extracted documents",
                "metadata": {"dificulty": inputs.get('difficulty')}
            }

        error = abs(correct_ts - retrieved_ts)   

        return {
            "key": "timestamp_error",
            "score": error,
            "value": error,
            "reasoning": None,
            "difficulty": inputs['difficulty']
        }   


        



    

    
        

    
