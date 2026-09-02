from src.yt_rag.evals.eval_pipeline import EvalPipeline
from yt_rag.evals.retriever_eval import RetrivalEval
from src.yt_rag.evals.generator_eval import GeneratorEval
import os

if __name__ == "__main__":
    # eval_set_dir = "rag_evaluation/evaluation_set"
    # eval_sets = os.listdir(eval_set_dir)

    # for eval_set in eval_sets:
    #     result = RetrivalEval(eval_set_name= eval_set)

    

    # video_url = "https://www.youtube.com/watch?v=sD468LfeVdc&t=4858s"

    # eval_set_dir = "rag_evaluation/evaluation_set"
    # eval_pipeline = EvalPipeline(eval_set_dir, video_url=video_url)

    # eval_pipeline.run_evaluations()

    # ------ Generator eval ----
    gen_eval_pipeline = GeneratorEval()
    gen_eval_pipeline.run(eval_set_names="7ARBJQn6QkM.json")





