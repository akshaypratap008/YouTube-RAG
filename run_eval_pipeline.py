from src.yt_rag.evals.eval_pipeline import EvalPipeline

eval_set_dir = "rag_evaluation/evaluation_set"
eval_pipeline = EvalPipeline(eval_set_dir)

eval_pipeline.run_evaluations()
