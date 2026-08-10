from src.yt_rag.evals.eval_pipeline import EvalPipeline

video_url = "https://www.youtube.com/watch?v=sD468LfeVdc&t=4858s"

eval_set_dir = "rag_evaluation/evaluation_set"
eval_pipeline = EvalPipeline(eval_set_dir, video_url=video_url)

eval_pipeline.run_evaluations()
