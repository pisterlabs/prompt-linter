# set up openai api
import openai, os
import pandas as pd
from rag_with_cog_search import rag, parse_prompt_templates
from patch import log_json_artifact
import mlflow, json, tempfile

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

from mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default="./data/amldocs/questions/validation-10.csv")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--chain_type", type=str, default="stuff")
    parser.add_argument("--meta_prompt", type=str, default=None)
    parser.add_argument("--no-log", action='store_true')
    parser.add_argument("--scores", type=str, default="./data/amldocs/scores/validation_scores-10.json")
    args = parser.parse_args()

    verbose = not args.no_log

    if verbose:
        # mlflow.start_run()
        mlflow.log_param("questions", args.questions)
        mlflow.log_param("top", args.top)
        mlflow.log_param("chain_type", args.chain_type)
        mlflow.log_param("meta_prompt", args.meta_prompt)

    # load question
    df = pd.read_csv(args.questions)
    questions = df["question"].tolist()
    scores = []
    context_artifact_name = "cog_search_docs.json"

    if args.meta_prompt:
        # load meta_prompt from file
        with open(args.meta_prompt, "r") as f:
            meta_prompt = f.read()

        if meta_prompt.startswith("Content-Type: multipart/mixed;"):
            # if meta_prompt is a multipart mime file, split it into system and user templates
            system_template, user_template = parse_prompt_templates(meta_prompt)
        else:
            # otherwise, assume it's a single template
            system_template = meta_prompt
            user_template = None
    else:
        system_template = None
        user_template = None

    for i, question in enumerate(questions):
        # Start a child run
        with mlflow.start_run(run_name=f"RAG Run #{i}", nested=True) as child_run:
            # Log some metrics and parameters for the child run
            mlflow.log_param("question", question)
            mlflow.log_param("top", args.top)
            mlflow.log_param("chain_type", args.chain_type)
            mlflow.log_param("meta_prompt", args.meta_prompt)

            try:
                result = rag(question, top=args.top, chain_type=args.chain_type, 
                            context_artifact_name=context_artifact_name,
                            system_template=system_template, user_template=user_template, verbose=verbose)
                
                # load the cog_search context back from MLFlow 
                if verbose:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        artifact_local_path = mlflow.artifacts.download_artifacts(f"runs:/{child_run.info.run_id}/{context_artifact_name}", dst_path=temp_dir)
                        with open(artifact_local_path, 'r') as f:
                            result["context"] = json.load(f)

                    log_json_artifact(result, "result.json")
                
                
                
                print("Q:", result["query"])
                print("A:", result["result"])

                scores.append(result)
    
                # save scores to --scores output json file
                with open(args.scores, "w") as f:
                    json.dump(scores, f, indent=4)
            
            except Exception as e:
                print(e)
                print("skipping question: ", question)
        


