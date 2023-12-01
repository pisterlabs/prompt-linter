import logging
import evals
import evals.api
import evals.base
import evals.record
from evals.eval import Eval
from evals.registry import Registry
from types import SimpleNamespace
import openai
evals.eval.set_max_samples(8)

# define the arguments
args = {
    "completion_fn": "gpt-3.5-turbo",
    "eval": "match_mmlu_machine_learning",
    "cache": True,
    "seed": 20220722
    }
args = SimpleNamespace(**args)

# evaluation specification
registry = Registry()
eval_spec = registry.get_eval(args.eval)

# eval object
eval_class = registry.get_class(eval_spec)

openai.api_key = "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"
completion_fn_instance = registry.make_completion_fn(args.completion_fn) 
eval: Eval = eval_class(
    completion_fns=[completion_fn_instance],
    samples_jsonl=eval_spec.args["samples_jsonl"],
    name=eval_spec.key, # match_mmlu_machine_learning.test.v1,
    seed=args.seed
    )

# recorder
eval_name = eval_spec.key # match_mmlu_machine_learning.test.v1
run_spec = evals.base.RunSpec(
    completion_fns=[args.completion_fn],
    eval_name=eval_name,
    base_eval=eval_name.split(".")[0],
    split=eval_name.split(".")[1],
    run_config = {
        "completion_fns": [args.completion_fn],
        "eval_spec": eval_spec,
        "seed": args.seed,
    },
    created_by="xinzhe", # my name
)
recorder_path = f"evallogs/{run_spec.run_id}_{args.completion_fn}_{args.eval}.jsonl"
recorder = evals.record.LocalRecorder(recorder_path, run_spec)

# run the evaluation
result = eval.run(recorder)
recorder.record_final_report(result)



