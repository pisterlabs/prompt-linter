import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
from functools import reduce
from pathlib import Path

import openai
import pandas as pd
import wandb
import wandb.apis.reports as wr
import yaml


EVALS_REPO_PATH = "/setup/evals"


def expand(df, col):
    return pd.concat(
        [
            df.reset_index(drop=True),
            pd.json_normalize(df[col]).reset_index(drop=True),
        ],
        axis=1,
    ).drop(col, axis=1)


def extend_chatml(r):
    def get_correct(r):
        if hasattr(r, "correct"):
            return r.correct
        if hasattr(r, "choice"):
            return True if r.choice == "Y" else False
        return None

    extensions = [
        {"role": "assistant", "content": r.sampled, "correct": get_correct(r)}
    ]
    try:
        return r.prompt + extensions
    except Exception:
        return []


def make_chatml_viz(convo):
    styles = """
    <style>
        /* Message Bubble Container */
        .message-bubble {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: flex-start;
        margin-bottom: 10px;
        max-width: 600px;
        }

        /* Role Section */
        .message-role {
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 5px;
        color: #555;
        }

        /* Message Section */
        .message-content {
        background-color: #e6e6e6;
        padding: 10px;
        border-radius: 10px;
        font-size: 16px;
        max-width: 100%;
        word-wrap: break-word;
        box-shadow: 0px 1px 1px rgba(0, 0, 0, 0.2);
        }

        /* System messages */
        .message-bubble.system .message-content {
        background-color: #f2f2f2;
        color: #999;
        }

        /* Assistant messages */
        .message-bubble.assistant .message-role {
        color: #6666ff;
        }

        .message-bubble.assistant .message-content {
        background-color: #e6e6ff;
        color: #000;
        }

        /* Sampled messages */
        .message-bubble.sampled {
        justify-content: flex-end;
        align-items: flex-end;
        margin-left: 20%;
        }

        .message-bubble.sampled .message-role {
        color: #1e6ba1;
        text-align: right;
        }

        .message-bubble.sampled .message-content {
        background-image: linear-gradient(to left, #1e6ba1, #7fb1dd);
        color: white;
        /* fallback for browsers that don't support gradients */
        background-color: #1e6ba1;
        /* adjust these values to control the gradient effect */
        background-size: 150% 100%;
        background-position: right;
        transition: background-position 0.3s ease-out;
        border-radius: 10px;
        padding: 10px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0px 1px 1px rgba(0, 0, 0, 0.2);
        }

        /* Picked messages */
        .message-bubble.picked .message-role {
        color: #555;
        }

        .message-bubble.assistant.true .message-content {
        background-color: #c3e6cb;
        color: #000;
        }

        .message-bubble.assistant.false .message-content {
        background-color: #f5c6cb;
        color: #000;
        }

        .message-bubble.assistant.true .message-role {
        color: #006400;
        }

        .message-bubble.assistant.false .message-role {
        color: #8b0000;
        }

        /* Right-aligned message bubble for the user */
        .message-bubble.user {
        justify-content: flex-end;
        align-items: flex-end;
        margin-left: 20%;
        }

        /* Role section for the user */
        .message-bubble.user .message-role {
        color: #555;
        text-align: right;
        }

        /* Message section for the user */
        .message-bubble.user .message-content {
        background-image: linear-gradient(to right, #80b6f4, #5c7cfa);
        color: white;
        /* fallback for browsers that don't support gradients */
        background-color: #5c7cfa;
        /* adjust these values to control the gradient effect */
        background-size: 150% auto;
        background-position: left center;
        text-shadow: 0px 0px 2px rgba(0, 0, 0, 0.2);
        box-shadow: none;
        }
    </style>
    """

    msgs = "\n".join(stylize_msg(msg) for msg in convo)
    html = f"{styles}{msgs}"

    return wandb.Html(html)


def stylize_msg(msg):
    correct = msg.get("correct")
    if correct is True:
        correct = "true"
    if correct is False:
        correct = "false"
    return f"""
    <div class="message-bubble {msg['role']} {correct}">
        <div class="message-role">{msg['role']}</div>
        <div class="message-content">{msg['content']}</div>
    </div>
    """


def chatml_to_markdown(convo):
    md = ""
    for msg in convo:
        role = msg["role"]
        content = msg["content"]
        md += f"# {role}\n{content}\n\n"

    return md


def get_expand_subsets(df, col):
    return {
        subset: df.loc[df[col] == subset].pipe(expand, "data").reset_index(drop=True)
        for subset in df[col].unique()
    }


def override_base_prompt(convo, new_prompt):
    for i, msg in enumerate(convo):
        if msg.get("role") != "system":
            break

    new_convo = [{"role": "system", "content": new_prompt}] + convo[i:]
    return new_convo


def run_eval(run):
    model_name, override_prompt = get_model_name_prompt(run.config["model"])
    _eval = get_correct_eval_name(run)

    if registry := run.config.get("registry"):
        art = run.use_artifact(registry)
        art_path = art.download()
        shutil.copytree(art_path, f"{EVALS_REPO_PATH}/evals", dirs_exist_ok=True)

    if override_prompt and not is_meta_eval:
        registry_path = Path(f"{EVALS_REPO_PATH}/evals/registry")

        jsonl_args = {}
        for f in (registry_path / "evals").glob("**/*.yaml"):
            d = yaml.safe_load(f.open())

            if _eval not in d:
                continue

            eval_id = d[_eval]["id"]
            jsonl_args = {
                k: v for k, v in d[eval_id]["args"].items() if str(v).endswith(".jsonl")
            }
            break

        if not jsonl_args:
            raise ValueError(f"Could not find {_eval} in {registry_path / 'evals'}")

        for k, v in jsonl_args.items():
            fpath = registry_path / "data" / v
            df = pd.read_json(fpath, lines=True)
            if is_meta_eval:
                # if it does not end with -meta, it should overwrite the modelgraded prompt?
                pass
            else:
                df.input = df.input.apply(
                    override_base_prompt, new_prompt=override_prompt
                )
            df.to_json(fpath, lines=True, orient="records")

    oaieval_settings = run.config["oaieval_settings"]
    if "record_path" in oaieval_settings:
        del oaieval_settings["record_path"]
        wandb.termwarn("Using `record_path` in `oaieval_settings` is not supported.")

    args = [model_name, _eval]
    valid_keys = [
        "extra_eval_params",
        "max_samples",
        "cache",
        "visible",
        "seed",
        "user",
        "log_to_file",
        "registry_path",
        "debug",
        "local-run",
        "dry-run",
        "dry-run-logging",
    ]
    for k, v in oaieval_settings.items():
        if k in valid_keys:
            v = shlex.quote(str(v))
            args.append(f"--{k}={v}")

    cmd = ["oaieval"] + args + ["--record_path", "temp.jsonl"]
    subprocess.run(cmd, check=True, timeout=300)


def get_overlapping_cols(dfs):
    cols = {}
    for i, df in enumerate(dfs):
        for col in df:
            if col not in cols:
                cols[col] = [i]
            else:
                cols[col].append(i)

    overlaps = set(col for col, indices in cols.items() if len(set(indices)) > 1)
    return overlaps


def drop_if_exists(df, col):
    if col in df:
        df = df.drop(col, axis=1)
    return df


def merge(dfs, key, drop_duplicates=True, suffixes=("", "__extraneous")):
    df = reduce(lambda x, y: x.merge(y, on=list(key), suffixes=suffixes), dfs)
    if drop_duplicates:
        df = df.drop(df.filter(regex=suffixes[-1]), axis=1)
    return df


def reshape_by_eval_level(df, key):
    eval_levels = ["", "modelgraded_", "meta_modelgraded_"]
    eval_dfs = []
    for i, lvl in enumerate(eval_levels):
        df2 = (
            df.groupby(list(key))
            .nth(i)
            .pipe(lambda df: df.assign(extended_convo=df.apply(extend_chatml, axis=1)))
            .pipe(
                lambda df: df.assign(
                    chatml_viz=df.extended_convo.apply(make_chatml_viz),
                    markdown=df.extended_convo.apply(chatml_to_markdown),
                    prompt=df.prompt.apply(json.dumps),
                    extended_convo=df.extended_convo.apply(json.dumps),
                )
            )
            .add_prefix(lvl)
        )
        if len(df2) > 0:
            eval_dfs.append(df2)

    eval_df = reduce(lambda x, y: x.join(y), eval_dfs).reset_index()
    return eval_df


def get_evals_table(test_results):
    key = {"sample_id"}

    df = get_base_df(test_results)
    dfs = get_expand_subsets(df, "type")
    dfs["sampling"] = dfs["sampling"].pipe(reshape_by_eval_level, key=key)

    final_df = merge(dfs.values(), key).assign(
        completion_fns=str(spec.get("completion_fns")),
        eval_name=spec.get("eval_name"),
    )

    model_name, override_prompt = get_model_name_prompt(run.config["model"])
    final_df["completion_cost"] = final_df.usage_total_tokens.apply(
        add_completion_cost, model_name=model_name
    )

    if override_prompt and not is_meta_eval:
        final_df["override_prompt"] = override_prompt

    if registry := run.config.get("registry"):
        art = run.use_artifact(registry)
        final_df["registry_version"] = art.version

    scary_cols = final_df.columns[
        (final_df.dtypes == "object") & ~final_df.columns.str.contains("chatml_viz")
    ]
    for col in scary_cols:
        final_df[col] = final_df[col].astype(str)

    return final_df


def get_model_name_prompt(model):
    if isinstance(model, str):
        art = run.use_artifact(model)
        art_path = art.download()
        with open(f"{art_path}/model_spec.json") as f:
            model = json.load(f)

    elif isinstance(model, dict):
        art = wandb.Artifact("openai_evals_model", type="model")
        model_spec_fname = "model_spec.json"
        with open(model_spec_fname, "w") as f:
            json.dump(run.config["model"], f)
        art.add_file(model_spec_fname)
        run.use_artifact(art)

    run.summary["model"] = model.get("name")
    run.summary["override_prompt"] = model.get("override_prompt")

    return model.get("name"), model.get("override_prompt")


def generate_report(run):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name, override_prompt = get_model_name_prompt(run.config["model"])
    _eval = get_correct_eval_name(run)

    template_url = (
        "https://wandb.ai/wandb/jobs/reports/Alt-Report-Template--Vmlldzo0MTIwOTUx"
    )
    if _eval.startswith("manga"):
        template_url = "https://wandb.ai/wandb/jobs/reports/Baseline-Report-Template--Vmlldzo0MDk5NTUz"

    reference_report = wr.Report.from_url(template_url)
    blocks = []
    for b in reference_report.blocks:
        if isinstance(b, wr.PanelGrid):
            b.runsets = [wr.Runset(run.entity, run.project)]
        blocks.append(b)

    wr.Report(
        entity=run.entity,
        project=run.project,
        title=f"OpenAI Evals Profiling Report: {model_name}: {_eval}",
        description=f"{now}",
        width="fluid",
        blocks=blocks,
    ).save()


def get_test_results():
    return pd.read_json("temp.jsonl", lines=True)


def get_base_df(test_results):
    return test_results.loc[
        test_results.spec.isna() & test_results.final_report.isna()
    ].drop(["spec", "final_report"], axis=1)


def get_spec(test_results):
    return test_results.loc[test_results.spec.notna()].spec.iloc[0]


def get_final_report(test_results):
    final_report = test_results.loc[
        test_results.final_report.notna()
    ].final_report.iloc[0]
    if not isinstance(final_report, dict):
        final_report = {"final_report": final_report}
    return final_report


def add_completion_cost(n_tokens, model_name):
    cost_per_1k_tokens = {
        "gpt-4": 0.06,
        "gpt-3.5-turbo": 0.002,
        "ada": 0.0016,
        "babbage": 0.0024,
        "curie": 0.0120,
        "davinci": 0.12,
    }

    cost = 0
    for k, v in cost_per_1k_tokens.items():
        if k in model_name:
            cost = v / 1000 * n_tokens
            break
    return cost


def get_correct_eval_name(run):
    # a bit annoying to have to do this...

    _eval = run.config["eval"]

    pattern = r"\w+(?:\.|-)v\d$"

    valid_evals = []
    alternate_evals = {}
    for p in Path(f"{EVALS_REPO_PATH}/evals/registry/evals").glob("**/*.yaml"):
        with p.open() as f:
            d = yaml.safe_load(f)
        keys = [k for k in d.keys() if not re.search(pattern, k)]
        valid_evals.extend(keys)

        possibly_confused_name = p.stem
        alternate_evals[possibly_confused_name] = keys[0]

    if _eval not in valid_evals:
        if _eval not in alternate_evals:
            raise ValueError(
                f"Invalid eval: {_eval}.  Please choose from: {valid_evals}"
            )
        # User confused the name of the eval withthe name of the yaml
        _eval = alternate_evals[_eval]

    return _eval


openai.api_key = os.getenv("OPENAI_API_KEY")
settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    _eval = get_correct_eval_name(run)
    is_meta_eval = _eval.endswith("-meta")
    run_eval(run)
    test_results = get_test_results()

    spec = get_spec(test_results)
    run.summary["spec"] = spec
    final_report = get_final_report(test_results)
    evals = get_evals_table(test_results)

    total_completion_cost = evals.completion_cost.sum()
    tokens_generated = evals.usage_total_tokens.sum()

    run.log(
        {
            "tokens_generated": tokens_generated,
            "total_completion_cost": total_completion_cost,
            **final_report,
        }
    )

    if _eval.startswith("manga"):
        run.log(
            {
                "evals": wandb.plot_table(
                    vega_spec_name="megatruong/test",
                    data_table=wandb.Table(dataframe=evals),
                    fields={
                        "metric": "sacrebleu_sentence_score",
                        "color": "override_prompt",
                        "xaxis": "registry_version",
                        "hover": "markdown",
                    },
                ),
            }
        )
    else:
        run.log({"evals_table": evals})

    art = wandb.Artifact("results", type="results")
    art.add_file("temp.jsonl")
    run.log_artifact(art)

    generate_report(run)
