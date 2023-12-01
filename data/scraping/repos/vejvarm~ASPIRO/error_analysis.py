import argparse
import json
import logging
import pathlib

from langchain.schema import BaseOutputParser, OutputParserException
from tqdm import tqdm

from flags import LOG_ROOT, PROMPT_TEMPLATES_FOLDER, TemplateErrors, sTOTAL_ERRORS, sTOTAL_PIDs_w_ERROR
from helpers import setup_logger
from parsing import JSONOutputParser, TextOutputParser

LOGFILE_PATH = LOG_ROOT.joinpath(pathlib.Path(__file__).name.removesuffix(".py")+".log")
LOGGER = setup_logger(__name__, loglevel=logging.WARNING, output_log_file=LOGFILE_PATH)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated_template_file",
        type=str,
        help="Path to the json file with generated templates."
    )
    parser.add_argument(
        "--output_subfolder",
        type=str,
        default="err",
        help="The name of subfolder to store errors."
    )
    parser.add_argument(
        "--template_file",
        type=str,
        help="Used to decide which parsing to use. Must end with '.tmp' and must contain 'json' in the name if structured output."
    )
    args = parser.parse_args()
    return args


def analyze_and_save_errors(path_to_generated_templates: pathlib, dump_folder: pathlib.Path, parser: BaseOutputParser = None):
    output_pid_template_dict = json.load(path_to_generated_templates.open())

    # recalculate the errors if parser is provided
    if parser is not None:
        for rel, output_dict in output_pid_template_dict.items():
            output = output_dict["output"]

            try:
                parser.parse(output)
                error_codes = []
                error_messages = []
            except OutputParserException as parser_err_msg:
                out_dict = json.loads(str(parser_err_msg))
                error_codes = out_dict["error_codes"]
                error_messages = out_dict["error_messages"]

            output_pid_template_dict[rel]["error_codes"] = error_codes
            output_pid_template_dict[rel]["error_messages"] = error_messages

        json.dump(output_pid_template_dict, path_to_generated_templates.open("w"), indent=2)

    total_pids = len(output_pid_template_dict)

    error_dicts = {err.name: {} for err in TemplateErrors}
    total_errors = 0
    pid_error_set = set()
    for i, (pid, output_dict) in tqdm(enumerate(output_pid_template_dict.items()), total=total_pids):
        output = output_dict["output"]
        error_codes = output_dict["error_codes"]
        # error_messages = output_dict["error_messages"]

        # ERROR HANDLING
        has_error = False
        for err in TemplateErrors:
            if err.value in error_codes:
                error_dicts[err.name][pid] = output
                has_error = True
                total_errors += 1

        if has_error:
            pid_error_set.add(pid)

    pid_error_list = list(pid_error_set)
    err_counts_dict = {sTOTAL_ERRORS: total_errors,
                       sTOTAL_PIDs_w_ERROR: len(pid_error_list)}

    for err in TemplateErrors:
        err_counts_dict[err.name] = len(list(error_dicts[err.name].keys()))
        json.dump(error_dicts[err.name],
                  dump_folder.joinpath(
                      f"err{err.name}.json").open("w"),
                  indent=2)

    json.dump(err_counts_dict, dump_folder.joinpath(f"errCOUNTS.json").open("w"), indent=2)
    json.dump(pid_error_list, dump_folder.joinpath(f"errLISTofPIDs.json").open("w"), indent=2)


def main(args):
    if args.template_file:
        assert args.template_file.endswith(".tmp")

        if "json" in args.template_file:
            path_to_template_file = PROMPT_TEMPLATES_FOLDER.joinpath(args.template_file)
            path_to_metadata_file = path_to_template_file.with_suffix(".json")

            assert path_to_template_file.exists(), f"The provided template file doesn't exist at {PROMPT_TEMPLATES_FOLDER}."
            assert path_to_metadata_file.exists(), f"The metadata file `{path_to_metadata_file.name}` doesn't exists at {path_to_metadata_file.parent}."

            metadata = json.load(path_to_metadata_file.open())

            parser = JSONOutputParser.from_metadata(metadata["first_key"], metadata["output_key"])
        else:
            parser = TextOutputParser()
    else:
        parser = None

    path_to_generated_templates = pathlib.Path(args.generated_template_file)
    folder_to_dump_error_jsons = path_to_generated_templates.parent.joinpath(args.output_subfolder)
    folder_to_dump_error_jsons.mkdir(parents=True, exist_ok=True)

    # LOAD existing template file
    analyze_and_save_errors(path_to_generated_templates, folder_to_dump_error_jsons, parser)


if __name__ == "__main__":
    main(parse_args())

