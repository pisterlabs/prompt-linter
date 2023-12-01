import os
import json
import time
import argparse
import logging
import sys
import openai
from openai import cli
import subprocess
import signal
import datetime
from tqdm import tqdm
import requests

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

AOAI_FINETUNE_CANCEL_STATUS = "canceled"
AOAI_FINETUNE_FAILED_STATUS = "failed"
AOAI_DEPLOYMENT_CANCEL_STATUS = "canceled"
AOAI_DEPLOYMENT_FAILED_STATUS = "failed"
AOAI_FINETUNE_TERMINAL_STATUS = ["succeeded", AOAI_FINETUNE_FAILED_STATUS, AOAI_FINETUNE_CANCEL_STATUS]
AOAI_DEPLOYMENT_TERMINAL_STATUS = ["succeeded", AOAI_DEPLOYMENT_FAILED_STATUS, AOAI_DEPLOYMENT_CANCEL_STATUS]
INFERENCE_FILE_PROMPT_KEY = 'prompt'
INFERENCE_FILE_METADATA_KEY = 'metadata'
GPT3_PROMPT_KEYWORD = "prompt"
GPT3_METADATA_KEYWORD = "metadata"
GPT3_PRED_KEYWORD = "samples"
OUTPUT_FILE_NAME = "processed_file_samples.0.jsonl"

MAX_TOKEN_LENGTH = 360

NL2SQL_SPIDER_MODELS = ["code-cushman-001", "code-cushman"]


def AOAI_wait_until_done(status_retrieve_func, status_retrieve_func_args, terminal_status, check_interval=10, time_out=None, patience=20):
    cumulative_waiting_time, status, error_cnt = 0, None, 0
    while True:
        try:
            status = status_retrieve_func(**status_retrieve_func_args)
        except Exception as e:
            error_cnt += 1
            logger.info(f"hitting exception: {e} while retrieving status, will silently ignore and restart retrieving, current error_cnt: {error_cnt}, total error_cnt before raising error: {patience}")
            if error_cnt > patience:
                raise Exception(f"Exceeding maximum patience ({patience}), please check previous error reports")
        if status in terminal_status:
            logger.info(f'job: {status_retrieve_func_args} hitting terminal status, current status: {status}, current cumulative_waiting_time: {cumulative_waiting_time}')
            break
        logger.info(f'job: {status_retrieve_func_args} is not hitting terminal status, current status: {status}, will sleep for {check_interval} seconds, current cumulative_waiting_time: {cumulative_waiting_time}')
        time.sleep(check_interval)
        cumulative_waiting_time += check_interval
        if (time_out is not None) and (cumulative_waiting_time >= time_out):
            raise TimeoutError(f'job: {status_retrieve_func_args} status: {status}, track hit time_out ({time_out}) threshold')

    logger.info(f"job: {status_retrieve_func_args} hitting terminal status: {status}")
    return status


class FileRetriever:
    @staticmethod
    def retrieve_file_with_name(file_name, purpose="fine-tune"):
        logger.info(f'Checking for existing uploaded file for file name: {file_name}')
        files = openai.File.list().data
        logger.info(f'Found {len(files)} total uploaded files in the subscription.')
        found, res = False, None
        for item in files:
            if item["filename"] == file_name:
                found = True
                res = item
                break
        if found:
            logger.info(f'Found in subscription using file name: {file_name} with file_id {res["id"]}')
        else:
            raise ValueError(f"uploading file is not implemented yet")
            # logger.info(f'uploading from local path: {file_name}')
            # res = openai.File.create(file=open(file_name), purpose=purpose, user_provided_filename=file_name)
            # logger.info(f'uploading complete with file_id: {res["id"]}')

            # AOAI is still working on file upload, in the meantime user can import files from blob storage using below command
            # curl -X 'POST' 'https://<api_base>/openai/files/import?api-version=<api_version>' \
            # -H "api-key:<api_key>" \
            # -H 'Content-Type: application/json' \
            # -d '{
            #     "purpose": "fine-tune",
            #     "filename": "<file_name>",
            #     "content_url": "<ulr_with_SAS_token>"
            #     }'

        return res


class AOAIFinetuningRunCreator:
    def __init__(self, job_id=None):
        self.job_id = None
        if job_id:
            self.job_id = job_id

    def create_run(self, finetune_hp: dict, use_python_sdk: bool = True):
        if self.job_id:
            logger.info(f"class initiated with job_id: {self.job_id}, will skip creation")
        else:
            if use_python_sdk:
                logger.info(f"creating finetuning using openai python SDK with hyper parameters: {finetune_hp}")
                resp = openai.FineTune.create(**finetune_hp)
            else:
                logger.info(f"creating finetuning using URL with hyper parameters: {finetune_hp}")
                payload = {
                    "model": finetune_hp.pop("model"),
                    "training_file": finetune_hp.pop("training_file"),
                    "validation_file": finetune_hp.pop("validation_file"),
                    "hyperparams": finetune_hp
                }
                url = f"{openai.api_base}/openai/fine-tunes?api-version={openai.api_version}"
                r = requests.post(url,
                                  headers={
                                      "api-key": openai.api_key,
                                      "Content-Type": "application/json"
                                  },
                                  json=payload
                                  )
                assert r.status_code not in [400, 500], f"hitting request error: {r} with json: {r.json()}"
                resp = r.json()

            logger.info(f"resp: {resp}")

            self.job_id = resp['id']
            status = resp["status"]
            logger.info(f'Fine-tunning model with jobID: {self.job_id}, current status: {status}')
        return self.job_id

    def signal_handler(self, sig, frame):
        status = openai.FineTune.retrieve(self.job_id).status
        logger.info(f"Stream interrupted. Job is still {status}.")
        return

    def streaming_events(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        events = openai.FineTune.stream_events(self.job_id)
        try:
            for event in events:
                logger.info(f'At time point: {datetime.datetime.fromtimestamp(event["created_at"])} {event["message"]}')
        except Exception as e:
            logger.info(f"Stream interrupted (client disconnected) with error {e}.")

    def check_run_status(self, time_out=None, check_interval=10):
        def status_retrieve_func(job_id):
            return openai.FineTune.retrieve(id=job_id)["status"]

        return AOAI_wait_until_done(status_retrieve_func=status_retrieve_func,
                                    status_retrieve_func_args={"job_id": self.job_id},
                                    terminal_status=AOAI_FINETUNE_TERMINAL_STATUS,
                                    check_interval=check_interval,
                                    time_out=time_out)


class AOAIDeployer:
    def __init__(self, finetune_job_id):
        self.finetune_job_id = finetune_job_id
        self.deployment = None

    def deploy_job(self):
        result = openai.FineTune.retrieve(id=self.finetune_job_id)
        assert result["status"] == 'succeeded', f"Trying to deploy a unsuccessful job: {self.finetune_job_id}, current job status: {result}"
        model = result["fine_tuned_model"]
        self.deployment = openai.Deployment.create(model=model, scale_settings={"scale_type": "standard"})
        logger.info(f"Creating a new deployment with model: {model}, with deployment id: {self.deployment['id']}")
        return self.deployment

    def check_deployment_status(self, check_interval=10, time_out=None):
        def status_retrieve_func(deployment_id):
            return openai.Deployment.retrieve(id=deployment_id)["status"]

        return AOAI_wait_until_done(status_retrieve_func=status_retrieve_func,
                                    status_retrieve_func_args={"deployment_id": self.deployment['id']},
                                    terminal_status=AOAI_DEPLOYMENT_TERMINAL_STATUS,
                                    check_interval=check_interval,
                                    time_out=time_out)


# TODO truncate prompts
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def check_length(txt):
    tokens = tokenizer(txt)["input_ids"]
    return len(tokens) < MAX_TOKEN_LENGTH


def truncate(txt, truncate_txt=True):
    if truncate_txt:
        max_len = len(txt.split(" "))
        while not check_length(txt):
            max_len -= 10
            txt = " ".join(txt.split(" ")[-max_len:])
            logger.info(f"txt: {txt} exceeding maximum length: {MAX_TOKEN_LENGTH}, shortening to {max_len}")
            if max_len <= 0:
                txt = ''
                logger.info(f"max_len <= 0, setting txt to ''")
                break
    return txt


class AOAIInferencer:
    def __init__(self):
        self.total_inference_calls = 0
        self.total_successful_inference_calls = 0
        self.total_erroneous_inference_calls = 0

    def inference_single_prompt(self, prompt, deployment_id, max_tokens, patience=2, retry_interval=5, error_action="ignore", truncate_txt=True):
        self.total_inference_calls += 1
        error_cnt, response = 0, None
        while True:
            try:
                response = openai.Completion.create(engine=deployment_id,
                                                    prompt=truncate(prompt, truncate_txt=truncate_txt),
                                                    max_tokens=max_tokens,
                                                    temperature=0,
                                                    top_p=1.0,
                                                    resample_unstable_tokens=True,
                                                    presence_penalty=0.0,
                                                    frequency_penalty=0.0,
                                                    )
                self.total_successful_inference_calls += 1
            except Exception as e:
                self.total_erroneous_inference_calls += 1
                error_cnt += 1
                logger.info(f"hitting exception: {e} while calling API, will sleep for {retry_interval} seconds and restart, current error_cnt: {error_cnt}, total error_cnt before raising error: {patience}")
                if error_cnt > patience:
                    if error_action == "ignore":
                        logger.info(f"Exceeding maximum patience ({patience}), error_action is {error_action}, silently ignoring this row")
                        return None
                    else:
                        raise Exception(f"Exceeding maximum patience ({patience}), please check previous error reports")
            if response:
                break
            time.sleep(retry_interval)
        return response

    def _get_paths(self, dataset_raw_path):
        if not dataset_raw_path:
            return []
        if not os.path.isdir(dataset_raw_path):
            paths = [dataset_raw_path]
        else:
            paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_raw_path) for f in filenames]
        return paths

    def retrieve_text(self, response):
        return response['choices'][0]['text']

    def inference_dir(self, args, input_dir, output_file_path, deployment_id, max_tokens, print_interval=20):
        file_complete_paths = self._get_paths(input_dir)
        logger.info(f"file_complete_paths: {file_complete_paths}")
        logger.info(f"prepare to write all inference results into {output_file_path}")
        logger.info(f"Inferencing using deployment_id: {deployment_id}")
        if args.api_base:
            deployment_status = openai.Deployment.retrieve(id=deployment_id)["status"]
            assert deployment_status == "succeeded", f"deployment_id :{deployment_id} is not successfully deployed, current status: {deployment_status}"
        else:
            logger.info(f"not checking deployment status as we are using OAI services and Deployment operations are only available for the Azure API type.")

        output_file_handle = open(output_file_path, "w+")

        for file_complete_path in file_complete_paths:
            logger.info(f"inferencing file: {file_complete_path}")
            with open(file_complete_path) as ofile:
                for idx, line in tqdm(enumerate(ofile)):
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        res = self.inference_single_prompt(prompt=data[INFERENCE_FILE_PROMPT_KEY], deployment_id=deployment_id, max_tokens=max_tokens, truncate_txt=args.truncate_txt)
                        if res:
                            text = self.retrieve_text(res)
                            to_write = {GPT3_METADATA_KEYWORD: data[INFERENCE_FILE_METADATA_KEY], GPT3_PRED_KEYWORD: [text], GPT3_PROMPT_KEYWORD: data[INFERENCE_FILE_PROMPT_KEY]}
                            output_file_handle.write(json.dumps(to_write) + "\n")
                            if idx % print_interval == 0:
                                logger.info(f"line: {line}, res: {res}")
                                logger.info(f"total_inference_calls: {self.total_inference_calls}, total_successful_inference_calls: {self.total_successful_inference_calls}, total_erroneous_inference_calls: {self.total_erroneous_inference_calls}")

        output_file_handle.close()


class AOAIInferencer_nl2sql_spider(AOAIInferencer):
    def retrieve_text(self, response):
        return response['choices'][0]['text'].replace("@", ".")


def str2bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif val.lower() in ("no", "false", "f", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def updateGlobals(name, value):
    listOfGlobals = globals()
    assert name in listOfGlobals, f"name: {name} is not part of global variables: {listOfGlobals}"
    listOfGlobals[name] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default="./data/3train_uploaded")
    parser.add_argument("--validation_data", default="./data/3validation_uploaded")
    parser.add_argument("--model", default="ada")
    parser.add_argument("--n_epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--learning_rate_multiplier", default=0.1, type=float)
    parser.add_argument("--classification_n_classes", default=5, type=int)
    parser.add_argument("--compute_classification_metrics", default=False, type=bool)
    parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
    parser.add_argument("--api_version", default="2022-06-01-preview")
    parser.add_argument("--fine_tune_metadata", default="./data/4fine_tune")

    #parser.add_argument("--training_file", type=str, default=None)
    #parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--fine_tune_metadata", type=str, default=None)
    parser.add_argument("--use_python_sdk", type=str2bool, default=True)
    parser.add_argument("--check_interval", type=int, default=30, help="Time interval (in seconds) to check job status")
    # finetune hp
    parser.add_argument("--model", default="ada")
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.1)
    parser.add_argument("--sequence_length", type=int, default=None)
    parser.add_argument("--prompt_loss_weight", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lora_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    # inference hp
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--truncate_txt", type=str2bool, default=True)
    # constants hp
    parser.add_argument("--INFERENCE_FILE_PROMPT_KEY", type=str, default=None)
    parser.add_argument("--INFERENCE_FILE_METADATA_KEY", type=str, default=None)
    parser.add_argument("--GPT3_PROMPT_KEYWORD", type=str, default=None)
    parser.add_argument("--GPT3_METADATA_KEYWORD", type=str, default=None)
    parser.add_argument("--GPT3_PRED_KEYWORD", type=str, default=None)
    parser.add_argument("--OUTPUT_FILE_NAME", type=str, default=None)
    parser.add_argument("--MAX_TOKEN_LENGTH", type=int, default=None)


    args = parser.parse_args()

    logger.info(f"resolved args: {args}")
    vars_args = vars(args)
    for gName in ["INFERENCE_FILE_PROMPT_KEY", "INFERENCE_FILE_METADATA_KEY", "GPT3_PROMPT_KEYWORD", "GPT3_METADATA_KEYWORD", "GPT3_PRED_KEYWORD", "OUTPUT_FILE_NAME", "MAX_TOKEN_LENGTH"]:
        args_value = vars_args.get(gName, None)
        if args_value is not None:
            logger.info(f"updating global variable: {gName} with value: {args_value}")
            updateGlobals(gName, args_value)

    openai.api_key = args.api_key
    if args.api_base:
        openai.api_base = args.api_base
        openai.api_type = 'azure'  # hard coding for now
        openai.api_version = '2022-06-01-preview'
        logger.info(f"non-empty args.api_base: {args.api_base}, using AOAI services with openai.api_type: {openai.api_type}, openai.api_version: {openai.api_version}")
    else:
        logger.info(f"empty args.api_base: {args.api_base}, using OAI services")

    args.training_file = FileRetriever.retrieve_file_with_name(args.training_file_name)['id'] if args.training_file_name else None
    args.validation_file = FileRetriever.retrieve_file_with_name(args.validation_file_name)['id'] if args.validation_file_name else None
    finetune_hp = {
        "model": args.model,
    }
    for hp in ["n_epochs", "learning_rate_multiplier", "prompt_loss_weight", "weight_decay", "batch_size", "training_file", "validation_file"]:
        cur_hp = vars(args).get(hp, None)
        if cur_hp:
            logger.info(f"detected non-empty hyper parameter for {hp}: {cur_hp}")
            finetune_hp[hp] = cur_hp
    if args.api_base:
        if args.sequence_length:
            finetune_hp["sequence_length"] = args.sequence_length
        if args.lora_dim:
            finetune_hp["lora_dim"] = args.lora_dim
    logger.info(f"resolved finetune_hp: {finetune_hp}")

    AOAI_finetuning_run = AOAIFinetuningRunCreator(args.finetune_job_id)
    finetune_job_id = AOAI_finetuning_run.create_run(finetune_hp=finetune_hp, use_python_sdk=args.use_python_sdk)
    AOAI_finetuning_cur_status = AOAI_finetuning_run.check_run_status(check_interval=args.check_interval)
    if AOAI_finetuning_cur_status in [AOAI_FINETUNE_CANCEL_STATUS, AOAI_FINETUNE_FAILED_STATUS]:
        raise Exception(f"the finetuning is {AOAI_finetuning_cur_status} in server side, job id: {finetune_job_id}, terminal status:\n{openai.FineTune.retrieve(id=finetune_job_id)}")

    if args.api_base:
        logger.info(f"entering AOAI Deployer")
        AOAI_deployer = AOAIDeployer(finetune_job_id=finetune_job_id)
        deployment = AOAI_deployer.deploy_job()
        AOAI_deployment_cur_status = AOAI_deployer.check_deployment_status(check_interval=args.check_interval)
        logger.info(f"retrieved AOAI_deployment_cur_status: {AOAI_deployment_cur_status}")
        if AOAI_deployment_cur_status in [AOAI_DEPLOYMENT_CANCEL_STATUS, AOAI_DEPLOYMENT_FAILED_STATUS]:
            raise Exception(f"the deployment is {AOAI_deployment_cur_status} in server side, job id: {deployment['id']}, terminal status:\n{openai.Deployment.retrieve(id=deployment['id'])}")
        curr_deployment_id = deployment['id']
    else:
        logger.info(f"OAI finetuned model does not require deployment")
        curr_deployment_id = openai.FineTune.retrieve(id=finetune_job_id)["fine_tuned_model"]


if __name__ == "__main__":
    main()
