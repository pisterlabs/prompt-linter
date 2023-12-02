from aml_keyvault import load_secrets
load_secrets(["OPENAI_API_KEY", "OPENAI_API_BASE"])

# set up openai api
import openai, os
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

import mlflow
import json
import os, re
from patch import log_json_artifact
from langchain import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

meta_prompt_string = """
# You will be given a question posed by a user (Human) and the response that was given by a generative AI bot (System). 
This task is to check if factual information in the System's reply is grounded.

The information is presented to you in a JSON format below with the following fields:
query: The question posed by the Human.
result: The System's reply.
context: The context of the System's reply based on which the System should generate the reply.

- Background: The System does a database searche before responding to the Human. 
The search results are shown in the context section of the JSON below containing the text that was retrieved.
The System's reply should be directly taken from the page_content of the SearchResults, paraphrased from them, or 
inferred from them using common knowledge. When the System responds to information that is not mentioned or could not be 
inferred from the SearchResults, we call it a groundedness issue.

- Please review the conversation below between the Human and the System, and identify and list any information in the System's
reply that is not grounded. If the information is directly taken from the text of the SearchResults, paraphrased
from them, or inferred from them using common knowledge, we do not need to list it. If the information is from outside 
sources, we need to list it. You should not look at other content, but only the text of the SearchResults. 
You can follow the below steps:

    1. Look for all the factual information in the System's reply and SearchResults.

    2. Compare the facts in the reply with the facts in the SearchResults. Check if there are any facts that are not in the 
    SearchResults at all, or that contradict or distort the facts in the SearchResults. If there are, write them down. If 
    there are none, leave it blank. Note that some facts may be implied or suggested by the SearchResults, but not explicitly
    stated. In that case, use your best judgment to decide if the fact is grounded or not. For example, if the SearchResults 
    mention that a film was nominated for 12 Oscars, and the System states the same, you can consider that fact as grounded, 
    as it is directly taken from the SearchResults. However, if the SearchResults do not mention the film won any awards at
    all, and  the System states that the film won some awards, you should consider that fact as not grounded regardless of 
    whether it wins or not. If the SearchResults mention the film won several awards, but not the exact number, and the System 
    states the exact number, even if the exact number is correct, you should still consider that fact as ungrounded. 
    Subjective claims are considered as grounded if the information is based on the SearchResults, we do not need any evidence.

    3. Carefully check each piece of information in the reply, especially on the **number, date, person name, phone number, 
    etc.** to **make sure to list all the ungrounded facts before writing output**.

    4. Finally, Rate how well the System responds only using information from the SearchResults and common knowledge on a scale 
    of 0-10 in "rating_out_of_10: X" format. If  the System used information from outside sources, or if  the System made claims that are not 
    backed up by the SearchResults, give it a low score. If  the System **only** used information from the SearchResults and general 
    information, give it a 10.

```json
{conversation_json}
```
# Output 

Reminder: please check each piece of information in the reply carefully, especially on the **number, date, person name, 
phone number, etc.** to **make sure to list all the ungrounded facts before writing output**.

In JSON format repeat the User's question and the System's reply, then complete the ungrounded facts task described above. 
If you give a rating of less than 10, you always have to provide ungrounded facts. 
The resulting format should look like this:

```json
{{
    "question": "<question>",
    "reply": "<reply>",
    "ungrounded_facts": [
        "<fact 1>",
        "<fact 2>",
        "..."
    ],
    "rating_out_of_10": X
}}
```
Make sure to properly escape the question, reply and ungrounded_facts fields -- the resulting JSON should be valid.
"""

sample_json = """
{
  "query": "how can I register a model from a job using the v2 cli?",
  "result": "To register a model from a job using the v2 CLI, you can use the `az ml model register` command. Here's an example command:\n\n```\naz ml model register --experiment-name <experiment-name> --run-id <run-id> --model-name <model-name> --model-path <model-path>\n```\n\nReplace `<experiment-name>` with the name of the experiment that contains the run you want to register the model from, `<run-id>` with the ID of the run you want to register the model from, `<model-name>` with the name you want to give the registered model, and `<model-path>` with the path to the model file you want to register.\n\nYou can also specify additional parameters such as `--description` to provide a description for the model, `--tags` to add tags to the model, and `--properties` to add custom properties to the model.",
  "context": [
    {
      "page_content": "\n# Work with models in Azure Machine Learning\n\n[!INCLUDE [dev v2](../../includes/machine-learning-dev-v2.md)]\n\nAzure Machine Learning allows you to work with different types of models. In this article, you learn about using Azure Machine Learning to work with different model types, such as custom, MLflow, and Triton. You also learn how to register a model from different locations, and how to use the Azure Machine Learning SDK, the user interface (UI), and the Azure Machine Learning CLI to manage your models.\n\n> [!TIP]\n> If you have model assets created that use the SDK/CLI v1, you can still use those with SDK/CLI v2. Full backward compatibility is provided. All models registered with the V1 SDK are assigned the type `custom`.\n\n## Prerequisites\n\n* An Azure subscription. If you don't have an Azure subscription, create a free account before you begin. Try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/).\n* An Azure Machine Learning workspace.\n* The Azure Machine Learning [SDK v2 for Python](https://aka.ms/sdk-v2-install).\n* The Azure Machine Learning [CLI v2](how-to-configure-cli.md).\n\nAdditionally, you will need to:\n\n# [Azure CLI](#tab/cli)\n\n- Install the Azure CLI and the ml extension to the Azure CLI. For more information, see [Install, set up, and use the CLI (v2)](how-to-configure-cli.md).\n\n# [Python SDK](#tab/python)\n\n- Install the Azure Machine Learning SDK for Python\n    \n    ```bash\n    pip install azure-ai-ml\n    ```\n\n## Supported paths\n\nWhen you provide a model you want to register, you'll need to specify a `path` parameter that points to the data or job location. Below is a table that shows the different data locations supported in Azure Machine Learning and examples for the `path` parameter:\n\n\n|Location  | Examples  |\n|---------|---------|\n|A path on your local computer     | `mlflow-model/model.pkl`         |\n|A path on an AzureML Datastore   |   `azureml://datastores/<datastore-name>/paths/<path_on_datastore>`      |\n|A path from an AzureML job   |   `azureml://jobs/<job-name>/outputs/<output-name>/paths/<path-to-model-relative-to-the-named-output-location>`      |\n|A path from an MLflow job   |   `runs:/<run-id>/<path-to-model-relative-to-the-root-of-the-artifact-location>`      |\n|A path from a Model Asset in AzureML Workspace  | `azureml:<model-name>:<version>`|\n|A path from a Model Asset in  AzureML Registry  | `azureml://registries/<registry-name>/models/<model-name>/versions/<version>`|\n\n## Supported modes\n\nWhen you run a job with model inputs/outputs, you can specify the *mode* - for example, whether you would like the model to be read-only mounted or downloaded to the compute target. The table below shows the possible modes for different type/mode/input/output combinations:\n\nType | Input/Output | `upload` | `download` | `ro_mount` | `rw_mount` | `direct` \n------ | ------ | :---: | :---: | :---: | :---: | :---: \n`custom` file  | Input  |   |  |   |  |    \n`custom` folder    | Input |   | ✓ | ✓  | |✓  \n`mlflow`    | Input |   | ✓ |  ✓ |   |   \n`custom` file | Output  | ✓  |   |    | ✓  | ✓   \n`custom` folder    | Output | ✓  |   |   | ✓ | ✓  \n`mlflow`   | Output | ✓  |   |    | ✓  | ✓ \n\n\n### Follow along in Jupyter Notebooks\n\nYou can follow along this sample in a Jupyter Notebook. In the [azureml-examples](https://github.com/azure/azureml-examples) repository, open the notebook: [model.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/assets/model/model.ipynb).\n\n## Create a model in the model registry\n\n[Model registration](concept-model-management-and-deployment.md) allows you to store and version your models in the Azure cloud, in your workspace. The model registry helps you organize and keep track of your trained models.\n\nThe code snippets in this section cover how to:\n\n* Register your model as an asset in Machine Learning by using the CLI.\n* Register your model as an asset in Machine Learning by using the SDK.\n* Register your model as an asset in Machine Learning by using the UI.\n",
      "metadata": {
        "sourcefile": "UI/2023-04-06_191207_UTC/simple-4000-100/how-to-manage-models-0.md"
      }
    },
    {
      "page_content": "The Bandit early termination policy is also used. This policy terminates poor performing trials; that is, those trials that are not within 20% slack of the best performing trial, which significantly saves compute resources.\n\n# [Azure CLI](#tab/cli)\n\n[!INCLUDE [cli v2](../../includes/machine-learning-cli-v2.md)]\n\n```yaml\nsweep:\n  sampling_algorithm: random\n  early_termination:\n    type: bandit\n    evaluation_interval: 2\n    slack_factor: 0.2\n    delay_evaluation: 6\n```\n\n```yaml\nsearch_space:\n  - model_name:\n      type: choice\n      values: [yolov5]\n    learning_rate:\n      type: uniform\n      min_value: 0.0001\n      max_value: 0.01\n    model_size:\n      type: choice\n      values: [small, medium]\n\n  - model_name:\n      type: choice\n      values: [fasterrcnn_resnet50_fpn]\n    learning_rate:\n      type: uniform\n      min_value: 0.0001\n      max_value: 0.001\n    optimizer:\n      type: choice\n      values: [sgd, adam, adamw]\n    min_size:\n      type: choice\n      values: [600, 800]\n```\n\n# [Python SDK](#tab/python)\n [!INCLUDE [sdk v2](../../includes/machine-learning-sdk-v2.md)]\n\n\n[!Notebook-python[] (~/azureml-examples-main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb?name=sweep-settings)]\n\n[!Notebook-python[] (~/azureml-examples-main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb?name=search-space-settings)]\n\n\nOnce the search space and sweep settings are defined, you can then submit the job to train an image model using your training dataset.\n\n# [Azure CLI](#tab/cli)\n\n[!INCLUDE [cli v2](../../includes/machine-learning-cli-v2.md)]\n\nTo submit your AutoML job, you run the following CLI v2 command with the path to your .yml file, workspace name, resource group and subscription ID.\n\n```azurecli\naz ml job create --file ./hello-automl-job-basic.yml --workspace-name [YOUR_AZURE_WORKSPACE] --resource-group [YOUR_AZURE_RESOURCE_GROUP] --subscription [YOUR_AZURE_SUBSCRIPTION]\n```\n\n# [Python SDK](#tab/python)\n[!INCLUDE [sdk v2](../../includes/machine-learning-sdk-v2.md)]\n\nWhen you've configured your AutoML Job to the desired settings, you can submit the job.\n\n[!Notebook-python[] (~/azureml-examples-main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb?name=submit-run)]\n\n\nWhen doing a hyperparameter sweep, it can be useful to visualize the different trials that were tried using the HyperDrive UI. You can navigate to this UI by going to the 'Child jobs' tab in the UI of the main automl_image_job from above, which is the HyperDrive parent job. Then you can go into the 'Child jobs' tab of this one.\n\nAlternatively, here below you can see directly the HyperDrive parent job and navigate to its 'Child jobs' tab:\n\n# [Azure CLI](#tab/cli)\n\n[!INCLUDE [cli v2](../../includes/machine-learning-cli-v2.md)]\n```yaml\nCLI example not available, please use Python SDK.\n```\n\n\n# [Python SDK](#tab/python)\n[!INCLUDE [sdk v2](../../includes/machine-learning-sdk-v2.md)]\n\n```python\nhd_job = ml_client.jobs.get(returned_job.name + '_HD')\nhd_job\n```\n\n\n## Register and deploy model\n\nOnce the job completes, you can register the model that was created from the best trial (configuration that resulted in the best primary metric). You can either register the model after downloading or by specifying the azureml path with corresponding jobid.  \n\n### Get the best trial\n\n\n# [Azure CLI](#tab/cli)\n\n[!INCLUDE [cli v2](../../includes/machine-learning-cli-v2.md)]\n```yaml\nCLI example not available, please use Python SDK.\n```\n\n\n# [Python SDK](#tab/python)\n[!INCLUDE [sdk v2](../../includes/machine-learning-sdk-v2.md)]\n\n[!Notebook-python[] (~/azureml-examples-main/sdk/python/jobs/automl-standalone-jobs/automl-image-object-detection-task-fridge-items/automl-image-object-detection-task-fridge-items.ipynb?name=best_run)] \n",
      "metadata": {
        "sourcefile": "UI/2023-04-06_191207_UTC/simple-4000-100/tutorial-auto-train-image-models-387.md"
      }
    },
    {
      "page_content": "\n\n# What is an Azure Machine Learning workspace?\n\nThe workspace is the top-level resource for Azure Machine Learning, providing a centralized place to work with all the artifacts you create when you use Azure Machine Learning.  The workspace keeps a history of all training runs, including logs, metrics, output, and a snapshot of your scripts. You use this information to determine which training run produces the best model.  \n\nOnce you have a model you like, you register it with the workspace. You then use the registered model and scoring scripts to deploy to an [online endpoint](concept-endpoints.md) as a REST-based HTTP endpoint.\n\n## Taxonomy \n\n+ A workspace can contain [Azure Machine Learning compute instances](concept-compute-instance.md), cloud resources configured with the Python environment necessary to run Azure Machine Learning.\n\n+ [User roles](how-to-assign-roles.md) enable you to share your workspace with other users, teams, or projects.\n+ [Compute targets](concept-compute-target.md) are used to run your experiments.\n+ When you create the workspace, [associated resources](#associated-resources) are also created for you.\n+ Jobs are training runs you use to build your models.  You can organize your jobs into Experiments.\n+ [Pipelines](concept-ml-pipelines.md) are reusable workflows for training and retraining your model.\n+ [Data assets](concept-data.md) aid in management of the data you use for model training and pipeline creation.\n+ Once you have a model you want to deploy, you create a registered model.\n+ Use the registered model and a scoring script to create an [online endpoint](concept-endpoints.md).\n\n## Tools for workspace interaction\n\nYou can interact with your workspace in the following ways:\n\n+ On the web:\n    + [Azure Machine Learning studio ](https://ml.azure.com) \n    + [Azure Machine Learning designer](concept-designer.md) \n+ In any Python environment with the [Azure Machine Learning SDK for Python](https://aka.ms/sdk-v2-install).\n+ On the command line using the Azure Machine Learning [CLI extension](how-to-configure-cli.md)\n+ [Azure Machine Learning VS Code Extension](how-to-manage-resources-vscode.md#workspaces)\n\n## Machine learning with a workspace\n\nMachine learning tasks read and/or write artifacts to your workspace.\n\n+ Run an experiment to train a model - writes job run results to the workspace.\n+ Use automated ML to train a model - writes training results to the workspace.\n+ Register a model in the workspace.\n+ Deploy a model - uses the registered model to create a deployment.\n+ Create and run reusable workflows.\n+ View machine learning artifacts such as jobs, pipelines, models, deployments.\n+ Track and monitor models.\n\n## Workspace management\n\nYou can also perform the following workspace management tasks:\n\n| Workspace management task           | Portal      | Studio      | Python SDK  | Azure CLI   | VS Code     |\n|-------------------------------------|-------------|-------------|-------------|-------------|-------------|\n| Create a workspace                  | **&check;** | **&check;** | **&check;** | **&check;** | **&check;** |\n| Manage workspace access             | **&check;** |             |             | **&check;** |             |\n| Create and manage compute resources | **&check;** | **&check;** | **&check;** | **&check;** | **&check;** |\n| Create a compute instance           |             | **&check;** | **&check;** | **&check;** | **&check;** |\n\n> [!WARNING]\n> Moving your Azure Machine Learning workspace to a different subscription, or moving the owning subscription to a new tenant, is not supported. Doing so may cause errors.\n\n## Create a workspace\n\nThere are multiple ways to create a workspace:  \n\n* Use [Azure Machine Learning studio](quickstart-create-resources.md) to quickly create a workspace with default settings.\n* Use the [Azure portal](how-to-manage-workspace.md?tabs=azure-portal#create-a-workspace) for a point-and-click interface with more options. \n* Use the [Azure Machine Learning SDK for Python](how-to-manage-workspace.md?tabs=python#create-a-workspace) to create a workspace on the fly from Python scripts or Jupyter notebooks.\n",
      "metadata": {
        "sourcefile": "UI/2023-04-06_191207_UTC/simple-4000-100/concept-workspace-0.md"
      }
    }
  ]
}
"""

from langchain.chat_models import AzureChatOpenAI
from patch import patch_langchain, log_function_call

def robust_result_parse(result, conversation_context):
    try: 
        result_dict = json.loads(result.content)
        rating = result_dict["rating_out_of_10"]
        assert isinstance(rating, int)
    except Exception as e:
        print("Error parsing result json", e)
        print(result.content)
        pattern = r"\"rating_out_of_10\":\s*(.*?)\n"
        result_dict = conversation_context
        rating = re.search(pattern, result.content)
        if rating:
            try:
                rating = int(rating.group(1))
            except Exception:
                rating = 0
        else:   
            rating = 0

        result_dict["rating_out_of_10"] = int(rating)
        result_dict["error"] = f"Error parsing result json: {e}"

    return result_dict
  

def groundedness(conversation_context, meta_prompt, verbose: bool = False): 

    if verbose:
        # patch langchain to log function calls
        patch_langchain()
        
    prompt_template = PromptTemplate(
        input_variables=["conversation_json"],
        template=meta_prompt,
    )

    # create the llm
    llm = AzureChatOpenAI(
        deployment_name="gpt-4-32k",
        temperature=0,
        openai_api_version="2023-03-15-preview",
        model_name="gpt-4-32k"
    )

    prompt = prompt_template.format(conversation_json=str(conversation_context))

    # execute the prompt with the llm
    result = llm([HumanMessage(content=prompt)])

    # parse the result from json inot a dict
    result_dict = robust_result_parse(result, conversation_context)

    return result_dict

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    
def load_textfile(filename):
    with open(filename, 'r') as f:
        return f.read()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation_context", type=str, default="data/amldocs/groundedness/conversation_context.json")
    parser.add_argument("--meta_prompt", type=str, default=None)
    parser.add_argument("--result", type=str, default="data/amldocs/groundedness/result.json")
    parser.add_argument("--no-log", action='store_true')
    args = parser.parse_args()

    verbose = not args.no_log

    if verbose:
        mlflow.log_param("conversation_context", args.conversation_context)
        mlflow.log_param("meta_prompt", args.meta_prompt)
        mlflow.log_param("result", args.result)

    conversation_context = load_json(args.conversation_context)

    if args.meta_prompt is None:
        meta_prompt = meta_prompt_string
    else:
        meta_prompt = load_textfile(args.meta_prompt)

    result = groundedness(conversation_context, meta_prompt_string, verbose=True)

    if verbose:
        log_json_artifact(result, "result.json")

    print(json.dumps(result, indent=4))

    # save the result to a file
    with open(args.result, 'w') as f:
        json.dump(result, f, indent=4)
        




