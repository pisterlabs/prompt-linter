import wandb
from datetime import datetime
from langchain.callbacks import WandbCallbackHandler, StdOutCallbackHandler
from wandb.integration.langchain import WandbTracer
import spacy



''' Tracking config for MLOps Tools:
- Langchain
- WandB
- Humanloop

'''

'''
 WandB Callback
Parameters:
    job_type (str): The type of job.
    project (str): The project to log to.
    entity (str): The entity to log to.
    tags (list): The tags to log.
    group (str): The group to log to.
    name (str): The name of the run.
    notes (str): The notes to log.
    visualize (bool): Whether to visualize the run.
    complexity_metrics (bool): Whether to log complexity metrics.
    stream_logs (bool): Whether to stream callback actions to W&B
'''

session_group = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
'''wandbTrace = WandbTracer()
wandbTrace.init()'''

    
'''wandb_callback = WandbCallbackHandler(
    job_type="inference",
    project="langchain_callback_demo",
    group=f"minimal_{session_group}",
    name="llm",
    tags=["test"],
)
callbacks = [StdOutCallbackHandler(), wandb_callback]
'''