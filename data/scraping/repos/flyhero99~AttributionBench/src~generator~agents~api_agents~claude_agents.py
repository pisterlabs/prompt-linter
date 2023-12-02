import anthropic
from src.agent import Agent
from copy import deepcopy
from tqdm import tqdm

claudeapi_path = ""

class Claude(Agent):
    def __init__(self, api_args=None, batch_size = 1, **config):
        if not api_args:
            api_args = {}
        api_args = deepcopy(api_args)
        self.key = api_args.pop("key", None) or os.getenv('Claude_API_KEY')
        api_args["model"] = api_args.pop("model", None)
        if not self.key:
            try:
                with open(claudeapi_path) as f:
                    self.key = f.readline().strip()
            except:
                raise ValueError("Claude API KEY is required, please assign api_args.key or set OPENAI_API_KEY environment variable.")
        if not api_args["model"]:
            raise ValueError("Claude model is required, please assign api_args.model.")
        self.api_args = api_args
        if not self.api_args.get("stop_sequences"):
            self.api_args["stop_sequences"] = [anthropic.HUMAN_PROMPT]
        self.batch_size = 1
        super.__init__(**config)

    def inference(self, historys) -> str:
        pbar = tqdm(total=len(historys),desc="Inference claude")
        response_all = []
        c = anthropic.Client(self.key)
        for batch in Agent.batch_sample(historys,self.batch_size):
            resp = c.completion(
                prompt=batch[0],
                **self.api_args
            )
            response_all.append(resp)
            pbar.update(1)
        return response_all
