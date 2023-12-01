from openai import OpenAI
import json

PROJECT_ROOT = "/Users/ziqi/Work/pyc/workaround/refcountChecker/"

class LLMPlugin:
    def __init__(self):
        # read pretrain prompt from pretrain_prompt.txt

        # read LLMPlugin settings
        with open(PROJECT_ROOT + 'settings.json') as f:
            data = json.load(f)
            self.model = data['LLMPlugin']['model']
            self.cfile = data['ToolConfig']['module']
            self.apiKey = data['LLMPlugin']['apiKey']

        pretrain_path = PROJECT_ROOT + 'LLMPlugin/pretrain_prompt.txt'
        with open(pretrain_path, 'r') as f:
            self._pretrain_prompt = f.read()
            self.messages = [{"role": "system", "content": self._pretrain_prompt}]

        with open(self.cfile) as f:
            self.c_content = f.read()
            self.messages.append({"role": "user", "content": self.c_content})

    def generate(self):
        client = OpenAI(api_key=self.apiKey)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )
        return response.choices[0].message


if __name__ == "__main__":
    print("\n====================\nContent below is generated by LLMPlugin\n====================\n")
    plugin = LLMPlugin()
    print(plugin.generate().content)
