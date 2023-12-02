import os
import subprocess

from pandas import read_csv
import openai

class Generator:
    import random

    SYSTEM_PROMPT = """
You have to act like a Python programmer who also an expert in data science.
For visualization use the Lets-Plot library.
Use a consistent style when choosing colors, sizes and so on.
Set colors only in hex format.
Each peace of code with plot must end with an implicit call, without using `show()`, `print()`, etc.
If you are trying to plot some geospatial data, don't forget to use the `coord_map()` function.
And do not try to replace the Lets-Plot by another one visualization library.
    """.strip()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    def __init__(self, *,
                 plots_count=5, queries_limit=5,
                 model="gpt-3.5-turbo", temperature=1.0,
                 is_test_mode=False, seed=None):
        self.plots_count = plots_count
        self.queries_limit = queries_limit
        self.model = model
        self.temperature = temperature
        self.is_test_mode = is_test_mode
        self.random.seed(seed)

    def make_all_good(self, url, *, dataset_name=None, output_name=None, output_title=None, sample=10):
        dataset_name = dataset_name or url.split('/')[-1]
        output_name = output_name or "{0}_eda.ipynb".format(dataset_name.replace(".csv", ""))
        output_title = output_title or "EDA for {0}".format(dataset_name)

        df = read_csv(url)
        if sample is not None:
            df = df.sample(sample)

        self.file_content = self._init_file_content(url, output_title)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._get_planning_message(df)},
        ]
        response = self._get_response(messages)
        messages.append(response)
        for plot_description in response['content'].split('\n'):
            message = {"role": "user", "content": self._get_prompt_for(plot_description)}
            messages.append(message)
            messages.append(self._get_code_for(messages))

        return self._save_result(output_name)

    def _init_file_content(self, data_url, output_title):
        return """
'''
# {title}
'''

# %%
import pandas as pd
from lets_plot import *

LetsPlot.setup_html()

# %%
df = pd.read_csv("{url}")
df.head()
        """.format(
            url=data_url,
            title=output_title
        ).strip()

    def _get_planning_message(self, df):
        return """
I want you to plan the EDA.
List the plots that should be built according to the data below.
The list should consist of short descriptions of the plots, one sentence per plot, each from new line.
The response should not contain anything other than this list. No single extra word, that not part of the list.
The list should consist of no more than {n} sentences.

Here is the data to be visualized:
{data}
        """.format(
            data=df.to_csv(index=False),
            n=self.plots_count
        ).strip()

    def _get_prompt_for(self, plot_description):
        self.file_content = "{0}\n\n# %%\n# {1}".format(self.file_content, plot_description)
        return """
I want you to continue the code below with one additional plot, that correspond to description in the last comment.
In the answer, write only the new piece of code, no explanation - nothing at all but the code.

{0}
        """.format(self.file_content).strip()

    def _get_code_for(self, messages):
        for _ in range(self.queries_limit):
            response = self._get_response(messages)
            code = response['content']
            code = code[code.rfind("ggplot("):].replace('`', '').replace('\r', '').strip()
            file_content = "{0}\n{1}".format(self.file_content, code)
            if self._check_file_content(file_content):
                self.file_content = file_content
                return response
        return ""

    def _get_response(self, messages):
        if self.is_test_mode:
            return self._get_test_response(messages)
        response = openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return response['choices'][0]['message']

    def _get_test_response(self, messages):
        if len(messages) == 2:
            return {
                "role": "assistant",
                "content": "\n".join(["{0}. Plot #{0}".format(i + 1) for i in range(self.plots_count)])
            }
        else:
            content = [
                "import lets_plot as gg\r\n\r\ndata = {\r\n    'cty': [19, 19, 21, 17, 14],\r\n    'hwy': [28, 26, 29, 27, 20]\r\n}\r\n\r\nggplot({'x': [0], 'y': [0]}, aes('x', 'y')) + \\\r\n    geom_point()",
                "ggplot(mapping=aes(x=[0, 1], y=[0, 1])) + \\\r\n    geom_line()",
                "p = ggplot(mapping=aes(x=[0, 1], y=[0, 1])) + \\\r\n    geom_path()\r\np",
                "ggplot(data, aes(x=[0], y=[0])) + \\\r\n    geom_bar()"
            ]
            return {
                "role": "assistant",
                "content": content[self.random.randint(0, len(content) - 1)]
            }

    def _check_file_content(self, file_content):
        file_name = "_response.py"
        with open(file_name, 'w') as f:
            f.write(file_content)
        sproc = subprocess.run(["python", file_name])
        success = sproc.returncode == 0
        os.remove(file_name)
        return success

    def _save_result(self, output_name):
        input_name = output_name.replace(".ipynb", ".py")
        with open(input_name, 'w') as f:
            f.write(self.file_content)
        sproc = subprocess.run(["ipynb-py-convert", input_name, output_name])
        success = sproc.returncode == 0
        os.remove(input_name)
        return success
