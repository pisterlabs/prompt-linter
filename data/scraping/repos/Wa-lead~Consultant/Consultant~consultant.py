from typing import Callable
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from IPython.display import display, HTML
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import inspect
import guidance
import openai
import re

from Consultant.prompt import Prompt

@dataclass
class ConsultantRequest(DataClassJsonMixin):
    prompt: Prompt
    code: str
    model: str = 'gpt-3.5-turbo'
    
    
class Consultant:
    
    # _START_TAG = '<|im_start|>assistant'
    # _END_TAG = '<|im_end|>'
    
    _USER_NOTE = f'Rewrite the code please. Your response should STRICTLY only contain your suggested python code. Remove the decorator.'
    _SYSTEM_NOTE = ('Your role and task have been defined. Carefully read the '
                    'description and instructions and provide a solution that '
                    'fulfills the requirements. At times, you may encounter '
                    'incomplete code. Infer the intended functionality, e.g., '
                    'from the function name, and complete it.')

    @staticmethod
    def consult(prompt: Prompt) -> Callable:
        def wrapper(func: Callable):
            code = inspect.getsource(func).replace('pass', '')
            suggested_code = Consultant._call(prompt, code)
            Consultant.__ipython_display__(suggested_code)
            return lambda *args, **kwargs: None
        return wrapper
    
    @staticmethod
    def _call(prompt: Prompt, code: str) -> str:
        request = ConsultantRequest(prompt=prompt, code=code)
        response = Consultant._generate(request)
        # return Consultant._extract_code(response)
        return response
    
    
    # @staticmethod
    # def _generate(request: ConsultantRequest) -> str:
    #     guidance.llm = guidance.llms.OpenAI(request.model)
    #     prompt = guidance(
    #     '''{{#system~}}
    #             {{SYSTEM_NOTE}}}
    #             Your description: {{description}}
    #             Your goal: {{instruction}}
    #         {{~/system}}
    #     {{#user~}}
    #         {{USER_NOTE}}
    #         {{code}}
    #     {{~/user}}
    #     {{#assistant~}}
    #         {{gen 'code'}}
    #     {{~/assistant}}''', caching=False, silent=True)
    #     prompt = prompt(SYSTEM_NOTE=Consultant._SYSTEM_NOTE,
    #                     USER_NOTE=Consultant._USER_NOTE,
    #                     instruction=request.prompt.instruction,
    #                     description= request.prompt.description,
    #                     code=request.code
    #     )
    #     return prompt().text
    
    def _generate(request: ConsultantRequest) -> str:
        response = openai.ChatCompletion.create(
        model=request.model,
        messages=[
                {"role": "system", "content": f"{Consultant._SYSTEM_NOTE}"},
                {"role": "system", "content": f'{request.prompt.instruction}'},
                {"role": "user", "content": f"{Consultant._USER_NOTE} \n" + request.code + "\n"},
            ]
        )
        return response.choices[0]['message']['content']
            
        
    # @staticmethod
    # def _extract_code(response: str) -> str:
    #     return re.search(f'{Consultant._START_TAG}(.*){Consultant._END_TAG}', response).group(1)
    
    @staticmethod    
    def __ipython_display__(code: str) -> None:

        # Syntax highlighting using pygments
        lexer = PythonLexer()
        formatter = HtmlFormatter(style='colorful')
        # highlighted_code = highlight(a, lexer, formatter)

        # HTML code with copy button
        html_code = f'''
        <div style="position: relative;">
            <pre style="background-color: #black; padding: 10px;">{code}</pre>
            <button id="copy-button" onclick="copyToClipboard()" style="position: absolute; top: 0; right: 0; margin: 10px; padding: 10px; background-color: #000; color: #fff; border: none; border-radius: 5px; cursor: pointer;">Copy</button>
        </div>
        

        <script>
        function copyToClipboard() {{
            var text = `{code.strip()}`; // Stripped version of the code
            var tempInput = document.createElement('textarea');
            tempInput.value = text;
            document.body.appendChild(tempInput);
            tempInput.select();
            document.execCommand('copy');
            document.body.removeChild(tempInput);
                
            const copyButton = document.getElementById('copy-button');
            copyButton.textContent = 'Copied!';
            setTimeout(() => {{
                copyButton.textContent = 'Copy';
            }}, 1000);
    }}

        </script>
        '''
        
        
        # Display the HTML code
        display(HTML(html_code))
        


