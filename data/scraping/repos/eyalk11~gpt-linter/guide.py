import os

import guidance


class Guidance:
    @staticmethod
    def set_key(args):
        guidance.llm = guidance.llms.OpenAI(args.model, api_key=os.environ['OPEN_AI_KEY'])


    @staticmethod
    def guide_for_errors(args):

        text = '''
    {{#system~}}
        You are a helpful assistant. You will be given a file and an issue. You need to come up with fixes for the issue, even if it is a minor issue.
        {{~/system}}
    {{~#user}}
    Given this {{filename}}: .
            {{file}}.
    A list of issues will be given
    {{~/user}}
    
    {{#each errors}}
      {{~#user}}
      What is the fix for this issue on {{filename}}?
              {{this}}
       Be sure to inspect the entire relevant function before suggesting a fix. 
        Be short and precise regarding the fix, and refer to the change in code. In your answer, you shouldn't repeat instructions given to you,  and you shouldn't include more than 5 lines of code.
      {{~/user}}
      {{#assistant~}}
        {{gen 'fix' list_append=True temperature=%d max_tokens=%d}}
        {{~/assistant}}
    {{/each~}}''' % (args.temperature_per_fix, args.max_tokens_per_fix)
        return  guidance(text, log=True, caching=False)  # type: ignore

    @staticmethod
    def guide_for_fixes(args):
        text = '''
            {{#system~}}
            You are a helpful assistant. You will be given a list of corrections to do in a file, and will update the file accordingly. 
            Reply only with xml that has the following format:  
            ```xml
            <pythonfile>the updated file content after the corrections are made</pythonfile>
            ```
            {{~/system}}
            {{#user~}}
            This is the file:
            {{file}}
            Those are the fixes
            {{#each fixes}}- {{this}}
                {{/each~}}
            Make sure you apply all the corrections in the resulted file, even if the issues aren't clear.
            {{~/user}}
    
            {{#assistant~}}
            {{gen 'fixedfile' temperature=%d max_tokens=%d}}
            {{~/assistant~}}
        '''
        return guidance(text % (args.temperature_for_file , args.max_tokens_for_file), log=True, caching=False)  # type: ignore
