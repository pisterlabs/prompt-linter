import subprocess
import os
import pandas as pd
import yaml
from langchain.prompts import PromptTemplate


class BtCopilot:
    """
    backtrader copilote provides a user friendly, AI enabled interface to generate and analyse quantitative trading
    strategy backtest using the backtrader framework in Python.

    bt_copilot acts as the front-end facing client to interact with.
    """

    def __init__(self, coding_agent):
        '''
        Constructs a new instance of bt_copilot.

        :param coding_agent: intermediate client between bt-copilot and the llm. Orchestrates the implementation of code
        :param prompt_library_file: contains the prompt templates.
        '''

        # load settings
        with open('settings.yaml') as f:
            self.settings = yaml.safe_load(f)

        # initiate coding agent
        self.coding_agent = coding_agent

        # load prompt library
        self.prompt_library = self.load_prompt_library()

        # initiate code kept in memory
        self.code = ''

        # initiate prompt memory
        self.prompt_elements = {'datapipeline' : '',
                                'strategy' : '',
                                'analysers' : '',
                                'custom' : ''}

        # initiate compiled prompt
        self.compiled_prompt = ''

    def load_prompt_library(self):
        '''
        loads the prompt library file into the bt_copilote client
        :return: Pandas dataframe containing prompt library
        '''
        # read the prompt library CSV file
        try:
            df = pd.read_csv(os.path.join(self.settings["resources_dir"], self.settings["prompt_lib"]))
            return df
        except FileNotFoundError:
            print(f"Prompt library file not found in {self.settings['resources_dir']}/{self.settings['prompt_lib']}")
            return None
    def load_code(self, file_path):
        '''
        loads existing backtesting code into the bt_copilote client
        :param file_path: string
        :return:
        '''
        try:
            with open(file_path, 'r') as file:
                self.code = file.read()
        except FileNotFoundError:
            print(f"Code file not found at {file_path}")
    def save_code(self):
        '''
        saves code that is stored in memory of bt_copilote client to file
        :return:
        '''
        # Open the file in write mode
        with open(os.path.join(self.settings["output_dir"], f"{self.settings['project_name']}.py"), 'w') as f:
            # Write code to file
            f.write(self.code)
        return
    def _get_prompt_template(self, goal_code):
        '''
        pulls prompt template for specific goal_code from prompt library.
        Example for goal_code: 'build_strategy'
        :param goal_code: string
        :return: prompt template: string
        '''
        # find the row with the key "suggest_strategy_improvement"
        df = self.prompt_library
        row = df.loc[df['goal_code'] == goal_code]

        # store "prompt" and "temperature" in variables
        prompt = row['prompt'].values[0]

        prompt = {'prompt' : prompt}

        return prompt
    def compose_prompt_from_elements(self):
        '''
        composes the individual elements of the prompt into one prompt describing the entire desired backtest.
        This prompt can be sent to the coding agent to obtain the backtesting code.
        :return:
        '''

        # combine prompt elements
        combined_prompt = ''.join(self.prompt_elements.values())

        # load general context
        submission_prompt_template = self._get_prompt_template(goal_code='submission_prompt_template')['prompt']
        coding_context = self._get_prompt_template(goal_code='coding_context')['prompt']

        # create langchain prompt_template
        prompt_template = PromptTemplate(
            input_variables=["context","combined_prompt"],
            template = submission_prompt_template,
        )

        # store formatted prompt with context in client memory
        self.prompt = prompt_template.format(context = coding_context,
                                             combined_prompt = combined_prompt)
        return
    def _build_prompt(self, goal_code, user_input):
        '''
        builds a prompt based on a prompt template from the prompt library, general context (backtrader, python)
        and user input.
        :param goal_code: str
        :param user_input: str
        :return: formatted prompt: str
        '''
        # load prompt template
        prompt_from_library = self._get_prompt_template(goal_code=goal_code)
        template = prompt_from_library['prompt']

        # create langchain prompt_template
        prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=template,
        )

        # insert user input into template
        formatted_prompt = prompt_template.format(user_input=user_input)
        return formatted_prompt
    def load_boilerplate(self, boilerplate_type = 'basic'):
        '''
        load boilerplate code from a selection of backtrader boilerplates in the resources directory
        :param type: str
        :return:
        '''

        self.code = self.load_code(f'{self.settings["resources_dir"]}/boilerplate_{boilerplate_type}.py')
        return
    def set_datapipeline(self, user_input):
        '''
        sets the datapipeline part of the composed prompt
        :param user_input: str
        :return:
        '''
        # generate boilerplate code
        new_prompt = self._build_prompt(user_input=user_input,
                                       goal_code='set_datapipeline')

        # append prompt element to entire prompt
        self.prompt_elements['datapipeline'] = new_prompt
        return
    def set_strategy(self, user_input):
        '''
        sets the strategy part of the composed prompt
        :param user_input defining the contribution of the user to be filled into the prompt template: str
        :return:
        '''
        # generate boilerplate code
        new_prompt = self._build_prompt(user_input=user_input,
                                       goal_code='set_strategy')

        # append prompt element to entire prompt
        self.prompt_elements['strategy'] = new_prompt
        return
    def set_analysers(self, user_input):
        '''
        sets the analyzer part of the composed prompt
        :param user_input: str
        :return:
        '''
        # generate boilerplate code
        new_prompt = self._build_prompt(user_input=user_input,
                                       goal_code='set_analysers')

        # append prompt element to entire prompt
        self.prompt_elements['analysers'] = new_prompt
        return
    def set_custom_prompt(self,prompt):
        '''
        sets the custom part of the composed prompt
        :param user_input: str
        :return:
        '''
        self.prompt_elements['custom'] = prompt
        return
    def build_code_from_prompt(self):
        '''
        builds executable python backtesting code from the compiled prompt (compilation of prompt elements) by calling
        the coding agent on compiled_prompt.
        :param temperature: flaot
        :return:
        '''
        self.code = self.coding_agent.code(prompt = self.compiled_prompt,
                                           temperature = self.settings['coding_temp'])
        return
    def get_strategy_feedback(self , feedback_basis = 'code'):
        '''
        provides natural language feedback on the trading strategy.
        This function can be called on either
        A) the python code, or
        B) the natural language description of the strategy.

        In case A) the strategy is extracted from the code, and feedback is provided based on this understanding of the
        trading rules. The code stored in the bt_copilote client is used. Useful to load coad and analyze it.

        In case B) the natural language description of the strategy is used. Useful to get feedback on strategy before
        building code.
        :param basis: str
        :return: strategy_feedback: str
        '''

        # obtain feedback based on backtesting code (llm extracts strategy from code)
        if feedback_basis == 'code':

            # get prompt for strategy feedback on stored strategy description
            prompt = self._build_prompt(goal_code='get_strategy_feedback_from_code',
                                       user_input= self.code)

        # obtain feedback based on natural language description of strategy
        elif feedback_basis == 'description':

            # get prompt for strategy feedback on stored strategy description
            prompt = self._build_prompt(goal_code='get_strategy_feedback_from_description',
                                       user_input= self.prompt_elements['strategy'])

        # execute LLM call
        strategy_feedback = self.coding_agent.simple_LLMcall(prompt = prompt,
                                                             temperature= self.settings['strategy_feedback_temp'])

        return strategy_feedback
    def get_strategy_description(self):
        '''
        generates a natural language description of the trading strategy based on the loaded code.
        useful to quickly understand an implemented strategy
        :return: strategy_description: str
        '''

        # get prompt for strategy description based on code
        prompt = self._build_prompt(goal_code='get_strategy_description',
                                   user_input=self.code)

        # execute LLM call
        strategy_description = self.coding_agent.simple_LLMcall(prompt=prompt,
                                                                temperature= self.settings['strategy_descr_temp'])
        return strategy_description
    def visualise_strategy(self, vis_basis='code'):

        if vis_basis == 'code':
            # get prompt for strategy description based on code
            prompt = self._build_prompt(goal_code='get_strategy_visualisation_from_code',
                                        user_input=self.code)

        elif vis_basis == 'description':
            # get prompt for strategy description based on code
            prompt = self._build_prompt(goal_code='get_strategy_visualisation_from_description',
                                        user_input=self.prompt_elements['strategy'])

        # execute LLM call
        visualisation_code = self.coding_agent.simple_LLMcall(prompt=prompt,
                                                              temperature=self.settings['vis_strat_temp'])

        # Define the path where the file will be saved
        file_path = os.path.join(self.settings["output_dir"], f'{self.settings["project_name"]}_plotscript.py')

        # Write the code to a Python file at the defined path
        with open(file_path, 'w') as f:
            f.write(visualisation_code)

        # Run the Python file using subprocess.run()
        try:
            result = subprocess.run(["python", file_path], capture_output=True, text=True)

            # Print the output of the subprocess
            print('Plotted flow chart')
            print(result.stdout)
        except Exception as e:
            print(
                f"An error occurred while visualising the strategy. Ensure graphviz is installed on your system: {str(e)}")
        return
    def autopilot(self):
        '''
        runs the bt_copilote autopilot, which guides user through required questions to set up a backtest.
        :return:
        '''

        # Intro
        logo = '''
 _     _                       _ _       _   
| |__ | |_      ___ ___  _ __ (_) | ___ | |_ 
| '_ \| __|    / __/ _ \| '_ \| | |/ _ \| __|
| |_) | |_    | (_| (_) | |_) | | | (_) | |_ 
|_.__/ \__|    \___\___/| .__/|_|_|\___/ \__|
                        |_|                  
        '''
        print(logo)
        print('Welcome to backtrader autopilot - AI enabled algrotrading outputs')
        print('I will guide you through the steps of setting up your backtest in python backtrader.')
        print('---')

        # Project name
        project_name = input("Please enter the name of your project: ")
        self.settings['project_name'] = project_name

        # Data pipeline
        datapipeline = input("Please describe the data sources you would like to load for your backtest: ")
        self.prompt_elements['datapipeline'] = datapipeline

        # Strategy
        strategy = input("Please describe your trading strategy rules: ")
        self.prompt_elements['strategy'] = strategy

        # Analyzers
        analyzers = input("Please describe which analyzers you would like to include: ")
        self.prompt_elements['analyzers'] = analyzers

        print('---')
        print("That's a wrap!")
        print("Would you like to build the code?")

        return
    def run_backtest(self):
        '''
        Runs the backtest by executing the python code.
        :return:
        '''
        try:
            subprocess.run(["python3", f"{self.settings['project_name']}.py"])
        except Exception as e:
            print(f"An error occurred when trying to execute backtest: {str(e)}")



