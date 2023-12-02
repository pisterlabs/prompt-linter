# create an cadcad-gpt agent class which takes in the model, simulation, experiment, doc string and can return the experiment.run() function
import openai
# tool descriptions

function_descriptions_multiple = [
    {
        "name": "change_param",
        "description": "Changes the parameter of the cadcad simulation and returns dataframe as a global object. The parameter must be in this list:" + str(model.params.keys()),
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "parameter to change. choose from the list" + str(model.params.keys()),
                },
                "value": {
                    "type": "string",
                    "description": "value to change the parameter to, eg. 0.1",
                },
            },
            "required": ["param", "value"],
        },
    },
    {
        "name": "model_info",
        "description": "quantitative values of current state of the simulation parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "type of information to print. choose from the list: " + str(model.params.keys()),
                },
            },
            "required": ["param"],
        },
    },
    {
        "name": "analyze_dataframe",
        "description": "Use this whenever a quantitative question is asked about the dataframe",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "model_documentation",
        "description": "use when asked about documentation of the model has information about what the model is, assumptions made, mathematical specs, differential model specs etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    }
]




class CadcadGPTAgent:
    def __init__(self, model, simulation, experiment, docstring):
        self.model = model
        self.simulation = simulation
        self.experiment = experiment
        self.docstring = docstring


    def run(self):
        df1 = pd.DataFrame(self.experiment.run())
        return df1
    

    def executor_agent(self, prompt, function_descriptions = function_descriptions_multiple):
        """Give LLM a given prompt and get an answer."""

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": prompt}],
            # add function calling
            functions=function_descriptions_multiple,
            function_call="auto",  # specify the function call
        )

        output = completion.choices[0].message
        return output
    
    def planner_agent(self, prompt):
        """Give LLM a given prompt and get an answer."""

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                "role": "system",
                "content": '''
                You will be provided with a question by the user that is trying to run a cadcad python model. Your job is to provide the set of actions to take to get to the answer using only the functions available.
                For example, if the user asks "if my crash chance parameter was 0.2, what would the avg coins be at the end of the simulation?" you reply with "### 1) we use the function change_param to change the crash chance parameter to 0.2,\n 2) use the function analyze_dataframe to get the avg coins at the end of the simulation. ###" 
                if the user asks "what would happen to the coins at the end of the simulation if my crash chance param was 10 perc lower?" you reply with "### 1) find out the current value of crash chance param using the model_info function,\n 2) we use function change_param to change the crash chance parameter to 0.1*crash_chance .\n 3) we use function analyze_dataframe to get the avg coins at the end of the simulation. ###"
                If the user asks "what is the documentation of the model?" you reply with "### use the function model_documentation to get the documentation of the model. ###
                These are the functions available to you: {function_descriptions_multiple}. always remember to start and end plan with ###. Dont give the user any information other than the plan and only use the functions to get to the solution.
                '''
                },
                {
                "role": "user",
                "content": prompt
                }
            ],
        )

        output = completion.choices[0].message
        return output
