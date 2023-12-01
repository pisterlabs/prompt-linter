# Importing the necessary Python libraries
import os
import yaml
import openai
import gradio as gr
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.utilities import WikipediaAPIWrapper



## OPENAI API CONNECTION
## ---------------------------------------------------------------------------------------------------------------------
# Loading the API key and organization ID from file (NOT pushed to GitHub)
with open('../keys/openai-keys.yaml') as f:
    keys_yaml = yaml.safe_load(f)

# Setting the OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = keys_yaml['API_KEY']

# Using LangChain to connect to OpenAI
openai_llm = OpenAI()



## LANGCHAIN CONFIGURATION
## ---------------------------------------------------------------------------------------------------------------------
# Creating a prompt template for checking to see if the inputted individual is a historical figure
is_historical_figure_template = PromptTemplate(
    input_variables = ['entity_name'],
    template = 'Is the following entity a person, and if yes, would you consider this person to be a historical figure: {entity_name}. Please give me back just a yes or no answer.'
)

# Creating a prompt template for generating a research paper outline the three most important events in the historical figure's life
research_paper_outline_template = PromptTemplate(
    input_variables = ['entity_name', 'wikipedia_entry'],
    template = 'The following is a Wikipedia entry about {entity_name}. Please provide for me an outline of a basic research paper with an introduction, the three most important events of this person\'s life, and a conclusion. Here is the Wikipedia information:\n\n {wikipedia_entry}'
)

# Creating a prompt template for generating a research paper based on the outline generated from the previous prompt template
research_paper_template = PromptTemplate(
    input_variables = ['entity_name', 'research_paper_outline'],
    template = 'You are a high schooler who has just been assigned research paper about a historical figure. The historical figure is {entity_name}, and the following is an outline to follow:\n {research_paper_outline}. Please write a research paper approximately three pages in length.'
)

# Creating a prompt template for re-writing the research paper as Jar Jar Binks
jar_jar_template = PromptTemplate(
    input_variables = ['research_paper'],
    template = 'Please rewrite the following research paper in the tone of Jar Jar Binks from Star Wars:\n\n {research_paper}'
)

# Instantiating the LangChains objects for all the prompt templates defined above
is_historical_figure_chain = LLMChain(llm = openai_llm, prompt = is_historical_figure_template, verbose = True, output_key = 'is_historical_figure')
research_paper_outline_chain = LLMChain(llm = openai_llm, prompt = research_paper_outline_template, verbose = True, output_key = 'research_paper_outline')
research_paper_chain = LLMChain(llm = openai_llm, prompt = research_paper_template, verbose = True, output_key = 'research_paper')
jar_jar_chain = LLMChain(llm = openai_llm, prompt = jar_jar_template, verbose = True, output_key = 'jar_jar_paper')


# Instantiating an object to obtain results from the Wikipedia API
wikipedia_api = WikipediaAPIWrapper(top_k_results = 1)

# Creating the LangChain chain
research_paper_langchain = SequentialChain(chains = [research_paper_outline_chain, research_paper_chain, jar_jar_chain],
                                           input_variables = ['entity_name', 'wikipedia_entry'],
                                           output_variables = ['research_paper_outline', 'research_paper', 'jar_jar_paper'],
                                           verbose = True)



## HELPER FUNCTIONS
## ---------------------------------------------------------------------------------------------------------------------
def generate_book_report(entity_name_prompt):
    '''
    Generates a book report based on a provided entity name

    Inputs:
        - entity_name_prompt (str): The name of the Fortune 500 company submitted by the user

    Returns:
        - entity_name_prompt (str): The cleared out entity name ready to take in the next submission
        - research_paper_outline (str): The outline of the research paper
        - research_paper (str): The research paper itself
        - jar_jar_paper (str): The research paper rewritten in the tone of Jar Jar Binks
    '''

    # Setting the name of the entity being profiled
    entity_name = entity_name_prompt

    # Retrieving information about the company from Wikipedia
    wikipedia_entry = wikipedia_api.run(f'{entity_name} (person)')

    # Passing the wikipedia entry and historical figure name into the LangChain
    langchain_response = research_paper_langchain({'entity_name': entity_name, 'wikipedia_entry': wikipedia_entry})

    # Obtaining each of the elements from the LangChain response
    research_paper_outline = langchain_response['research_paper_outline']
    research_paper = langchain_response['research_paper']
    jar_jar_paper = langchain_response['jar_jar_paper']


    # Clearing out the company name for the next run
    entity_name_prompt = ''

    return entity_name_prompt, research_paper_outline, research_paper, jar_jar_paper


def clear_results():
    '''
    Clears the results from the page

    Inputs:
        - research_paper_outline (str): The outline of the research paper
        - research_paper (str): The research paper itself
        - jar_jar_paper (str): The research paper rewritten in the tone of Jar Jar Binks

    Returns:
        - research_paper_outline (str): Cleared out field ready for next entry
        - research_paper (str): Cleared out field ready for next entry
        - jar_jar_paper (str): Cleared out field ready for next entry
    '''

    # Clearing out the results for each field
    research_paper_outline = ''
    research_paper = ''
    jar_jar_paper = ''

    return research_paper_outline, research_paper, jar_jar_paper



## GRADIO UI LAYOUT & FUNCTIONALITY
## ---------------------------------------------------------------------------------------------------------------------
# Defining the building blocks that represent the form and function of the Gradio UI
with gr.Blocks(title = 'Book Report Generator', theme = 'base') as book_report_generator:

    title_markdown = gr.Markdown('# Book Report Generator')
    entity_name_prompt = gr.Textbox(label = 'Historical person to profile:', placeholder = 'Please enter the name of the historical entity.')
    research_paper_outline = gr.Textbox(label = 'Book Report Outline', interactive = False)
    research_paper = gr.Textbox(label = 'Book Report', interactive = False)
    jar_jar_paper = gr.Textbox(label = 'Book Report (as Jar Jar Binks)', interactive = False)

    # Creating a button to clear the results
    clear_results_button = gr.Button('Clear Results')

    # Defining the behavior for what occurs when the user hits "Enter" after typing a prompt
    entity_name_prompt.submit(fn = generate_book_report,
                              inputs = [entity_name_prompt],
                              outputs = [entity_name_prompt, research_paper_outline, research_paper, jar_jar_paper])
    
    # Clearing out all results when the appropriate button is clicked
    clear_results_button.click(fn = clear_results, inputs = None, outputs = [research_paper_outline, research_paper, jar_jar_paper])
    



## SCRIPT INVOCATION
## ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Launching the Gradio interface
    book_report_generator.launch()