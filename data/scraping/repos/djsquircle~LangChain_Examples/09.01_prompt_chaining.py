from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Load environment variables, like API keys
load_dotenv()

def create_llm_chain(llm, template, input_variables):
    """
    Helper function to create a LLMChain with a given template.
    
    Args:
    - llm: An instance of the language model.
    - template: A string template for the prompt.
    - input_variables: A list of input variable names expected by the template.
    
    Returns:
    - LLMChain instance.
    """
    prompt = PromptTemplate(template=template, input_variables=input_variables)
    return LLMChain(llm=llm, prompt=prompt)

def main(chosen_ingredient):
    # Initialize LLM
    llm = OpenAI(model_name="text-davinci-003", temperature=0.7)
    
    # Define templates
    template_properties = "Describe the magical properties of the ingredient {ingredient}.\nAnswer: "
    template_recipe = "Create a potion recipe using {ingredient} with its magical properties {properties}.\nAnswer: "
    template_effects = "What are the effects when a person consumes a potion made with the recipe {recipe}?\nAnswer: "
    
    # Create LLM Chains
    chain_properties = create_llm_chain(llm, template_properties, ["ingredient"])
    chain_recipe = create_llm_chain(llm, template_recipe, ["ingredient", "properties"])
    chain_effects = create_llm_chain(llm, template_effects, ["recipe"])
    
    # Fetch magical properties
    response_properties = chain_properties.run({"ingredient": chosen_ingredient}).strip()
    
    # Construct a potion recipe
    response_recipe = chain_recipe.run({
        "ingredient": chosen_ingredient,
        "properties": response_properties
    }).strip()
    
    # Describe the effects of the potion
    response_effects = chain_effects.run({"recipe": response_recipe}).strip()
    
    # Output the results
    print(f"Ingredient: {chosen_ingredient}")
    print(f"Magical Properties: {response_properties}")
    print(f"Potion Recipe: {response_recipe}")
    print(f"Effects of the Potion: {response_effects}")

# Run the main function with a default ingredient
if __name__ == "__main__":
    default_ingredient = "Dragon's Scale"
    main(default_ingredient)

#hint: from name_of_this_script_file import main 
# main("Unicorn's Horn")
