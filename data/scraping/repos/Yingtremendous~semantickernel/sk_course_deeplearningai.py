"""
Code from lesson 1-3 tested
"""

# Configure your local machine to run Semantic Kernel
from operator import le

from click import prompt
import semantic_kernel as sk
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
"""
    kernel is for managing resoruces that necessary to run code in 
    ai application
    resoruce: confguration, services and pugins 
"""
# import openai
# import os
# openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureioenai", AzureChatCompletion)
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
print("you made a kernel!")


"""
for hugging face need torch and transformers
"""
# get semantic kernel from hugging face
# kernel.add_text_completion_service("huggingface", HuggingFaceTextCompletion("gpt2"),task="text-generation")
# print("you made a kernel for hugging faces!")

# make a semantic function which is like encapsulations of repeatable LLM prompts
sk_prompt = """
{{$input}}

Summarize the content above in less than 140 characters
"""
summary_function = kernel.create_semantic_function(
    prompt_template=sk_prompt,
    max_tokens=200,
    temperature=0.1,
    top_p=0.5,
)
# use this function to summarize a text

sk_input = """"
New York. Die wegen der Opioid-Krise unter Druck geratene 
Apothekenkette Rite Aid hat in den USA Gläubigerschutz beantragt.
Rite Aid reichte am Sonntag beim Insolvenzgericht für den Bezirk
New Jersey einen Antrag auf Gläubigerschutz nach Chapter 11 ein. 
Damit würde das Unternehmen vor seinen Gläubigern geschützt, um sich zu sanieren. In dem Antrag gab das Unternehmen geschätzte Vermögenswerte und Verbindlichkeiten zwischen einer und zehn Milliarden Dollar an.
"""
# summary_result = await kernel.run_async(summary_function, input_str=sk_input)
# print(summary_result)

# write a native function and use kernel
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter

class ExoticLanguagePlugin:
    def word_to_pig_latin(self, word: str) -> str:
        vowels = "AEIOUaeiou"
        if word[0] in vowels:
            return word + "way"
        for idx,letter in enumerate(word):
            if letter in vowels:
                    break
        else:
            return word + "ay"
        return word[idx:] + word[:idx] + "ay"
    @sk_function(
        description="Takes text and converts it to pig latin",
        name="pig_latin",
        input_description="The text to convert to pig latin",
    )
    def pig_latin(self, sentence: str) -> str:
        words = sentence.split()
        pig_latin_words = []
        for word in words:
            pig_latin_words.append(self.word_to_pig_latin(word))
        return " ".join(pig_latin_words)
async def skNativeFunction():
    kernel = sk.Kernel()

    useAzureOpenAI = False

    if useAzureOpenAI:
        deployment, api, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service("azureioenai", AzureChatCompletion)
    else:
        api_key, org_id = sk.openai_settings_from_dot_env()
        kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
    print("you made a kernel!")
    exotic_language_plugin = kernel.import_skill(ExoticLanguagePlugin(), skill_name="exotic_language_plugin") 
    pig_latin_function = exotic_language_plugin("pig_latin")

    final_result = await kernel.run_async(summary_function, pig_latin_function, input_str=sk_input) 
    print(final_result)


# change text for different situation
swot_interview= """
1. **Strengths**
    - What unique recipes or ingredients does the pizza shop use?
    - What are the skills and experience of the staff?
    - Does the pizza shop have a strong reputation in the local area?
    - Are there any unique features of the shop or its location that attract customers?
2. **Weaknesses**
    - What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)
    - Are there financial constraints that limit growth or improvements?
    - Are there any gaps in the product offering?
    - Are there customer complaints or negative reviews that need to be addressed?
3. **Opportunities**
    - Is there potential for new products or services (e.g., catering, delivery)?
    - Are there under-served customer segments or market areas?
    - Can new technologies or systems enhance the business operations?
    - Are there partnerships or local events that can be leveraged for marketing?
4. **Threats**
    - Who are the major competitors and what are they offering?
    - Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?
    - Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?
    - Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"""

sk_prompt = """
{{$input}}

convert the analysis provided above to the business domain of {{$domain}}
"""
async def shiftDomainswot():
    kernel = sk.Kernel()

    useAzureOpenAI = False

    if useAzureOpenAI:
        deployment, api, endpoint = sk.azure_openai_settings_from_dot_env()
        kernel.add_text_completion_service("azureioenai", AzureChatCompletion)
    else:
        api_key, org_id = sk.openai_settings_from_dot_env()
        kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
    shift_domain_function = kernel.create_semantic_function(
        prompt_template=sk_prompt,
        description="translate the idea to another domain",
        max_tokens=1000,
        temperature=0.1,
        top_p=0.5)
    my_context = kernel.create_new_context()
    my_context["input"] = swot_interview
    my_context["domain"] = "construction management"

    """
    # domain shift ---> text for child 
    # add level into context
    my_context_child['input'] = swot_interview
    my_context_child['domain'] = "construction management"
    my_context_child["level"] = "child"
    results_child = await kernel.run_async(shift_domain_function, input_context=my_context)


    """
    results = await kernel.run_async(shift_domain_function, input_context=my_context)
    print(results)


import asyncio

asyncio.run(shiftDomainswot())


