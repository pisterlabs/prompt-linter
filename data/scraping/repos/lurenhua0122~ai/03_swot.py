from semantic_kernel.skill_definition import (
    sk_function,
    sk_function_context_parameter,
)
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(
        deployment_name=deployment, endpoint=endpoint, api_key=api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service(
        "openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key=api_key, org_id=org_id))
print("A Kernel is now ready")

sk_prompt = """
{{$input}}

Summarize the content above in less than 140 characters.
"""

summary_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                   description="Summarizes the input to length of an old tweet.",
                                                   max_tokens=200,
                                                   temperature=0.1,
                                                   top_p=0.5)
print("A semantic function for summarization has been registered.")

sk_input = """
Let me illustrate an example. Many weekends, I drive a few minutes from my house to a local pizza store to buy 
a slice of Hawaiian pizza from the gentleman that owns this pizza store. And his pizza is great, but he always 
has a lot of cold pizzas sitting around, and every weekend some different flavor of pizza is out of stock. 
But when I watch him operate his store, I get excited, because by selling pizza, he is generating data. 
And this is data that he can take advantage of if he had access to AI.

AI systems are good at spotting patterns when given access to the right data, and perhaps an AI system could spot 
if Mediterranean pizzas sell really well on a Friday night, maybe it could suggest to him to make more of it on a 
Friday afternoon. Now you might say to me, "Hey, Andrew, this is a small pizza store. What's the big deal?" And I 
say, to the gentleman that owns this pizza store, something that could help him improve his revenues by a few 
thousand dollars a year, that will be a huge deal to him.
"""
# Text source: https://www.ted.com/talks/andrew_ng_how_ai_could_empower_any_business/transcript


# summary_result =  kernel.run_async(summary_function, input_str=sk_input)
summary_result = summary_function(sk_input)

display(Markdown('###'+str(summary_result)))


class ExoticLanguagePlugin:
    def word_to_pig_latin(self, word):
        vowels = "AEIOUaeiou"
        if word[0] in vowels:
            return word + "way"
        for idx, letter in enumerate(word):
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
        pig_latin_word = []
        for word in words:
            pig_latin_word.append(self.word_to_pig_latin(word))
        return ' '.join(pig_latin_word)


exotic_language_plugin = kernel.import_skill(
    ExoticLanguagePlugin(), skill_name="exotic_language_plugin")
pig_latin_function = exotic_language_plugin['pig_latin']

print("this is kind of not going to feel awesome but know this is a big deal")

final_result = kernel.run_async(
    summary_function, pig_latin_function, input_str=sk_input)
display(Markdown("###" + str(final_result)))
