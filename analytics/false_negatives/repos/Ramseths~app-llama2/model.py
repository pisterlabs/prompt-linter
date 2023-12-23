from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getResponse(story_topic, genre, main_character):

    llm = CTransformers(model = "./models/llama-2-7b-chat.ggmlv3.q8_0.bin",
                        model_type="llama",
                        config={'max_new_tokens' : 256,
                                'temperature' : 0.01})
    
    print('Template')
    template = """
    Write a story of the genre {genre} and include the topic of: {story_topic} with the main character {main_character}:
    """

    prompt = PromptTemplate(
        input_variables=['story_topic', 'genre', 'main_character'],
        template=template
    )

    response = llm(prompt.format(story_topic = story_topic, genre = genre, main_character = main_character))
    print(response)

    return response