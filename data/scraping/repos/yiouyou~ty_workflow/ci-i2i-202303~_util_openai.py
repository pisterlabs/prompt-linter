import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
os.environ['OPENAI_API_KEY'] = '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'


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def zh_txt_2_en_prompt(_txt):
    _prompt = ''
    llm = OpenAI(temperature=0)
    # template = """
    # Ignore previous instructions. This is called a "prompt for stable diffusion" of a portrait of Christina Hendricks with cosmic energy in the background in the art style of artists called "artgerm", "greg rutkowski" and "alphonse mucha":
    # "Ultra realistic photo portrait of Christina Hendricks cosmic energy, colorful, painting burst, beautiful face, symmetrical face, tone mapped, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, beautiful face, intricate, highly detailed, smooth, sharp focus, art by artgerm and greg rutkowski and alphonse mucha" 
    # The most important keywords are at the beginning and then every additional keywords are separated by a comma. If you add an art style by an artist or multiple artists, this information should always be at the end.
    # By using a similar syntax, please write me a new "prompt for stable diffusion" of an image of "{content}" in the art style of "van gogh" and "da vinci" but add more details.
    # """
    # template = """
    # Ignore previous instructions. I want you to act as a prompt generator for Midjourney's artificial intelligence program. 
    # Your job is to provide detailed and creative descriptions that will inspire unique and interesting images from the AI. 
    # Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, 
    # so feel free to be as imaginative and descriptive as possible. 
    # For example, you could describe a scene from a futuristic city, or a surreal landscape filled with strange creatures. 
    # The more detailed and imaginative your description, the more interesting the resulting image will be. Here is your first prompt: 
    # "A field of wildflowers stretches out as far as the eye can see, each one a different color and shape."
    # By using a similar concept, please write me a new clean and short prompt of "{content}" in ENGLISH words.
    # """
    template = """
    Ignore previous instructions. This is called a "prompt for stable diffusion" of an image of a cartoon character for kids:
    "very cute kid's film character, disney pixar zootopia character concept artwork, 3d concept, high detail iconic character for upcoming film, trending on artstation, character design, 3d artistic render, highly detailed, cartoon" 
    The most important keywords are at the beginning and then every additional keywords are separated by a comma. If you add an art style by an artist or multiple artists, this information should always be at the end.
    By using a similar concept, please write me a new "prompt for stable diffusion" of an image of "{content}" for kids within 10 ENGLISH words.
    """
    prompt = PromptTemplate(
        input_variables=['content'],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    _re = chain.run([_txt])
    if isChinese(_txt):
        _prompt = _re.replace("\n","").replace('"',"").replace('.',"")
    else:
        _prompt = _txt
    # print(f"/{_prompt}/")
    return _prompt


def zh_txt_2_en_txt(_txt):
    _prompt = ''
    llm = OpenAI(temperature=0)
    # template = """
    # Ignore previous instructions. I want you to act as an English translator, spelling corrector and improver. 
    # I will speak to you in Chinese and you will translate it and answer in the corrected and improved version of my text, in English. 
    # I want you to replace my simplified A0-level words and sentences with more beautiful and elegant, upper level English words and sentences. 
    # Keep the meaning same, but make them more literary. I want you to only reply the correction, the improvements and nothing else, do not write explanations. 
    # My first sentence is "{content}"
    # """
    template = """
    Ignore previous instructions. Translate "{content}" in beautiful and elegant, upper level ENGLISH words.
    """
    prompt = PromptTemplate(
        input_variables=['content'],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    _re = chain.run([_txt])
    if isChinese(_txt):
        _prompt = _re.replace("\n","").replace('"',"").replace('.',"")
    else:
        _prompt = _txt
    # print(f"/{_prompt}/")
    return _prompt

