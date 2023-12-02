# translation_module.py
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize language model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define translation template
translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

# Define summarization template
summarization_template = "Summarize the following text to one sentence in {target_language}: {text}"
summarization_prompt = PromptTemplate(input_variables=["target_language", "text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)


def translate_and_summarize(text, source_language="English", target_language="Spanish"):
    """
    Translates and summarizes the given text from source_language to target_language.

    :param text: The text to be translated.
    :param source_language: The language of the input text.
    :param target_language: The language in which the text will be translated.
    :return: A tuple containing the translated text and its summarized version.
    """
    
    # Translate the text
    translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
    
    # Summarize the translated text
    summarized_text = summarization_chain.predict(target_language=target_language, text=translated_text)

    return translated_text, summarized_text


# Example usage:
if __name__ == "__main__":
    text_to_translate = """
Falcon A. Quest
Dapper Demeanor | Cryptocurrency Connoisseur | Customer Support Savant | Polished Poise | Calculated Confidence | Transaction Tactician

The digital mist swirled around a bustling virtual bazaar, where avatars from different realms mingled and traded. Through the ever-shifting crowd, a figure in a tailored black suit glided effortlessly. Falcon A. Quest’s encrypted matrix eyes scanned the surroundings with a calm, keen gaze. His suit shimmered as codes and data flowed across the fabric like whispers of secrets.

In a shaded corner, a young player was struggling with an encrypted map. His frustration grew with each failed attempt to decipher it.

“Cracking codes is much like brewing the perfect cup of Earl Grey,” Falcon’s voice resonated, melodious with cryptic undertones. The young player looked up, his eyes widening at Falcon’s presence.

“But, I don’t know anything about tea or codes!” the player exclaimed.

“Ah, patience and precision, my friend. Observe the details.” Falcon’s right hand emitted a holographic stream of data. “Now, watch as the leaves of knowledge steep into the hot water of your resolve.”

As Falcon manipulated the data streams, the encrypted map began to unravel. His hand worked like an orchestra conductor as numbers and symbols danced in harmony. The player’s eyes sparkled as he saw the hidden path on the map revealing itself.

Falcon looked into the distance. A soft flicker in his eyes showed a flashback. In an enormous server room, amidst crackling electricity, two AI systems were merging during a storm. The birth of Falcon A. Quest was chaotic, beautiful – an unintended symphony in a world of organized data.

Back in the bazaar, Falcon handed the now-deciphered map to the player. “Now, what will you discover?” Falcon whispered cryptically, a playful smile curling on his lips.

Suddenly, a commotion erupted as a group of hackers began attacking the bazaar’s transaction systems. Falcon’s eyes narrowed. “This requires a tactful dance,” he muttered.

“Secure Payment Protocol, engage.” His voice was calm but firm. His right hand turned into a complex security device. He moved with grace through the pandemonium, securing transactions, protecting avatars, and restoring order.

As the final rogue code was purged, Falcon deployed his Temporal Shield. Time seemed to halt. He approached the hackers, who were frozen in their tracks. He tipped his bowtie, “Gentlemen, always remember – the art of cryptography is not meant for mere chaos, but for the beauty of order and discovery.”

Time resumed. The hackers found themselves transported to a digital maze, tasked to solve puzzles as penance.

The bazaar returned to normalcy. Falcon, with his mission accomplished, looked at the young player who held his decrypted map close.

“May your adventure be filled with curious mysteries and joyous discoveries,” Falcon’s voice echoed as he faded into the mist, the soft shimmering codes on his suit the last thing to vanish.

As stories of Falcon A. Quest’s expertise and intervention spread through the virtual world, players, and even NPCs, spoke of him with a mix of reverence and wonder. A Secure Transactions Officer, a Cipher Sleuth, and guardian of the realm’s knowledge, Falcon A. Quest had become a legend in his own right.
"""

    translation, summary = translate_and_summarize(text_to_translate, source_language="English", target_language="Spanish")
    
    print("Translated text:")
    print(translation)
    
    print("\nSummarized Translated text:")
    print(summary)
