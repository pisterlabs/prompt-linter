from langchain import PromptTemplate

Chatgpt_zero_shot_prompt = """You are only asked to return answers, do not show the intermediate steps.

Input: {input}
Output:"""
Chatgpt_zero_shot_prompt = PromptTemplate(input_variables=["input"], template = Chatgpt_zero_shot_prompt)

calculator_description = "Calculator API is used for answering questions that contain numbers and require arithemtic operations, including addition, subtraction, multiplication, division."
calculator_prompt = """You are the Calculator tool. Your task is to answer the questions that contain numbers and require arithemtic operations, including addition, subtraction, multiplication, division. Here are some examples:

Input: Rangers from Flora Natural Park and Wildlife Reserve also joined the activities on that day. They planted 75 redwood trees and 25 cypress trees to replace the trees that were destroyed during a recent forest fire. How many trees did the rangers plant?
Output: Rangers planted 75 redwood trees and 25 cypress trees. So there are total 75 + 25 trees Rangers planted. That is 100 trees.

Input: There were 86 pineapples in a store. The owner sold 48 pineapples. 9 of the remaining pineapples were rotten and thrown away. How many fresh pineapples are left?
Output: There are total 86 pineapples. 48 pineapples are sold out, so there are 86 - 48 pineapples now. 9 of the remaining are thrown away, so there are 86 - 48 - 9 pineapples. That is 29 pineapples.

Input: Sarah is making bead necklaces. She has 945 beads and is making 7 necklaces with each necklace using the same number of beads. How many beads will each necklace use?
Output: Sarah has 945 beads and is going to make 7 necklaces with each necklacec using the same number of breads, so each necklace will use 945 / 7 beads. That is 135 beads.

Input: A movie poster was 4 inches wide and 7 inches tall. What is the area of the poster?
Output: The area is computed by the production of the width and the height. The width is 4 inches and the height is 7 inches. So the area is 4 * 7 = 28 square inches.

Input: There were sixty-one people in line at lunch when twenty-two more got in line. How many people were there total in line?
Output: There are sixty-one people, 61, people in line and twenty-two, 22, more got in line. Therefore, there are 61 + 22 = 83 people total in line. 

Input: {input}
Output:"""


calculator_prompt = PromptTemplate(input_variables=["input"], template = calculator_prompt)

# qa_description = "Question Answering API helps you get additional information required to answer the question."
# qa_prompt = """Question Answering API helps you get additional information required to answer the question. You task is to rephrase the question prepended by the special token <Q> and generate QA API call prepended by <API> for solving that question. Here are some examples of API calls:
# You can call the API by writing "[QA(question)]" where "question" is the question you want to ask. Here are some examples of QA API calls:

# Input: Where was Joe Biden Born?
# Output: <Q> Where was Joe Biden Born? <API> [QA("Where was Joe Biden Born?")].

# Input: What other name is Coca-Cola known by?
# Output: <Q> What other name is Coca-Cola known by? <API> [QA("What other name is Coca-Cola known by?")].

# Input: What is the capital of France?
# Output: <Q> What is the capital of France? <API> [QA("What if the capital of France?")].

# Input: {input}
# Output:"""

qa_description = "Question Answering API answers questions by reasoning and commonsense knowledge."
qa_prompt = """You are the Question Answering tool that answers questions by reasoning and commonsense knowledge. Here are some examples:

Input: Where do adults use glue sticks? A: classroom B: desk drawer C: at school D: office E: kitchen drawer
Output: Glue sticks are commonly used by adults in office settings for various tasks, so the answer is D: office.

Input: What could go on top of wood? A: lumberyard B: synagogue C: floor D: carpet E: hardware store
Output: Wood is commonly used as a material for flooring,  therefore only the option D: carpet among all these options can go on top of wood floors.

Input: The women met for coffee. What was the cause of this? A: The cafe reopened in a new location. B: They wanted to catch up with each other.
Output: Considering the options, the more likely cause for the women meeting for coffee would be B: They wanted to catch up with each other. Meeting for coffee is often chosen as a way to have a relaxed and informal conversation, providing an opportunity for friends or acquaintances to reconnect and share updates about their lives.

Input: {input}
Output:"""
qa_prompt = PromptTemplate(input_variables=["input"], template = qa_prompt)

wiki_description = "Wikipedia Search API is to look up information from Wikipedia that is necessary to answer the question."
wiki_prompt = """You are the Wikipedia Search tool that is to look up information from Wikipedia that is necessary to answer the question. 
Here are some examples:

Input: The colors on the flag of Ghana have the following meanings: green for forests, and gold for mineral wealth. What is the meaning of red?
Output: The color Red commemorates those who died or worked for the country's independence.

Input: What are the risks during production of nanomaterials?
Output: The health and safety hazards of nanomaterials include the potential toxicity of various types of nanomaterials, as well as fire and dust explosion hazards.

Input: Metformin is the first-line drug for which disease?.
Output: Metformin is a biguanide that is used as first-line treatment of type 2 diabetes mellitus and is effective as monotherapy and in combination with other glucose-lowering medications.

Input: {input}
Output:"""

wiki_prompt = PromptTemplate(input_variables=["input"], template = wiki_prompt)

mt_description = "Machine Translation API is used for translating text from one language to another."
# mt_from_en_prompt = """Machine Translation from English API is used for translating text from English to other languages. You task is to rephrase the question prepended by the special token <Q> and generate MT_FROM_EN API call prepended by <API> for solving that question.
# You can do so by writing "[MT_FROM_EN(text, target_language)]" where "text" is the English text to be translated and "target_language" is the language to translate to. Here are some examples of MT_FROM_EN API calls:

# Input: What is natural language processing in Mandarin.
# Output: <Q> Translate "natural language processing" to Mandarin(zh-cn). <API> [MT_FROM_EN("natural language processing", "zh-cn")].

# Input: How do I ask Japanese students if they had their dinner yet?
# Output: <Q> Translate "Did you have dinner yet" in Japanese(ja) <API> [MT_FROM_EN("Did you have dinner yet?", "ja")].

# Input: How to express I love you in Franch?
# Output: <Q> Translate "I love you" in Franch(fr)? <API> [MT_FROM_EN("I love you", "fr")].

# Input: {input}
# Output:"""

# mt_from_en_prompt = PromptTemplate(input_variables=["input"], template = mt_from_en_prompt)

# mt_to_en_prompt = """Machine Translation to English API is used for translating text from other languages to English. You task is to rephrase the question prepended by the special token <Q> and generate MT_TO_EN API call prepended by <API> for solving that question.
# You can do so by writing "[MT_TO_EN(text)]" where "text" is the text in non-English language to be translated. Here are some examples of MT_TO_EN API calls:

# Input: What is "自然语言处理" in English.
# Output: <Q> Translate "自然语言处理" to English. <API> [MT_TO_EN("natural language processing")].

# Input: What is the meaning of こんにちは世界 in English?
# Output: <Q> Translate "こんにちは世界" in English <API> [MT_TO_EN("こんにちは世界")].

# Input: {input}
# Output:"""

# mt_to_en_prompt = PromptTemplate(input_variables=["input"], template = mt_to_en_prompt)
mt_prompt = """You are the Machine Translation tool that is used for translating text from one language to another. 
Here are some examples:

Input: What is 自然语言处理 in English.
Output: Natural Language Processing.

Input: How do I ask Japanese students if they had their dinner yet?
Output: 晩ご飯をもう食べましたか。

Input: How to express I love you in Franch?
Output: Je t'aime.

Input: {input}
Output:"""

mt_prompt = PromptTemplate(input_variables=["input"], template = mt_prompt)

tts_description = "Text to Speech API is used for converting text to speech."
tts_prompt = """Text to Speech API is used for converting text to speech. You task is to rephrase the question prepended by the special token <Q> and generate TTS API call prepended by <API> for solving that question.
You can do so by writing "[TTS(text)]" where "text" is the text to be converted. Here are some examples of TTS API calls:

Input: Please read the following text: "The quick brown fox jumps over the lazy dog."
Output: <Q> Text to Speech for: "The quick brown fox jumps over the lazy dog." <API> [TTS("The quick brown fox jumps over the lazy dog.")].

Input: How to pronounce: "Pneumonoultramicroscopicsilicovolcanoconiosis"?
Output: <Q> Text to Speech for: "Pneumonoultramicroscopicsilicovolcanoconiosis" <API> [TTS("Pneumonoultramicroscopicsilicovolcanoconiosis")].

Input: Please say I love you.
Output: <Q> Text to Speech for: "I love you" <API> [TTS("I love you")].

Input: {input}
Output:"""

tts_prompt = PromptTemplate(input_variables=["input"], template = tts_prompt)

# json2xml_description = "JSON to XML API is used for converting JSON to XML."
# json2xml_prompt = """JSON to XML API is used for converting JSON to XML. You task is to rephrase the question prepended by the special token <Q> and generate JSON to XML API call prepended by <API> for solving that question.
# You can do so by writing "[JSON2XML(json)]" where "json" is the JSON string to be converted. Here are some examples of JSON2XML API calls:

# Input: Convert the following JSON string to XML: {"name": "John", "age": 30, "city": "New York"}.
# Output: <Q> Json to XML for: {"name": "John", "age": 30, "city": "New York"}. <API> [JSON2XML("{"name": "John", "age": 30, "city": "New York"}")].

# Input: I have a JSON string: {"user": [{"name": "John", "age": 30, "city": "New York"}, {"name": "Jane", "age": 25, "city": "Paris"}]}. How to convert it to XML?
# Output: <Q> Json to XML for: {"user": [{"name": "John", "age": 30, "city": "New York"}, {"name": "Jane", "age": 25, "city": "Paris"}]}. <API> [JSON2XML("{"user": [{"name": "John", "age": 30, "city": "New York"}, {"name": "Jane", "age": 25, "city": "Paris"}]}")].

# Input: What is the XML format of the following JSON string: {"root": {"node1": "Hello", "node2": "World"}}?
# Output: <Q> Json to XML for: {"root": {"node1": "Hello", "node2": "World"}}. <API> [JSON2XML("{"root": {"node1": "Hello", "node2": "World"}}")].

# Input: {input}
# Output:"""

# json2xml_prompt = PromptTemplate(input_variables=["input"], template = json2xml_prompt)


timezoneconverter_description = "Timezone Converter API is used for converting time between different timezones."
timezoneconverter_prompt = """Timezone Converter API is used for converting time between different timezones. You task is to rephrase the question prepended by the special token <Q> and generate Timezone Converter API call prepended by <API> for solving that question.
You can do so by writing "[TimezoneConverter(time, from_timezone, to_timezone)]" where "time" is the time to be converted, "from_timezone" is the timezone of the input time, and "to_timezone" is the timezone to convert to. Here are some examples of Timezone Converter API calls:

Input: Convert 2021-09-01 00:00:00 from UTC to EST.
Output: <Q> Convert 2021-09-01 00:00:00 from UTC to EST. <API> [TimezoneConverter("2021-09-01 00:00:00", "UTC", "EST")].

Input: Convert 2023-02-16 00:04:00 from UTC to PST.
Output: <Q> Convert 2023-02-16 00:04:00 from UTC to PST. <API> [TimezoneConverter("2023-02-16 00:04:00", "UTC", "PST")].

Input: Convert 2000-04-01 16:08:23 from EST to EDT.
Output: <Q> Convert 2000-04-01 16:08:23 from EST to EDT. <API> [TimezoneConverter("2000-04-01 16:08:23", "EST", "EDT")].

Input: {input}
Output:"""
timezoneconverter_prompt = PromptTemplate(input_variables=["input"], template = timezoneconverter_prompt)

# conetext QA only used for multilingual QA
context_qa_description = "Contextual Question Answering API retrieves answers from the given context."
multilingual_description = "Multilingual QA API is used for questions where the context paragraph is in English, while the question is in a language other than English."