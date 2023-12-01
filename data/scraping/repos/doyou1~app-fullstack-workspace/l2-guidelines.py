import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")

# helper function
# Throughout this course, we will use OpenAI's gpt-3.5-turbo model and the chat completions endpoint.
# This helper function will make it easier to use prompts and look at the generated outputs.
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


### Promopting Principles
    # Principle 1: Write clear and specific instructions
    # Principle 2: Give the model time to "think"

## Tactics
# Tactic 1: Use delimiters to clearly indicate distinct parts of the input
    # - Delimiters can be anything like ```, ''', <>, <tag> </tag>, :

# (translation) 
# 가능한 한 명확하고 구체적인 지침을 제공하여 모델이 수행할 작업을 표현해야 합니다. 이 의지 
# 원하는 출력으로 모델을 안내하고 관련이 없거나 잘못된 응답을 받을 가능성을 줄입니다. 명확하게 쓰기를 혼동하지 마십시오 
# 프롬프트에서 짧은 프롬프트를 표시합니다. 대부분의 경우 긴 프롬프트는 모델에서 더 명확하고 맥락을 제공하므로 더 상세하고 관련성 있는 출력으로 이어질 수 있습니다.
text = f"""
You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the desired output, \
and reduce the chances of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context from the model, which can lead to \
more detailed and relevant outputs.
"""

# (translation)
# 삼중 백택으로 구분된 텍스트를 하나의 문장으로 요약합니다. ```{text}```
prompt = f"""
Summarize the text delimited by triple backticks \
into a single sentence. In Korean
```{text}```
"""

# response = get_completion(prompt)
# (translation) 모델을 원하는 출력으로 안내하기 위해 명확하고 구체적인 지침을 제공해야 하며, 긴 프롬프트는 보다 상세하고 관련성이 높은 출력에 대해 보다 명확한 설명과 맥락을 제공할 수 있습니다.
# (In Korean) 모델이 원하는 출력물을 만들기 위해서는 가능한 한 명확하고 구체적인 지시사항을 제공하여 모델이 원하는 방향으로 이끌어야 하며, 길이가 짧은 프롬프트와 명확한 프롬프트를 혼동하지 말아야 하며, 많은 경우 더 긴 프롬프트가 모델로부터 더 많은 명확성과 문맥을 제공하여 더 자세하고 관련성 높은 출력물 을 얻을 수 있다.
# Clear and specific instructions should be provided to guide a model towards the desired output, and longer prompts can provide more clarity and context for more detailed and relevant outputs.
# print(response) 

# Tactic 2: Ask for a structured output
 # JSON, HTML
prompt = f"""
Generate a list of three made-up book titles along \
with their authors and genres.
Provide them in JSON format with the following keys:
book_id, title, author, genre.
"""
# response = get_completion(prompt)
# print(response)
# [
#   {
#     "book_id": 1,
#     "title": "The Lost City of Zorath",
#     "author": "Aria Blackwood",
#     "genre": "Fantasy"
#   },
#   {
#     "book_id": 2,
#     "title": "The Last Survivors",
#     "author": "Ethan Stone",
#     "genre": "Science Fiction"
#   },
#   {
#     "book_id": 3,
#     "title": "The Secret Life of Bees",
#     "author": "Lila Rose",
#     "genre": "Romance"
#   }
# ]

# Tactic 3: Ask the model to check whether conditions are satisfied

# (translation)
# 차 한 잔을 만드는 것은 쉽습니다! 
# 일단 물을 끓여야 하는데 그럴 때 컵을 들고 티백을 넣어주세요. 
# 물이 충분히 뜨거우면 티백 위에 붓기만 하면 됩니다. 
# 차가 끓을 수 있도록 잠시 놔두세요. 
# 몇 분 후에 티백을 꺼냅니다. 
# 만약 여러분이 원한다면, 맛을 내기 위해 설탕이나 우유를 첨가할 수 있습니다. 
# 그게 다야! 당신은 맛있는 차 한 잔을 즐길 수 있습니다.
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \
water boiling, While that's happening, \
grab a cup and put a tea bag in it. Once the water is \
hot enough, just pour it over the tea bag. \
Let it sit for a bit so the tea can steep. After a \
few minutes, take out the tea bag. If you \
like, you can add some sugar or milk to taste. \
And that's it! You've got yourself a delicious \
cup of tea to enjoy.
"""

# (translation)
# 세 개의 따옴표로 구분된 텍스트가 제공됩니다.
# 일련의 지시가 포함되어 있다면, 
# 다음 형식으로 해당 지침을 다시 작성합니다:
# 1단계 - ...
# 2단계 - …
# …
# N단계 - …
# 텍스트에 일련의 지침이 포함되어 있지 않으면, 
# 그런 다음 "단계가 제공되지 않음"이라고 간단히 적습니다
prompt = f"""
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequnce of instructions, \
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""

# response = get_completion(prompt)
# print("Completion for Text 1: ")
# print(response)
# Completion for Text 1: 
# Step 1 - Get some water boiling.
# Step 2 - Grab a cup and put a tea bag in it.
# Step 3 - Once the water is hot enough, pour it over the tea bag.
# Step 4 - Let it sit for a bit so the tea can steep.
# Step 5 - After a few minutes, take out the tea bag.
# Step 6 - Add some sugar or milk to taste (optional).
# Step 7 - Enjoy your delicious cup of tea!

# (translation)
# 오늘은 태양이 밝게 빛나고, 새들이 지저귀고 있습니다.
# 공원으로 산책을 가기에 좋은 날씨입니다. 
# 꽃이 피고, 나무들이 산들바람에 살랑살랑 흔들리고 있습니다. 
# 사람들은 밖에서 돌아다니며 멋진 날씨를 즐기고 있습니다. 
# 어떤 사람들은 소풍을 가고 있는 반면, 다른 사람들은 게임을 하거나 잔디에서 휴식을 취하고 있습니다. 
# 야외에서 시간을 보내고 자연의 아름다움을 감상하기에 완벽한 날입니다.
text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \
walk in the park. The flowers are blooming, and the \
trees are swaying gently in the breeze. People \
are out and about, enjoying the lovely weather. \
Some are having picnics, while others are playing \
games or simply relaxing on the grass. It's a \
perfect day to spend time outdoors and appreciate the \
beauty of nature.
"""

prompt = f"""
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequnce of instructions, \
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""

# response = get_completion(prompt)
# print("Completion for Text 2: ")
# print(response)
# Completion for Text 2: 
# No steps provided.

# Tactic 4: "Few-shot" prompting
    # what is "Few-shot" ?  
    # : 자연어 처리에서 상황 내 학습, 퓨샷 학습 또는 퓨샷 프롬팅은 모델이 작업을 시도하기 전에 예를 처리할 수 있도록 하는 프롬팅 기술입니다. 이 방법은 GPT-3의 출현 이후 대중화되었으며 대규모 언어 모델의 창발적 속성으로 간주됩니다


# (translation)
# 당신의 임무는 일관된 스타일로 대답하는 것입니다.

# <아이>: 인내심에 대해 가르쳐 주세요.
# <조부모>: 가장 깊은 계곡을 깎는 강은 수수한 샘에서 흐릅니다. 가장 위대한 교향곡은 하나의 음표에서 비롯됩니다. 가장 복잡한 태피스트리는 단독 실에서 시작합니다.
# <아이>: 회복력에 대해 가르쳐 주세요.
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \
valley flows from a modest spring; the \
grandest symphony originates from a single note; \
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""

# response = get_completion(prompt)
# print(response)
# <grandparent>: Resilience is like a tree that bends with the wind but never breaks. 
# It is the ability to bounce back from adversity and keep moving forward, even when things get tough. 
# Just like a tree needs strong roots to weather a storm, we need to cultivate inner strength and perseverance to overcome life's challenges.
# (translation) 
# <조부모>: 회복력이란 바람과 함께 휘어지되 결코 꺾이지 않는 나무와 같은 것입니다. 
# 그것은 역경에서 다시 일어설 수 있는 능력이고, 상황이 어려워질 때에도 계속해서 앞으로 나아갈 수 있는 능력입니다. 
# 나무가 폭풍을 이겨내기 위해 튼튼한 뿌리가 필요한 것처럼, 우리는 삶의 도전을 극복하기 위해 내면의 힘과 끈기를 길러야 합니다.


### Principle 2: Give the model time to "think"
### Tactic 1: Specify the steps required to complete a task

# (translation)
# 매력적인 마을에서, 잭과 질 남매는 언덕 꼭대기 우물에서 물을 가져오기 위해 탐험을 떠났습니다. 
# 그들이 올라갈 때, 즐겁게 노래하며, 
# 불행 잭은 돌에 걸려 넘어졌고 질은 그 뒤를 따랐습니다. 약간의 구타를 당했지만, 그 두 사람은 집으로 돌아왔습니다 
# 위로의 포옹. 
# 그 불행에도 불구하고, 그들의 모험심은 흔들리지 않았고, 그들은 기쁨으로 탐험을 계속했습니다.
text  = f"""
In a charming village, sibiling Jack and Jill set out on \
a quest to fetch water from a hilltop \
well. As they climbed, singing joyfully, misfortune \
struck-Jack tripped on a stone and tumbled \
down the hill, with Jill following suit. \
Though slightly battered, the pair returned home to \
comforting embraces. Despite the mishap, \
their adventurous spirits remained undimmed, and they \
continued exploring with delight.
"""

# example 1
prompt_1 = f"""
Perform the following actions:
1 - Summarize the following text delimited by triple \
back ticks with 1 sentence.
2 - Translate the summary into Korean.
3 - List each name in the Korean summary.
4 - Output a json object that contains the following \
keys: korean_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
# response = get_completion(prompt_1)
# print("Completion for prompt 1:")
# print(response)
# 1 - Jack and Jill go on a quest to fetch water from a well, but misfortune strikes when Jack trips and tumbles down the hill, yet they return home slightly battered but with their adventurous spirits undimmed.
# 2 - 잭과 질은 우물에서 물을 가져오기 위해 여행을 떠난다. 그러나 잭이 바위에 걸 
# 려 언덕을 굴러내리고, 질도 따라가게 된다. 그들은 약간 상처를 입었지만 모험적인 
# 정신은 여전히 유지되며 기쁨으로 탐험을 계속한다.
# 3 - 잭, 질
# 4 - {
#       "korean_summary": "잭과 질은 우물에서 물을 가져오기 위해 여행을 떠난다.  
# 그러나 잭이 바위에 걸려 언덕을 굴러내리고, 질도 따라가게 된다. 그들은 약간 상처
# 를 입었지만 모험적인 정신은 여전히 유지되며 기쁨으로 탐험을 계속한다.",        
#       "num_names": 2
#    }

# Ask for output in a specified format
prompt_2 = f"""
Your task is to perform the following actions:
1 - Summarize the following text delimited by <> with 1 sentence.
2 - Translate the summary into Korean.
3 - List each name in the Korean summary.
4 - Output a json object that contains the following keys: korean_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Korean summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
# response = get_completion(prompt_2)
# print("\nCompletion for prompt 2:")
# print(response)
# Summary: Jack and Jill go on a quest to fetch water from a well, but encounter misfortune on the way back home.
# Translation: 잭과 질은 우물에서 물을 가져오기 위해 여행을 떠나지만, 집으로 돌아오는 길에 불운을 만납니다.
# Names: 잭, 질
# Output JSON: {"korean_summary": "잭과 질은 우물에서 물을 가져오기 위해 여행을 떠나지만, 집으로 돌아오는 길에 불운을 만납니다.", "num_names": 2}

### Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion (결론을 내리기 전에 모델이 자체 솔루션을 해결하도록 지시)

# (translation)
# 학생의 솔루션이 올바른지 여부를 결정합니다.

# 질문:
# 저는 태양광 발전 시설을 짓고 있는데 재정 문제를 해결하는 데 도움이 필요합니다. 
# - 땅값은 평방피트당 100달러입니다
# - 저는 평방피트당 250달러에 태양 전지판을 살 수 있습니다
# - 연간 10만 달러의 고정 비용과 평방피트당 10달러의 추가 비용이 드는 유지보수 계약을 협상했습니다
# 평방 피트의 수에 대한 함수로서 운영 첫 해의 총 비용은 얼마입니까.

# 학생 솔루션:
# x를 평방 피트 단위의 설치 크기로 합니다.
# 비용:
# 1. 지가: 100배
# 2. 태양 전지판 가격: 250배
# 3. 유지관리비 : 100,000 + 100배
# 총 비용: 100x + 250x + 100,000 + 100x = 450x + 100,000
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power Installation and I need \
help working out the financials.
- Land costs $100 / square foot
- I can but solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
# response = get_completion(prompt)
# print("Wrong prompt")
# print(response)
# The student's solution is correct.

# Note that the student's solution is actually not correct.
# We can fix this by instructing the model to work out its own solution first.
# (translation)
# 학생의 솔루션은 실제로 올바르지 않습니다.
# 우리는 모델에게 먼저 해결책을 제시하도록 지시함으로써 이 문제를 해결할 수 있습니다.

# (translation)
# 학생의 솔루션이 올바른지 여부를 결정합니다.

# 질문:.
# 태양열 발전 시설을 짓고 있는데 
#  재정 문제를 해결하는 데 도움이 됩니다. 
# - 땅값은 평방피트당 100달러입니다
# - 저는 평방피트당 250달러에 태양 전지판을 살 수 있습니다
# - 비용이 드는 유지보수 계약을 협상했습니다  
# 나는 연간 10만 달러를 균일하게, 그리고 제곱당 10달러를 추가합니다 
# 발
# 운영 첫 해의 총 비용은 얼마입니까 
# 제곱 피트의 수에 대한 함수로서.

# 학생 솔루션:
# x를 평방 피트 단위의 설치 크기로 합니다.
# 비용:
# 1. 지가: 100배
# 2. 태양 전지판 비용: 250배 당신의 임무는 학생의 해결책을 결정하는 것입니다 
# 맞거나 그렇지 않습니다.
# 문제를 해결하려면 다음을 수행합니다:
# - 먼저, 그 문제에 대한 당신만의 해결책을 생각해 보세요.
# - 그런 다음 자신의 솔루션을 학생의 솔루션과 비교합니다 
# 학생의 정답이 맞는지 평가합니다.
# 여러분이 직접 문제를 해결하기 전에는 학생의 해결책이 맞는지 결정하지 마세요.

# 3. 유지관리비 : 100,000 + 100배
# 총 비용: 100x + 250x + 100,000 + 100x = 450x + 100,000
prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem.
- Then compare your solution to the student's solution \
and evaludate if the student's solution is correct or not.
Don't decide if the student's solution is correct until you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help \
working out the financials.
- Land costs $100 / sqaure foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \
me a flat $100k per year, and an additional $10 / square foot
What is the total cost for the first year of operations as a function of the number of square feet.
```
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,00 + 100x = 450x + 100,000
```
Actual solution:
"""
# response = get_completion(prompt)
# print("Fixed prompt")
# print(response)
# Let x be the size of the installation in square feet.
# Costs:
# 1. Land cost: 100x
# 2. Solar panel cost: 250x
# 3. Maintenance cost: 100,000 + 10x
# Total cost: 100x + 250x + 100,000 + 10x = 360x + 100,000

# Is the student's solution the same as actual solution just calculated:
# No

# Student grade:
# Incorrect

### Model Limitations: Hallucinations
# Boie is a real company, the product name is not real.
prompt = f"""
In korean, Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie.
"""
# response = get_completion(prompt)
# print("Model Limitations: Hallucinations")
# print(response)
# 보이의 AeroGlide UltraSlim Smart Toothbrush는 혁신적인 디자인과 기술로 만들어진 칫솔입니다. 이 제품은 초슬림한 디자인으로 입안에서 편안하게 사용할 수 있으며, 스마트 기술을 통해 칫솔 사
# 용 시간, 압력, 브러싱 패턴 등을 모니터링하여 사용자의 구강 건강을 도와줍니다. 또한, 칫솔 머리는 부드러운 실리콘 소재로 만들어져 치아와 잇몸을 부드럽게 마사지하며 치아의 표면을 깨끗하게
#  닦아줍니다. 이 제품은 USB 충전이 가능하며, 한 번 충전으로 최대 30일간 사용할 수 있습니다.

prompt = f"""
In korean, Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie.

To answer, do the following.
- First, look for quotes related to the text.
- Use the quote to answer the question.
- If you cannot give an appropriate answer, make it clear that you cannot.
"""
# response = get_completion(prompt)
# print("Fix prompt")
# print(response)
# 죄송합니다, 인공지능 언어 모델로는 인터넷을 검색할 수 없고 텍스트와 관련된 인용문을 찾을 수 없습니다.
# 하지만 보이의 에어로글라이드 울트라슬림 스마트 칫솔은 초음파 기술을 이용해 치아를 청소하는 칫솔로, 조작성이 용이하도록 슬림한 디자인을 갖췄다고 말씀드릴 수 있습니다. 
# 또한 적절한 칫솔질 시간을 보장하는 스마트 타이머가 있으며 충전 기반이 제공됩니다.

prompt = f"""
In korean, Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie.

To answer, do the following.
- First, look for quotes related to the text.
- Use the quote to answer the question in korean.
- If you can't give an appropriate answer, make it clear that you can't give an answer, and don't give an additional answer that may be an error.
"""

# response = get_completion(prompt)
# print("Fix prompt 2")
# print(response)
# "I love the AeroGlide UltraSlim Smart Toothbrush by Boie. It's so sleek and easy to use." 
# "AeroGlide UltraSlim Smart Toothbrush by Boie는 너무 멋지고 사용하기 쉬워서 좋아해요."

prompt = f"""
In korean, Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie.

To answer, do the following.
- First, look for quotes related to the text.
- Use the quote to answer the question in korean.
- If you can't give an appropriate answer, make it clear that you can't give an answer, and then simply write \"No answer provided.\"
"""

response = get_completion(prompt)
print("Fix prompt 3")
print(response)
# "No answer provided."