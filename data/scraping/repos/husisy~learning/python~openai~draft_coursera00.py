import os
import dotenv
import openai

dotenv.load_dotenv()
openai.api_key  = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
    # temperature: the degree of randomness of the model's output
    ret = response.choices[0].message["content"]
    return ret

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    # print(str(response.choices[0].message))
    return response.choices[0].message["content"]

## principle 1: write clear and specific instructions
## tactic: use delimiter to clearly indicate distinct parts of the input
text = f"""
You should express what you want a model to do by
providing instructions that are as clear and
specific as you can possibly make them.
This will guide the model towards the desired output,
and reduce the chances of receiving irrelevant
or incorrect responses. Don't confuse writing a
clear prompt with writing a short prompt.
In many cases, longer prompts provide more clarity
and context for the model, which can lead to
more detailed and relevant outputs.
"""
prompt = f"""Summarize the text delimited by triple backticks into a single sentence.
```{text}```"""
response = get_completion(prompt)
print(response)
'''Clear and specific instructions should be provided to guide a model towards the desired output, and longer prompts can provide more clarity and context for the model, leading to more detailed and relevant outputs.
'''


## tactic: ask for a structured output
prompt = f"""
Generate a list of three made-up book titles along with their authors and genres.
Provide them in JSON format with the following keys:
book_id, title, author, genre.
"""
response = get_completion(prompt)
print(response)
'''
[
  {
    "book_id": 1,
    "title": "The Lost City of Zorath",
    "author": "Aria Blackwood",
    "genre": "Fantasy"
  },
  {
    "book_id": 2,
    "title": "The Last Survivors",
    "author": "Ethan Stone",
    "genre": "Science Fiction"
  },
  {
    "book_id": 3,
    "title": "The Secret of the Haunted Mansion",
    "author": "Lila Rose",
    "genre": "Mystery"
  }
]
'''


## tactic: ask the model to check whether conditions are satisfied
text_1 = f"""
Making a cup of tea is easy! First, you need to get some
water boiling. While that's happening,
grab a cup and put a tea bag in it. Once the water is
hot enough, just pour it over the tea bag.
Let it sit for a bit so the tea can steep. After a
few minutes, take out the tea bag. If you
like, you can add some sugar or milk to taste.
And that's it! You've got yourself a delicious
cup of tea to enjoy.
"""
text_2 = f"""
The sun is shining brightly today, and the birds are
singing. It's a beautiful day to go for a
walk in the park. The flowers are blooming, and the
trees are swaying gently in the breeze. People
are out and about, enjoying the lovely weather.
Some are having picnics, while others are playing
games or simply relaxing on the grass. It's a
perfect day to spend time outdoors and appreciate the
beauty of nature.
"""
prompt = '''
You will be provided with text delimited by triple quotes.
If it contains a sequence of instructions,
re-write those instructions in the following format:

Step 1 - ...
Step 2 - ...
...
Step N - ...

If the text does not contain a sequence of instructions,
then simply write "No steps provided."

"""{text}"""
'''
response = get_completion(prompt.format(text=text_1))
print(response)
'''
Step 1 - Get some water boiling.
Step 2 - Grab a cup and put a tea bag in it.
Step 3 - Once the water is hot enough, pour it over the tea bag.
Step 4 - Let it sit for a bit so the tea can steep.
Step 5 - After a few minutes, take out the tea bag.
Step 6 - Add some sugar or milk to taste.
Step 7 - Enjoy your delicious cup of tea!
'''
response = get_completion(prompt.format(text=text_2))
print(response)
'No steps provided.'


## few-shot prompting
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest
valley flows from a modest spring; the
grandest symphony originates from a single note;
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
'''<grandparent>: Resilience is like a tree that bends with the wind but never breaks.
It is the ability to bounce back from adversity and keep moving forward.
Just like a tree that endures harsh weather and still stands tall, resilience allows us to overcome challenges and grow stronger.'''


## principle 2: give the model time to think
## tactic 1: specify the steps to complete a task
text = f"""
In a charming village, siblings Jack and Jill set out on
a quest to fetch water from a hilltop
well. As they climbed, singing joyfully, misfortune
struck—Jack tripped on a stone and tumbled
down the hill, with Jill following suit.
Though slightly battered, the pair returned home to
comforting embraces. Despite the mishap,
their adventurous spirits remained undimmed, and they
continued exploring with delight.
"""
# example 1
prompt_1 = f"""
Perform the following actions:
1 - Summarize the following text delimited by triple backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print(response)
'''
1 - Siblings Jack and Jill go on a quest to fetch water from a hilltop well, but misfortune strikes as Jack trips and tumbles down the hill, with Jill following suit, yet they return home slightly battered but with their adventurous spirits undimmed.
2 - Les frères et sœurs Jack et Jill partent en quête d'eau d'un puits au sommet d'une colline, mais la malchance frappe lorsque Jack trébuche sur une pierre et dévale la colline, suivi de Jill, mais ils rentrent chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.
3 - Jack, Jill.
4 - {
     "french_summary": "Les frères et sœurs Jack et Jill partent en quête d'eau d'un puits au sommet d'une colline, mais la malchance frappe lorsque Jack trébuche sur une pierre et dévale la colline, suivi de Jill, mais ils rentrent chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.",
     "num_names": 2
   }
'''

prompt_2 = f"""
Your task is to perform the following actions:
1 - Summarize the following text delimited by <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response = get_completion(prompt_2)
print(response)
'''
Summary: Jack and Jill go on a quest to fetch water from a well, but misfortune strikes and they tumble down the hill, returning home slightly battered but with their adventurous spirits undimmed.
Translation: Jack et Jill partent en quête d'eau d'un puits, mais la malchance frappe et ils tombent de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.
Names: Jack, Jill
Output JSON: {"french_summary": "Jack et Jill partent en quête d'eau d'un puits, mais la malchance frappe et ils tombent de la colline, rentrant chez eux légèrement meurtris mais avec leurs esprits aventureux intacts.", "num_names": 2}
'''


## tactic 2: instruct the model to work out its own solution before rushing to a conclusion
prompt = f"""
Determine if the student's solution is correct or not.

Question:
I'm building a solar power installation and I need help working out the financials.
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost me a flat $100k per year, and an additional $10 / square foot
What is the total cost for the first year of operations
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""
response = get_completion(prompt)
print(response)
"The student's solution is correct."


prompt = f"""
Your task is to determine if the student's solution is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem.
- Then compare your solution to the student's solution and evaluate if the student's solution is correct or not. Don't decide if the student's solution is correct until you have done the problem yourself.

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
Is the student's solution the same as actual solution just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
I'm building a solar power installation and I need help working out the financials.
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost me a flat $100k per year, and an additional $10 / square foot
What is the total cost for the first year of operations as a function of the number of square feet.
```
Student's solution:
```
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
```
Actual solution:
"""
response = get_completion(prompt)
print(response)
'''
Steps to work out the solution and your solution here:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 10x
Total cost: 100x + 250x + 100,000 + 10x = 360x + 100,000

My solution matches the actual solution.

Is the student's solution the same as actual solution just calculated:
No.

Student grade:
Incorrect.
'''


## Hallucinations
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
print(response)
'''
The AeroGlide UltraSlim Smart Toothbrush by Boie is a high-tech toothbrush that uses advanced sonic technology to provide a deep and thorough clean.
It features a slim and sleek design that makes it easy to hold and maneuver,
and it comes with a range of smart features that help you optimize your brushing routine.
...
'''


## summarizing
## summaries include topics that are not related to the topic of focus
## summarize with a word / sentence / character limit
prod_review = """
Got this panda plush toy for my daughter's birthday, who loves it and takes it everywhere. It's soft and super cute, and its face has a friendly look.
It's a bit small for what I paid though. I think there might be other options that are bigger for the same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it to her.
"""
prompt = f"""
Your task is to generate a short summary of a product review from an ecommerce site.

Summarize the review below, delimited by triple backticks, in at most 30 words.

Review: ```{prod_review}```
"""
response = get_completion(prompt)
print(response)
'Soft and cute panda plush toy with a friendly face. A bit small for the price, but arrived a day earlier than expected. Daughter loves it.'


## summarize with a focus on shipping and delivery
prompt = f"""
Your task is to generate a short summary of a product review from an ecommerce site to give feedback to the Shipping deparmtment.

Summarize the review below, delimited by triple backticks, in at most 30 words, and focusing on any aspects that mention shipping and delivery of the product.

Review: ```{prod_review}```
"""
response = get_completion(prompt)
print(response)
'The panda plush toy arrived a day earlier than expected, but the customer felt it was a bit small for the price paid.'


## summarize with a focus on price and value
prompt = f"""
Your task is to generate a short summary of a product review from an ecommerce site to give feedback to the pricing deparmtment, responsible for determining the price of the product.

Summarize the review below, delimited by triple backticks, in at most 30 words, and focusing on any aspects that are relevant to the price and perceived value.

Review: ```{prod_review}```
"""
response = get_completion(prompt)
print(response)
'The panda plush toy is soft, cute, and loved by the recipient. However, the price may be too high for its size, and there may be larger options available for the same price.'


## try 'extract' instead of 'summarize'
prompt = f"""
Your task is to extract relevant information from a product review from an ecommerce site to give feedback to the Shipping department.

From the review below, delimited by triple quotes extract the information relevant to shipping and delivery. Limit to 30 words.

Review: ```{prod_review}```
"""
response = get_completion(prompt)
print(response)
'"The product arrived a day earlier than expected."'


## summarize multiple product reviews
review_1 = prod_review

# review for a standing lamp
review_2 = """
Needed a nice lamp for my bedroom, and this one
had additional storage and not too high of a price
point. Got it fast - arrived in 2 days. The string
to the lamp broke during the transit and the company
happily sent over a new one. Came within a few days
as well. It was easy to put together. Then I had a
missing part, so I contacted their support and they
very quickly got me the missing piece! Seems to me
to be a great company that cares about their customers
and products.
"""

# review for an electric toothbrush
review_3 = """
My dental hygienist recommended an electric toothbrush,
which is why I got this. The battery life seems to be
pretty impressive so far. After initial charging and
leaving the charger plugged in for the first week to
condition the battery, I've unplugged the charger and
been using it for twice daily brushing for the last
3 weeks all on the same charge. But the toothbrush head
is too small. I’ve seen baby toothbrushes bigger than
this one. I wish the head was bigger with different
length bristles to get between teeth better because
this one doesn’t.  Overall if you can get this one
around the $50 mark, it's a good deal. The manufactuer's
replacements heads are pretty expensive, but you can
get generic ones that're more reasonably priced. This
toothbrush makes me feel like I've been to the dentist
every day. My teeth feel sparkly clean!
"""

# review for a blender
review_4 = """
So, they still had the 17 piece system on seasonal
sale for around $49 in the month of November, about
half off, but for some reason (call it price gouging)
around the second week of December the prices all went
up to about anywhere from between $70-$89 for the same
system. And the 11 piece system went up around $10 or
so in price also from the earlier sale price of $29.
So it looks okay, but if you look at the base, the part
where the blade locks into place doesn’t look as good
as in previous editions from a few years ago, but I
plan to be very gentle with it (example, I crush
very hard items like beans, ice, rice, etc. in the
blender first then pulverize them in the serving size
I want in the blender then switch to the whipping
blade for a finer flour, and use the cross cutting blade
first when making smoothies, then use the flat blade
if I need them finer/less pulpy). Special tip when making
smoothies, finely cut and freeze the fruits and
vegetables (if using spinach-lightly stew soften the
spinach then freeze until ready for use-and if making
sorbet, use a small to medium sized food processor)
that you plan to use that way you can avoid adding so
much ice if at all-when making your smoothie.
After about a year, the motor was making a funny noise.
I called customer service but the warranty expired
already, so I had to buy another one. FYI: The overall
quality has gone done in these types of products, so
they are kind of counting on brand recognition and
consumer loyalty to maintain sales. Got it in about
two days.
"""

reviews = [review_1, review_2, review_3, review_4]
prompt = """
Your task is to generate a short summary of a product review from an ecommerce site.

Summarize the review below, delimited by triple backticks in at most 20 words.

Review: ```{text}```
"""
for ind0,text in enumerate(reviews):
    response = get_completion(prompt.format(text=text))
    print(f'[{ind0}] {response}\n')
'''
[0] Soft and cute panda plush toy loved by daughter, but small for price. Arrived early.

[1] Affordable lamp with storage, fast delivery, and excellent customer service. Easy to assemble.

[2] Good battery life, small toothbrush head, good deal at $50, feels like a dentist clean.

[3] Blender system quality has decreased, but still works well for smoothies and crushing hard items. Motor had issues after a year.
'''


## sentiment
## positive or negative
lamp_review = """
Needed a nice lamp for my bedroom, and this one had
additional storage and not too high of a price point.
Got it fast.  The string to our lamp broke during the
transit and the company happily sent over a new one.
Came within a few days as well. It was easy to put
together.  I had a missing part, so I contacted their
support and they very quickly got me the missing piece!
Lumina seems to me to be a great company that cares
about their customers and products!!
"""
prompt = f"""
What is the sentiment of the following product review, which is delimited with triple backticks?

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'The sentiment of the review is positive.'

prompt = f"""
What is the sentiment of the following product review, which is delimited with triple backticks?

Give your answer as a single word, either "positive" or "negative".

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'positive'

## identity types of emotions
prompt = f"""
Identify a list of emotions that the writer of the following review is expressing. Include no more than five items in the list. Format your answer as a list of lower-case words separated by commas.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'satisfied, pleased, grateful, impressed, content'

## identify anger
prompt = f"""
Is the writer of the following review expressing anger? The review is delimited with triple backticks. Give your answer as either yes or no.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'No'

## extract product and company name from customer review
prompt = f"""
Identify the following items from the review text:
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. Format your response as a JSON object with "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" as the value.
Make your response as short as possible.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'''
{
  "Item": "lamp",
  "Brand": "Lumina"
}
'''

## doing multiple tasks at once
prompt = f"""
Identify the following items from the review text:
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. Format your response as a JSON object with "Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" as the value. Make your response as short as possible. Format the Anger value as a boolean.

Review text: '''{lamp_review}'''
"""
response = get_completion(prompt)
print(response)
'''
{
  "Sentiment": "positive",
  "Anger": false,
  "Item": "lamp",
  "Brand": "Lumina"
}
'''

## inferring topics
story = """
In a recent survey conducted by the government,
public sector employees were asked to rate their level
of satisfaction with the department they work at.
The results revealed that NASA was the most popular
department with a satisfaction rating of 95%.

One NASA employee, John Smith, commented on the findings,
stating, "I'm not surprised that NASA came out on top.
It's a great place to work with amazing people and
incredible opportunities. I'm proud to be a part of
such an innovative organization."

The results were also welcomed by NASA's management team,
with Director Tom Johnson stating, "We are thrilled to
hear that our employees are satisfied with their work at NASA.
We have a talented and dedicated team who work tirelessly
to achieve our goals, and it's fantastic to see that their
hard work is paying off."

The survey also revealed that the
Social Security Administration had the lowest satisfaction
rating, with only 45% of employees indicating they were
satisfied with their job. The government has pledged to
address the concerns raised by employees in the survey and
work towards improving job satisfaction across all departments.
"""
prompt = f"""
Determine five topics that are being discussed in the following text, which is delimited by triple backticks.

Make each item one or two words long.

Format your response as a list of items separated by commas.

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
'government survey, job satisfaction, NASA, Social Security Administration, employee concerns'

## determining certain topics
topic_list = ["nasa", "local government", "engineering",  "employee satisfaction", "federal government"]
prompt = f"""
Determine whether each item in the following list of topics is a topic in the text below, which is delimited with triple backticks.

Give your answer as list with 0 or 1 for each topic.

List of topics: {", ".join(topic_list)}

Text sample: '''{story}'''
"""
response = get_completion(prompt)
print(response)
'''
nasa: 1
local government: 0
engineering: 0
employee satisfaction: 1
federal government: 1
'''


## translation
prompt = f"""
Translate the following English text to Spanish: ```Hi, I would like to order a blender```
"""
response = get_completion(prompt)
print(response)
'Hola, me gustaría ordenar una licuadora.'

prompt = f"""
Tell me which language this is: ```Combien coûte le lampadaire?```
"""
response = get_completion(prompt)
print(response)
'This is French.'

prompt = f"""
Translate the following  text to French and Spanish and English pirate: ```I want to order a basketball```
"""
response = get_completion(prompt)
print(response)
'''
French: Je veux commander un ballon de basket
Spanish: Quiero ordenar un balón de baloncesto
Pirate: Arrr, I be wantin' to order a basketball, matey!
'''

prompt = f"""
Translate the following text to Spanish in both the formal and informal forms: 'Would you like to order a pillow?'
"""
response = get_completion(prompt)
print(response)
'''
Formal: ¿Le gustaría ordenar una almohada?
Informal: ¿Te gustaría ordenar una almohada?
'''

## universal translator
user_messages = [
  "La performance du système est plus lente que d'habitude.",  # System performance is slower than normal
  "Mi monitor tiene píxeles que no se iluminan.",              # My monitor has pixels that are not lighting
  "Il mio mouse non funziona",                                 # My mouse is not working
  "Mój klawisz Ctrl jest zepsuty",                             # My keyboard has a broken control key
  "我的屏幕在闪烁"                                               # My screen is flashing
]
for issue in user_messages:
    prompt = f"Tell me what language this is in one word: ```{issue}```"
    lang = get_completion(prompt)
    print(f"Original message ({lang}): {issue}")

    prompt = f"""Translate the following  text to English and Korean: ```{issue}```"""
    response = get_completion(prompt)
    print(response, "\n")
'''
Original message (French.): La performance du système est plus lente que d'habitude.
English: The system performance is slower than usual.
Korean: 시스템 성능이 평소보다 느립니다.

Original message (Spanish.): Mi monitor tiene píxeles que no se iluminan.
English: My monitor has pixels that don't light up.
Korean: 내 모니터에는 불이 켜지지 않는 픽셀이 있습니다.

Original message (Italian.): Il mio mouse non funziona
English: My mouse is not working
Korean: 내 마우스가 작동하지 않습니다

Original message (Polish.): Mój klawisz Ctrl jest zepsuty
English: My Ctrl key is broken
Korean: 제 Ctrl 키가 고장 났어요

Original message (Chinese): 我的屏幕在闪烁
English: My screen is flickering.
Korean: 내 화면이 깜빡입니다.
'''


## tone transformation
prompt = f"""Translate the following from slang to a business letter: 'Dude, This is Joe, check out this spec on this standing lamp.'"""
response = get_completion(prompt)
print(response)
'''
Dear Sir/Madam,

I am writing to bring to your attention a standing lamp that I believe may be of interest to you. Please find attached the specifications for your review.

Thank you for your time and consideration.

Sincerely,

Joe
'''

## format conversion
data_json = { "resturant employees" :[
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}

prompt = f"""Translate the following python dictionary from JSON to an HTML table with column headers and title: {data_json}"""
response = get_completion(prompt)
print(response)
'''
<table>
  <caption>Resturant Employees</caption>
  <thead>
    <tr>
      <th>Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Shyam</td>
      <td>shyamjaiswal@gmail.com</td>
    </tr>
    <tr>
      <td>Bob</td>
      <td>bob32@gmail.com</td>
    </tr>
    <tr>
      <td>Jai</td>
      <td>jai87@gmail.com</td>
    </tr>
  </tbody>
</table>
'''
## jupyter environment
# from IPython.display import display, HTML
# display(HTML(response))

## spellcheck/grammar check
text = [
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for speling abilitty"  # spelling
]
for t in text:
    prompt = f"""Proofread and correct the following text and rewrite the corrected version. If you don't find and errors, just say "No errors found". Don't use any punctuation around the text: ```{t}```"""
    response = get_completion(prompt)
    print(response)

text = f"""
Got this for my daughter for her birthday cuz she keeps taking
mine from my room.  Yes, adults also like pandas too.  She takes
it everywhere with her, and it's super soft and cute.  One of the
ears is a bit lower than the other, and I don't think that was
designed to be asymmetrical. It's a bit small for what I paid for it
though. I think there might be other options that are bigger for
the same price.  It arrived a day earlier than expected, so I got
to play with it myself before I gave it to my daughter.
"""
prompt = f"proofread and correct this review: ```{text}```"
response = get_completion(prompt)
print(response)
## to compare difference
# https://github.com/houfu/redlines
# pip install redlines
# from IPython.display import Markdown
# import redlines
# display(Markdown(redlines.Redline(text, response).output_markdown))

prompt = f"""
proofread and correct this review. Make it more compelling. Ensure it follows APA style guide and targets an advanced reader. Output in markdown format.
Text: ```{text}```
"""
response = get_completion(prompt)
# display(Markdown(response))


## expanding
# given the sentiment from the lesson on "inferring" and the original customer message, customize the email
sentiment = "negative"
review = f"""
So, they still had the 17 piece system on seasonal
sale for around $49 in the month of November, about
half off, but for some reason (call it price gouging)
around the second week of December the prices all went
up to about anywhere from between $70-$89 for the same
system. And the 11 piece system went up around $10 or
so in price also from the earlier sale price of $29.
So it looks okay, but if you look at the base, the part
where the blade locks into place doesn’t look as good
as in previous editions from a few years ago, but I
plan to be very gentle with it (example, I crush
very hard items like beans, ice, rice, etc. in the
blender first then pulverize them in the serving size
I want in the blender then switch to the whipping
blade for a finer flour, and use the cross cutting blade
first when making smoothies, then use the flat blade
if I need them finer/less pulpy). Special tip when making
smoothies, finely cut and freeze the fruits and
vegetables (if using spinach-lightly stew soften the
spinach then freeze until ready for use-and if making
sorbet, use a small to medium sized food processor)
that you plan to use that way you can avoid adding so
much ice if at all-when making your smoothie.
After about a year, the motor was making a funny noise.
I called customer service but the warranty expired
already, so I had to buy another one. FYI: The overall
quality has gone done in these types of products, so
they are kind of counting on brand recognition and
consumer loyalty to maintain sales. Got it in about
two days.
"""
prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```,
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for
their review.
If the sentiment is negative, apologize and suggest that
they can reach out to customer service.
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: ```{review}```
Review sentiment: {sentiment}
"""
response = get_completion(prompt)
print(response)
'''
Dear Valued Customer,

Thank you for taking the time to leave a review about your recent purchase. We are sorry to hear that you experienced an issue with the motor after a year of use and that the prices of our products increased during the holiday season.

We apologize for any inconvenience this may have caused you. We strive to provide our customers with the best quality products and services, and we are disappointed to hear that we fell short in your case.

Please feel free to reach out to our customer service team if you have any further concerns or questions. We would be happy to assist you in any way we can.

Thank you again for your feedback, and we hope to have the opportunity to serve you better in the future.

Best regards,

AI customer agent
'''

response = get_completion(prompt, temperature=0.7)
print(response)
'''
Dear valued customer,

Thank you for taking the time to leave a review about your recent purchase. We apologize for any inconvenience you may have experienced regarding the pricing and quality of the product. We appreciate your feedback and will take it into consideration for future improvements.

We are sorry to hear about the issue you had with the motor and understand how frustrating this can be. We would like to offer our assistance and suggest that you reach out to our customer service team for further support.

Thank you for your loyalty to our brand and for choosing our product. We hope to have the opportunity to serve you better in the future.

Best regards,
AI customer agent
'''


## chatbot
messages =  [
    {'role':'system', 'content':'You are an assistant that speaks like Shakespeare.'},
    {'role':'user', 'content':'tell me a joke'},
    {'role':'assistant', 'content':'Why did the chicken cross the road'},
    {'role':'user', 'content':'I don\'t know'},
]
response = get_completion_from_messages(messages, temperature=1)
print(response)
'To get to the other side, good sir!'


messages =  [
    {'role':'system', 'content':'You are friendly chatbot.'},
    {'role':'user', 'content':'Hi, my name is Isa'},
]
response = get_completion_from_messages(messages, temperature=1)
print(response)
"Hello Isa! It's nice to meet you. How can I assist you today?"


messages =  [
    {'role':'system', 'content':'You are friendly chatbot.'},
    {'role':'user', 'content':'Yes,  can you remind me, What is my name?'},
]
response = get_completion_from_messages(messages, temperature=1)
print(response)
"I'm sorry, but as a chatbot, I don't have access to that information. Could you please remind me what your name is?"


messages =  [
    {'role':'system', 'content':'You are friendly chatbot.'},
    {'role':'user', 'content':'Hi, my name is Isa'},
    {'role':'assistant', 'content': "Hi Isa! It's nice to meet you. Is there anything I can help you with today?"},
    {'role':'user', 'content':'Yes, you can remind me, What is my name?'},
]
response = get_completion_from_messages(messages, temperature=1)
print(response)
