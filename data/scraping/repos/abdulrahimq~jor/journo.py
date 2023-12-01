import os
import openai


openai.api_key = "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"
paragraph = "TEMPLATE TEXT"

prompts_dict = {
    "headline": f'''
				Write a headline for the following article for the New York Times:
				""""""
				The Brady-Johnson Program in Grand Strategy is one of Yale University’s most celebrated and prestigious programs. Over the course of a year, it allows a select group of about two dozen students to immerse themselves in classic texts of history and statecraft, while also rubbing shoulders with guest instructors drawn from the worlds of government, politics, military affairs and the media. But now, a program created to train future leaders how to steer through the turbulent waters of history is facing a crisis of its own. Beverly Gage, a historian of 20th-century politics who has led the program since 2017, has resigned, saying the university failed to stand up for academic freedom amid inappropriate efforts by its donors to influence its curriculum and faculty hiring.
				Headline: Leader of Prestigious Yale Program Resigns, Citing Donor Pressure
				""""""
				{paragraph}\n\n
				Headline: 
				''',
    "interview_questions": f'''
	Create a list of questions for my interview with the CEO of a big corporate:

1. Do you believe there is a right or wrong when it comes to how companies function in the world, or is it all a matter of degree?
2. How would you characterize that disposition of a socially responsible C.E.O.? What are the attributes?
3. There are reasons your company has been a positive force in the world — chiefly economic ones but also ones having to do with personal pleasure. But inarguably some of those beneficial aspects have come at a cost to both human and environmental health. How did you think about those ethical trade-offs?
4. When you meet with business leaders today, is there anything you’re hearing that makes you think, gosh, these people are living in the past?
5. So let’s say you were running a company in Texas, and that company believed in the importance of supporting young families. What would the company’s thinking be around the state’s abortion laws?
6. But surely the issues you just described are connected to the ability to have a family when you want to have a family?
7. More generally, when it comes to corporate responses to political changes, what factors would you be looking at to help you determine the right response for your company?
8. How did your attitude about money change the further along you went in your career, once you became very handsomely compensated?
9. Do you think there is anything gendered to your decision to turn down a raise?
10. Did you feel as if you understood the blowback to the ad and agreed with it? Or was the blowback itself sufficient for you to feel you’d made a mistake?

------------

Create a list of questions for my interview with a cognitive psychologist, psycholinguist, popular science author and public intellectual: 

1. Your new book is driven by the idea that it would be good if more people thought more rationally. But people don’t think they’re irrational. So what mechanisms would induce more people to test their own thinking and beliefs for rationality?
2. Are there aspects of your own life in which you’re knowingly irrational?
3. Do you see any irrational beliefs as useful?
4. What about love?
5. I don’t think I’m alone in feeling that rising authoritarianism, the pandemic and the climate crisis, among other things, are signs that we’re going to hell in a handbasket. Is that irrational of me?
6. How can we know if the fights happening in academia over free speech — which you’ve experienced firsthand — are just the labor pains of new norms? And how do we then judge if those norms are ultimately positive or negative?
7. You said we have to look at whether or not new norms are designed to reward more accurate beliefs or marginalize less accurate ones. How does that apply to subjective issues like, for example, ones to do with identity?
8. What links do you see between rationality and morality?
9. If we agree that well-being is better than its opposite, where does economic equality fit in? Is that a core aspect of well-being?
10. Is it possible that the rising-tide-lifts-all-boats economic argument provides the wealthy with an undue moral cover for the self-interested inequality that their wealth grants them?

------
Create a list of questions for my interview with '''
}


def generate_interview_question(max_tokens=100, paragraph=""):
    if not paragraph:
        paragraph = input(f'''Please enter the main paragraph\n-------------------------\n''')
    print("GENERATED:", paragraph)
    text = prompts_dict['interview_questions'] + f"{paragraph}:\n1."
    print("GENERATED:", text)
    response = openai.Completion.create(engine="davinci-instruct-beta", prompt=text, max_tokens=max_tokens,
                                        temperature=0.8, top_p=1)
    return response['choices'][0]['text']


def generate_article_outline():
    pass


def generate_article_ideas(paragraph="", max_tokens=100) -> str:
    pass


def generate_headline(paragraph="", max_tokens=20) -> str:
    if not paragraph:
        paragraph = input(f'''Please enter the main paragraph\n-------------------------\n''')
    text = prompts_dict['headline'] + f"{paragraph}:\n1."
    response = openai.Completion.create(engine="davinci-instruct-beta", prompt=paragraph, max_tokens=max_tokens,
                                        temperature=0.8, top_p=1)
    return response['choices'][0]['text']


def clean_prompt(prompt):
    prompt = prompt.replace("\n", " ")
    prompt = prompt.replace("\t", " ")
    return prompt


task_dict = {
    "interview_questions": generate_interview_question,
    "article_outline": generate_article_ideas,
    "article_ideas": generate_article_ideas,
    "headline": generate_headline
}


def choose_generation(task_type="1"):
    num = input(f'''Please choose the number what type of text to generate: 
				1) Inteview Questions
				2) Article Outline
				3) Article Ideas
				4) Headline 
				''')
    tasks = {
        "1": "interview_questions",
        "2": "article_outline",
        "3": "article_ideas",
        "4": "headline"
    }
    name = tasks[num]
    return task_dict.get(name, lambda: 'Invalid')()


def parse_important_info(max_tokens=100):
    paragraph = input(f'''Please enter the main paragraph\n-------------------------\n''')
    text = f'''
Text: Outside the headquarters of Asaib Ahl al-Haq, one of the main Iranian-backed militias in Iraq, fighters have posted a giant banner showing the U.S. Capitol building swallowed up by red tents, symbols of a defining event in Shiite history. It’s election time in Iraq, and Asaib Ahl al-Haq — blamed for attacks on American forces and listed by the United States as a terrorist organization — is just one of the paramilitary factions whose political wings are likely to win Parliament seats in Sunday’s voting. The banner’s imagery of the 7th century Battle of Karbala and a contemporaneous quote pledging revenge sends a message to all who pass: militant defense of Shiite Islam. Eighteen years after the United States invaded Iraq and toppled a dictator, the run-up to the country’s fifth general election highlights a political system dominated by guns and money, and still largely divided along sectarian and ethnic lines.

Keywords: Iraq, Iran, Asaib Ahl al-Haq, Karbala, Shiite, Islam, United States

---------------------------------------------------------------------------------------------------

Text: Bitcoin’s proponents dream of a financial system largely free of government meddling. But the first time that cryptocurrency became a national currency, it was imposed on an unwilling population by an increasingly authoritarian ruler using a secretive state-run system. The surprising announcement last month that El Salvador had adopted bitcoin, the world’s largest cryptocurrency, as legal tender caught its population by surprise, and made the poor, conservative Central American nation an unlikely bellwether of a global technological transformation. The outcome of the uncharted experiment could help determine whether cryptocurrency delivers the freedom from regulation that its proponents envision — or whether it becomes another tool of control and enrichment for autocrats and corporations.

Keywords: Bitcoin, El Salvador, Central America, Cryptocurrency, Inflation, Technology

---------------------------------------------------------------------------------------------------

Text: {paragraph}

Keywords:'''
    response = openai.Completion.create(
        engine="davinci-instruct-beta",
        prompt=text,
        temperature=0.3,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.8,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response['choices'][0]['text']


def show_example(task_type: str) -> str:
    return prompts_dict[task_type]


if __name__ == "__main__":
    #print(parse_important_info())
    print("HE")
    #print(choose_generation())
