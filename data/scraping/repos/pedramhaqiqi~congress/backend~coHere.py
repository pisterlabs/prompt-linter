from cgitb import text
import cohere

# Paste your API key here. Remember to not share it publicly 
api_key = 'CaQKrQ4nVgxhALvBmN1OlRmBilGhUs7iIiMBV8Q5'
co = cohere.Client(api_key)

def GenerateSummary(text):
    paraphrasing = '''--Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words--.
    TLDR: Wordle has not gotten more difficult to solve.

    --ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.--
    TLDR: ArtificialIvan has raised $190 million in Series C funding.

    --It’s a breath of fresh air after what we learned last week about residential schools. I sincerely hope that this program aligns with Indigenous culture.
    My second question is about the COVID-19 medical research and vaccine development fund and is for the Public Health Agency of Canada.
    Regarding the $467.6 million for vaccines, Supplementary Estimates (A) state as follows on page 5:
    This funding will support the timely acquisition and deployment of COVID-19 vaccines as well as effective therapeutic treatments for those with COVID-19.
    Yet the paragraph is entitled “Funding for medical research and vaccine developments (COVID-19).” This brings to mind the oft-repeated criticism that it’s difficult to take into account invoices for vaccine purchases.
    Is the $467.6 million for vaccine purchases or partly for new vaccine development?--
    TLDR: The $467.6 million will support the timely acquisition and deployment of COVID-19 vaccines'''

    prompt = f"{paraphrasing}\n\n--{text}--"

    prediction = co.generate(
    model='xlarge',
    prompt=prompt,
    max_tokens=60, 
    temperature=0.3, 
    k=20, 
    p=1, 
    frequency_penalty=0, 
    presence_penalty=0,
    stop_sequences=['--'],
    return_likelihoods='NONE')

    results = prediction.generations[0].text
    results = results.replace("TLDR:","")
    results = results.replace("--","")
    results = results.replace("\n","")
    results.strip()

    return results