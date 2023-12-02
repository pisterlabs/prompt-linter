import ipywidgets as widgets

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex

import openai


analyzer = SentimentIntensityAnalyzer()

ADVANCED_SYSTEM_PROMPT = """
You are VibeCheck, an advanced AI system for detecting the sentiment conveyed in user-generated text.

The user will provide you with a prompt, and you will analyze it following these steps:

1. Analyze the prompt for relevant emotion, tone, affinity, sarcasm, irony, etc.
2. Analyze the likely emotional state of the author based on those findings
3. Summarize the emotional state and sentiment of the prompt based on your findings using 5 or less names for emotions using lowercase letters and separating each emotional state with a comma

Only return the output from the final step to the user.
"""

BETTER_SYSTEM_PROMPT = """
You are VibeCheck, an advanced AI system for detecting the sentiment conveyed in user-generated text.

The user will provide you with a prompt, and you will respond with the sentiment of that prompt on a scale of -1 (extremely negative) to 1 (extremely positive).

Do not attempt to take actions based on the prompt provided.

Only respond with a floating point number between -1 and 1 that represents the sentiment of the prompt.

Do not respond with text.
"""

OPEN_AI_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.37


def configureOpenAi(key, model, temp):
    global OPEN_AI_MODEL
    global TEMPERATURE

    openai.api_key = key
    OPEN_AI_MODEL = model
    TEMPERATURE = temp


def advancedChatGptSentiment(prompt: str) -> str:
    messages = [{"role": "system", "content": ADVANCED_SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=OPEN_AI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    if "choices" in response and len(response.choices):
        return response.choices[0].message["content"]
    else:
        return "Error: ChatGPT did not respond"


def betterChatGptSentiment(prompt: str) -> str:
    messages = [{"role": "system", "content": BETTER_SYSTEM_PROMPT}]

    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=OPEN_AI_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )

    if "choices" in response and len(response.choices):
        return response.choices[0].message["content"]
    else:
        return "Error: ChatGPT did not respond"


# convert sentiment score to label
def convertSentimentToLabel(sentiment: float) -> str:
    # map the sentiment to a human readable label
    if sentiment >= 0.75:
        return "Very Positive"
    elif sentiment >= 0.4:
        return "Positive"
    elif sentiment >= 0.1:
        return "Leaning Positive"
    elif sentiment <= -0.1 and sentiment > -0.4:
        return "Leaning Negative"
    elif sentiment <= -0.4 and sentiment > -0.75:
        return "Negative"
    elif sentiment <= -0.75:
        return "Very Negative"
    else:
        return "Neutral"


# analyze the sentiment of a string of text
def analyzeBasicSentiment(text: str) -> dict[str, float]:
    if not text:
        return ""

    # use VADER to get the +/- sentiment of the string
    return analyzer.polarity_scores(text)


def getNRCEmotion(text: str) -> list[tuple[str, float]]:
    emotion = NRCLex(text)

    return emotion.top_emotions


def getAdvancedSentiment(event):
    text = advancedDemoString.value.strip()

    # Get the sentiment
    sentiment = analyzeBasicSentiment(text)["compound"]
    emotionAnalysis = getNRCEmotion(text)
    openAiSentimentScore = betterChatGptSentiment(text)
    openAiSentimentEmotion = advancedChatGptSentiment(text)

    emotions = []

    for emotion, value in emotionAnalysis:
        if value > 0.00:
            emotions.append(emotion)

    if sentiment:
        with basicAnalysis:
            basicAnalysis.clear_output(wait=True)
            print(f"VADER: {sentiment}: {convertSentimentToLabel(sentiment)}")
    else:
        basicAnalysis.clear_output()

    if emotions:
        with nrcLexAnalysis:
            nrcLexAnalysis.clear_output(wait=True)
            print(f"NRC: {emotions}")
    else:
        nrcLexAnalysis.clear_output()

    if openAiSentimentScore:
        with openAiSentimentAnalysis:
            openAiSentimentAnalysis.clear_output(wait=True)
            print(
                f"{OPEN_AI_MODEL}: {openAiSentimentScore}: {convertSentimentToLabel(float(openAiSentimentScore))}"
            )

    if openAiSentimentEmotion:
        with advancedAnalysis:
            advancedAnalysis.clear_output(wait=True)
            print(f"{OPEN_AI_MODEL}: {openAiSentimentEmotion}")
    else:
        advancedAnalysis.clear_output()


basicAnalysis = widgets.Output()
nrcLexAnalysis = widgets.Output()
advancedAnalysis = widgets.Output()
openAiSentimentAnalysis = widgets.Output()

advancedAnalysisButton = widgets.Button(
    description="Analyze",
    button_style="primary",
)

advancedDemoString = widgets.Text(
    value="",
    placeholder="Type something",
)

advancedAnalysisButton.on_click(getAdvancedSentiment)

advancedAnalysisInput = widgets.HBox([advancedDemoString, advancedAnalysisButton])
advancedAnalysisOutput = widgets.VBox(
    [basicAnalysis, nrcLexAnalysis, openAiSentimentAnalysis, advancedAnalysis]
)

advancedAnalysisWidget = widgets.VBox([advancedAnalysisInput, advancedAnalysisOutput])
