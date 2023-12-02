import dsp
import openai
import dotenv

from gpt_text_gym import ROOT_DIR

LLM_MODEL = "text-davinci-002"
colbert_server = (
    "http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search"
)
OPENAI_TEMPERATURE = 0.0
OPENAI_API_KEY = dotenv.get_key(ROOT_DIR / ".env", "API_KEY")

train = [
    (
        'Who produced the album that included a re-recording of "Lithium"?',
        ["Butch Vig"],
    ),
    (
        "Who was the director of the 2009 movie featuring Peter Outerbridge as William Easton?",
        ["Kevin Greutert"],
    ),
    (
        "The heir to the Du Pont family fortune sponsored what wrestling team?",
        ["Foxcatcher", "Team Foxcatcher", "Foxcatcher Team"],
    ),
    ("In what year was the star of To Hell and Back born?", ["1925"]),
    (
        "Which award did the first book of Gary Zukav receive?",
        ["U.S. National Book Award", "National Book Award"],
    ),
    (
        "What city was the victim of Joseph Druces working in?",
        ["Boston, Massachusetts", "Boston"],
    ),
]

train = [dsp.Example(question=question, answer=answer) for question, answer in train]

Question = dsp.Type(prefix="Question:", desc="${the question to be answered}")
Answer = dsp.Type(
    prefix="Answer:",
    desc="${a short factoid answer, often between 1 and 5 words}",
    format=dsp.format_answers,
)

qa_template = dsp.Template(
    instructions="Answer questions with short factoid answers.",
    question=Question(),
    answer=Answer(),
)


def vanilla_QA_LM(question: str) -> str:
    demos = dsp.sample(train, k=7)
    example = dsp.Example(question=question, demos=demos)
    example, completions = dsp.generate(qa_template)(example, stage="qa")
    return completions[0].answer


Context = dsp.Type(
    prefix="Context:\n",
    desc="${sources that may contain relevant content}",
    format=dsp.passages2text,
)

qa_template_with_passages = dsp.Template(
    instructions=qa_template.instructions,
    context=Context(),
    question=Question(),
    answer=Answer(),
)


def retrieve_then_read_QA(question: str) -> str:
    demos = dsp.sample(train, k=7)
    passages = dsp.retrieve(question, k=1)

    example = dsp.Example(question=question, context=passages, demos=demos)
    example, completions = dsp.generate(qa_template_with_passages)(example, stage="qa")

    return completions.answer


if __name__ == "__main__":
    # Set up dsp
    lm = dsp.GPT3(LLM_MODEL, OPENAI_API_KEY)
    rm = dsp.ColBERTv2(url=colbert_server)
    dsp.settings.configure(lm=lm, rm=rm)

    question = "What is the capital of the United States?"
    answer = vanilla_QA_LM(question)
    print("Vanilla QA LM answer:")
    lm.inspect_history(n=1)

    # Doesn't work, because the retrieval model is no longer online
    # answer = retrieve_then_read_QA(question)
    # print("QA LM with retrieval answer:")
    # lm.inspect_history(n=1)
