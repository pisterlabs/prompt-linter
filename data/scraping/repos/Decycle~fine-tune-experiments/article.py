import yaml
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.output_parsers import ListOutputParser
from langchain.chains import LLMChain
import re
import json
import requests

from dotenv import load_dotenv
load_dotenv()

chat4 = ChatOpenAI(
    model="gpt-4"
)  # type: ignore

chat = ChatOpenAI(
    model="gpt-4"
)  # type: ignore


class NewLineOutputParser(ListOutputParser):
    def parse(self, output: str) -> list:
        outputs = output.split("\n")
        return [re.sub(r"\d+\.?", "", line).strip() for line in outputs if line != ""]


def clean_text(text: str):
    text = re.sub(r"\s*\n\s*", '\n', text)
    return text.strip()


article = clean_text("""
Whistleblower tells Congress the US is concealing ‘multi-decade’ program that captures UFOs

A former Air Force intelligence officer testifies that the U.S. is concealing a longstanding program that retrieves and reverse engineers unidentified flying objects or UAPs, “unidentified aerial phenomena.” The Pentagon has denied his claims. (July 26)

BY NOMAAN MERCHANT
Updated 1:03 PM PDT, July 26, 2023

WASHINGTON (AP) — The U.S. is concealing a longstanding program that retrieves and reverse engineers unidentified flying objects, a former Air Force intelligence officer testified Wednesday to Congress. The Pentagon has denied his claims.

Retired Maj. David Grusch’s highly anticipated testimony before a House Oversight subcommittee was Congress’ latest foray into the world of UAPs — or “unidentified aerial phenomena,” which is the official term the U.S. government uses instead of UFOs. While the study of mysterious aircraft or objects often evokes talk of aliens and “little green men,” Democrats and Republicans in recent years have pushed for more research as a national security matter due to concerns that sightings observed by pilots may be tied to U.S. adversaries.

Grusch said he was asked in 2019 by the head of a government task force on UAPs to identify all highly classified programs relating to the task force’s mission. At the time, Grusch was detailed to the National Reconnaissance Office, the agency that operates U.S. spy satellites.


“I was informed in the course of my official duties of a multi-decade UAP crash retrieval and reverse engineering program to which I was denied access,” he said.


Asked whether the U.S. government had information about extraterrestrial life, Grusch said the U.S. likely has been aware of “non-human” activity since the 1930s.

The Pentagon has denied Grusch’s claims of a coverup. In a statement, Defense Department spokeswoman Sue Gough said investigators have not discovered “any verifiable information to substantiate claims that any programs regarding the possession or reverse-engineering of extraterrestrial materials have existed in the past or exist currently.” The statement did not address UFOs that are not suspected of being extraterrestrial objects.

Grusch says he became a government whistleblower after his discovery and has faced retaliation for coming forward. He declined to be more specific about the retaliatory tactics, citing an ongoing investigation.

“It was very brutal and very unfortunate, some of the tactics they used to hurt me both professionally and personally,” he said.

NASA talks UFOs with public ahead of final report on unidentified flying objects
FILE - The Pentagon is seen from Air Force One as it flies over Washington, March 2, 2022. A new Pentagon office set up to track reports of unidentified flying objects has received “several hundreds” of new reports, but no evidence so far of alien life. That's according to the leadership of the All-domain Anomaly Resolution Office. (AP Photo/Patrick Semansky, File)
Pentagon has received ‘several hundreds’ of new UFO reports
Deputy Director of Naval Intelligence Scott Bray points to a video display of a UAP during a hearing of the House Intelligence, Counterterrorism, Counterintelligence, and Counterproliferation Subcommittee hearing on "Unidentified Aerial Phenomena," on Capitol Hill, Tuesday, May 17, 2022, in Washington. (AP Photo/Alex Brandon)
Congress dives into UFOs, but no signs of extraterrestrials
Rep. Glenn Grothman, R-Wis., chaired the panel’s hearing and joked to a packed audience, “Welcome to the most exciting subcommittee in Congress this week.”

There was bipartisan interest in Grusch’s claims and a more sober tone than other recent hearings featuring whistleblowers celebrated by Republicans and criticized by Democrats. Lawmakers in both parties asked Grusch about his study of UFOs and the consequences he faced and how they could find out more about the government’s UAP programs.

“I take it that you’re arguing what we need is real transparency and reporting systems so we can get some clarity on what’s going on out there,” said Rep. Jamie Raskin, D-Md.


Some lawmakers criticized the Pentagon for not providing more details in a classified briefing or releasing images that could be shown to the public. In previous hearings, Pentagon officials showed a video taken from an F-18 military plane that showed an image of one balloon-like shape.

Pentagon officials in December said they had received “several hundreds” of new reports since launching a renewed effort to investigate reports of UFOs.

At that point, “we have not seen anything, and we’re still very early on, that would lead us to believe that any of the objects that we have seen are of alien origin,” said Ronald Moultrie, the undersecretary of defense for intelligence and security. “Any unauthorized system in our airspace we deem as a threat to safety.”
""")


def get_question_chain():

    questions_instruction = clean_text("""
        Ask twenty extremely detailed questions about the following article, it should be impossible to answer them without reading the article.
        {article}
        Questions:
    """)
    question_template = PromptTemplate(
        template=questions_instruction,
        input_variables=["article"],
    )

    questions_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(prompt=question_template),
    ])

    return LLMChain(
        llm=chat4,
        prompt=questions_prompt,
        verbose=True,
        output_parser=NewLineOutputParser(),
    )


def get_answering_chain():

    instruction = clean_text("""
        Answer the question: "{question}" about the following article. Include how you get that answer.
        Article:{article}
        Answer:
    """)

    template = PromptTemplate(
        template=instruction,
        input_variables=["article", "question"],
    )

    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate(prompt=template),
    ])

    return LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True,
        output_parser=NewLineOutputParser(),
    )


def get_training_data():
    with get_openai_callback() as cb:
        question_chain = get_question_chain()
        answer_chain = get_answering_chain()

        question_outputs = question_chain.apply([
            {"article": article}
        ] * 5)

        questions = []

        for output in question_outputs:
            questions.extend(output['text'])

        with open("questions.json", 'w') as f:
            json.dump(questions, f)

        # with open("questions.txt", 'r') as f:
        #     questions = json.load(f)

        answer_outputs = answer_chain.apply([
            {"article": article, "question": question}
            for question in questions
        ])

        answers = []

        for output in answer_outputs:
            answers.append(output['text'][0])

        output = [
            {
                'instruction': question,
                'output': answer
            } for question, answer in zip(questions, answers)
        ]

        with open("datasets/article.json", 'w') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print(cb)


def get_cfg(use_lora=False):
    config = "default.yml"

    with open(config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    cfg['lora_name'] = "article"
    cfg['use_lora'] = use_lora
    cfg['warmup_steps'] = 5
    cfg['eval_steps'] = 20
    cfg['num_epochs'] = 3
    cfg['val_set_size'] = 0

    return cfg


endpoint = "http://localhost:8000"


def setup(cfg):
    result = requests.post(f"{endpoint}/setup", json={"cfg": cfg}).json()

    if result['status'] == 'success':
        print("Setup Complete")
    else:
        print(result['message'])


def train(trainning_data):
    result = requests.post(f"{endpoint}/train",
                           json={"trainning_data": trainning_data}).json()

    if result['status'] == 'success':
        print("Training Complete")
    else:
        print(result['message'])


def inference(instruction):
    result = requests.post(f"{endpoint}/inference",
                           json={"instruction": instruction}).json()
    if result['status'] == 'success':
        print("Inference Complete")
        return result['result']
    else:
        print(result['message'])


if __name__ == '__main__':

    # get_training_data()
    cfg = get_cfg(use_lora=False)
    setup(cfg)

    # with open("article.json") as f:
    #     trainning_data = json.load(f)
    # train(trainning_data)
    # old_result = inference(
    #     "What do you know about the recent Congress's whisteblower's testimony on UFOs?")
    # cfg = get_cfg(use_lora=True)
    # setup(cfg)
    result = inference(
        "What do you know about the recent Congress's whisteblower's testimony on UFOs?")

    # print("Old Result:", old_result)
    print("New Result:", result)
    # old_result = trainer.inference(
    #     "What do you know about the recent Congress's whisteblower's testimony on UFOs?")
    # # trainer.train("datasets/article.json")

    # print("Old Result:", old_result)
    # while True:
    #     question = input("Question: ")
    #     result = trainer.inference(
    #         question)
    #     print("New Result:", result)
