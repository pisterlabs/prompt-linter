import re, json
from datetime import timedelta

import openai
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret, String
from prefect.tasks import task_input_hash
from prefect_slack import SlackCredentials
from prefect_slack.messages import send_chat_message

from prefect_hermes.blocks import OpenAICompletion

@task(name="Generate chat log from prefect documentation")#, cache_key_fn=task_input_hash)
def parse_faq(qa: str = "faq") -> str:
    """Caching context discovery and cleaning - should READ some semi-structured data instead
    
    need to figure out a consistent manner to compile QAs from forums
    """
    avoid_strs = ["🔒", "<aside>"]
    
    secret_content = Secret.load(qa).get()

    content = open('prefect_hermes/context.md', 'r').read().replace("?", "? ??").replace("\n", "")

    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    md_links = link_pattern.findall(content)

    for text, link in md_links:
        content = content.replace(f"[{text}]({link})", text)

    raw_QAs = [i.split("??", 1) for i in content.strip().split("##") if i != ""]

    shareable_QAs = [i for i in raw_QAs if all(j not in i[-1] for j in avoid_strs)]

    annotated_QAs = "".join(
        map(lambda i: f"\nPerson: {i[0]}\nHermes: {i[1]}\n", shareable_QAs)
    )
    # # temporary way to generate training data - requires manual edits at the moment
    # with open('training.json', 'w') as f:
    #         for question, answer in shareable_QAs:
    #             prompt, completion = f"Person: {question}", f" {answer}\n"
    #             f.write(json.dumps({'prompt': prompt, 'completion': completion})+"\n")

    return annotated_QAs


@task(
    name="Solicit response from OpenAI Completion engine"#,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=1)
)
def ask(question: str, chat_log: str = None, model: str = "text-davinci-002") -> str:
    """Opportunity to cache more 
    """
    openai.api_key = Secret.load("openai-api-key").get()
    completion_scheme = OpenAICompletion.load('completion-scheme').to_dict()
    context = String.load("context").value
    
    start_sequence = "\n\nHermes:"
    restart_sequence = "\n\nPerson:"
    prompt_text = f"""{context}{chat_log}{restart_sequence}:{question}{start_sequence}"""
    
    completion_scheme.update(
        dict(
            engine=model,
            prompt=prompt_text
        )
    )
    
    response = openai.Completion.create(**completion_scheme)["choices"][0]["text"]
        
    match response:
        case "":
            print('bad context i think: raise CustomError')
            raise ValueError
        case str():
            return str(response)
        case _:
            raise AssertionError


@flow
def respond_in_slack(end_user_id: str, question: str):
    """A flow to process a user question in slack, provided question and user info by listener.

    Args:
        end_user_id (str): Slack ID of the end user making request via slash command
        question (str): Question to answer from slack.
    """
    logger = get_run_logger()
    
    logger.info(f"Received question from slack user with user_id: {end_user_id!r}")
    
    historical_context = parse_faq()

    response = ask(question=question, chat_log=historical_context)
    
    
    send_chat_message(
        slack_credentials=SlackCredentials(Secret.load("slack-token").get()),
        channel="#testing-slackbots",
        text=f"Q: *{question}*\nA: {response}",
    )

if __name__ == "__main__":
    respond_in_slack(
        end_user_id='U03RX2A8LK0',
        question="Does Marvin like being a rubber duck?"
    )