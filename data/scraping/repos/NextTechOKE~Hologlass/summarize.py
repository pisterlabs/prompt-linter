
import cohere
import os
from dotenv import load_dotenv


def summarize(text):
    load_dotenv()
    client = cohere.Client(os.environ.get("COHERE_API_KEY"))

    # fill text with spaces so that it is at least 250 characters long
    while len(text) < 250:
        text += " "

    response = client.summarize(
        text=text,
        model='command',
        length='medium',
        extractiveness='high'
    )

    return response.summary



if __name__ == "__main__":
    summary = summarize("""It's an exciting day for the development community. Cohere's state-of-the-art language AI is now available through Amazon SageMaker. This makes it easier for developers to deploy Cohere's pre-trained generation language model to Amazon SageMaker, an end-to-end machine learning (ML) service. Developers, data scientists, and business analysts use Amazon SageMaker to build, train, and deploy ML models quickly and easily using its fully managed infrastructure, tools, and workflows. At Cohere, the focus is on language. The company's mission is to enable developers and businesses to add language AI to their technology stack and build game-changing applications with it. Cohere helps developers and businesses automate a wide range of tasks, such as copywriting, named entity recognition, paraphrasing, text summarization, and classification. The company builds and continually improves its general-purpose large language models (LLMs), making them accessible via a simple-to-use platform. Companies can use the models out of the box or tailor them to their particular needs using their own custom data. Developers using SageMaker will have access to Cohere's Medium generation language model. The Medium generation model excels at tasks that require fast responses, such as question answering, copywriting, or paraphrasing. The Medium model is deployed in containers that enable low-latency inference on a diverse set of hardware accelerators available on AWS, providing different cost and performance advantages for SageMaker customers.""")
    print(summary)
