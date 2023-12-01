import os
import pickle
# os.environ["LANGCHAIN_HANDLER"] = "langchain"
# os.environ["LANGCHAIN_SESSION"] = "Test"

import numpy as np
from sklearn.manifold import TSNE
import faiss
import openai
import pandas as pd
import plotly.express as px
from clearml import Task
from colored import attr, fg
from joblib import Memory
import streamlit as st

from langchain import VectorDBQAWithSourcesChain
from langchain.callbacks import ClearMLCallbackHandler, StdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

location = './cachedir'
memory = Memory(location, verbose=0)


os.environ["OPENAI_API_KEY"] = "[YOUR_KEY_HERE]"
openai.api_key = "[YOUR_KEY_HERE]"

# Initialize the ClearML Task
task = Task.init(
    project_name="promptimyzer",
    task_name="Docs Optimizer",
    reuse_last_task_id=False,
    output_uri=True
)


def prepare_examples(examples):
    output = ""
    for example in examples:
        output += "====\n"
        output += f"Prompt: {example['instructions']}\n"
        output += f"Score: {example['score']}\n"
        output += f"Comments: {example['comments']}\n\n"
    output += "====\n"
    return output


class MetaOptimizer:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # ClearML Callback
        # Setup and use the ClearML Callback
        self.clearml_callback = ClearMLCallbackHandler(
            task_type="inference",
            project_name="langchain_callback_demo",
            task_name="llm",
            tags=["test"],
            # Change the following parameters based on the amount of detail you want tracked
            visualize=True,
            complexity_metrics=True,
            stream_logs=True
        )
        manager = CallbackManager([StdOutCallbackHandler(), self.clearml_callback])
        # Instructions Generator
        self.instruction_generator = ChatOpenAI(temperature=0.7, model_name=model_name, callback_manager=manager, verbose=True)
        self.examples = [
            {
                "instructions": """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.""",
                "score": "4",
                "comments": "It nicely says it doesn't know when applicable, but one of the answers was very wordy and became a tangent explanation instead of answering the questions."
            },
            {
                "instructions": "Below is a question from our forum. You have access to the relevant parts of our documentation to answer it. If you don't know the answer, please say so clearly. False information in this context is much worse than no information at all.",
                "score": "2",
                "comments": "There were no sources given! It is very important that the bot add sources."
            },
            {
                "instructions": "Please explain.",
                "score": "0",
                "comments": "The bot had no idea what it was doing. It probably needs better context and explanation first."
            }
        ]

        # Documentation
        with open("faiss_store.pkl", "rb") as f:
            self.store = pickle.load(f)
        self.store.index = faiss.read_index("docs.index")

        # QA Bot
        self.qa_bot = ChatOpenAI(model_name=model_name, temperature=0.2, callback_manager=manager, verbose=True)

        # Embedder for visualization
        self.embed_cached = memory.cache(embed)
        # self.embed_cached = embed
        self.iteration = 0

    def create_instructions(self):
        instruction_prompt_template = """
I'm looking for the optimal initial prompt for a documentation QA chatbot.
The prompt should instruct the documentation bot on how to best answer the question.

Below are examples of prompts as evaluated by a human annotator.
Your job is to generate a new unique instruction prompt that you think will be evaluated better than the example prompts.
In this case better means a higher score.
You're essentially interpolating in embeddingspace in search of the optimal prompt.
I want you to create an embeddingspace of the examples and create something that is closer to better prompts and further away from bad performing ones.

Only answer with the exact prompt you generated, no extra text is to be added.

Examples prompts, each with their score and some comments explaining that score:
{examples}

Optimal Prompt:
        """

        instruction_prompt = PromptTemplate(
            input_variables=["examples"],
            template=instruction_prompt_template,
        )

        instruction_chain = LLMChain(llm=self.instruction_generator, prompt=instruction_prompt)
        instructions = instruction_chain.run(examples=prepare_examples(self.examples))

        self.iteration += 1
        self.clearml_callback.flush_tracker(langchain_asset=instruction_chain, name="instruction_chain")

        return instructions
    

    def docs_qa(self):
        # Get the original chainlang prompt and add our own instructions
        combine_prompt_template = """{instructions}

Examples

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia's Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won't stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet's use this moment to reset. Let's stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet's stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can't change how divided we've been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who'd grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I've always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I'm taking robust action to make sure the pain of our sanctions  is targeted at Russia's economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what's happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson.
SOURCES: N/A

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["summaries", "question", "instructions"]
        )
        qa_chain = VectorDBQAWithSourcesChain.from_llm(llm=self.qa_bot, vectorstore=self.store, combine_prompt=combine_prompt)

        return qa_chain
    
    def save(self):
        with open("prompt_history.pkl", "wb") as f:
            pickle.dump(self.examples, f)
        print("Saved prompts!")
    
    def load(self):
        if os.path.isfile("prompt_history.pkl"):
            with open("prompt_history.pkl", "rb") as f:
                self.examples = pickle.load(f)
            print("Loaded previous prompts!")
    
    def plot_embeddings(self, instructions):
        df = pd.DataFrame(self.examples)
        df = df.append(
            {
                'instructions': instructions,
                'score': '-1',
                'comments': 'New Sample'
            },
            ignore_index=True
        )
        print(df)
        df['embedding'] = df['instructions'].apply(self.embed_cached)

        # Create a t-SNE model and transform the data
        tsne = TSNE(n_components=3, perplexity=2, random_state=42, init='random', learning_rate=200)
        vis_dims = pd.DataFrame(tsne.fit_transform(np.array(df['embedding'].to_list())))
        vis_dims['score'] = df['score'].astype(int)
        fig = px.scatter_3d(vis_dims, x=0, y=1, z=2, color='score')
        Task.current_task().get_logger().report_plotly(
            title="Embedding Space",
            series="Instructions",
            iteration=self.iteration,
            figure=fig
        )
    

def render_questions(qa_chain, questions, instructions):
    for i, question in enumerate(questions):
        st.subheader(f"Q{i}")
        reply = qa_chain({"question": question, "instructions": instructions})
        st.text_area(label="Question", value=reply["question"], disabled=True)
        st.text_area(label="Answer", value=f'{reply["answer"]}\n\nSOURCES: {reply["sources"]}', disabled=True)


def embed(text):
    embedder = OpenAIEmbeddings()
    return embedder.embed_query(text)


def main(meta_optimizer):
    # Capture the previous instructions and feedback (or skip if this is first run)
    if st.session_state.instructions:
        meta_optimizer.examples.append(
            {
                "instructions": st.session_state.instructions,
                "score": st.session_state.score,
                "comments": st.session_state.feedback
            }
        )
    
    # Given the examples, generate a new instruction prompt
    instructions = meta_optimizer.create_instructions()
    # st_instructions.text_area(label="Instructions Prompt", value=instructions, disabled=True, key="instructions")
    st.session_state.instructions = instructions

    # Plot the new instructions in embedding space together with the
    # previous ones, so we can visually follow along!
    meta_optimizer.plot_embeddings(instructions=instructions)

    # Given the generated new prompt from above, create a QA bot
    # based on our own documentation.
    qa_chain = meta_optimizer.docs_qa()

    # Inference over each question and write the answer to the streamlit app
    for i, question in enumerate(questions):
        reply = qa_chain({"question": question, "instructions": instructions})
        st.session_state[f'A{i}'] = f"{reply['answer']}\n\nSOURCES: {reply['sources']}"

    # Capture this run in clearml
    meta_optimizer.clearml_callback.flush_tracker(langchain_asset=qa_chain, name="qa_chain")


if __name__ == "__main__":
    # Create the metaoptimizer, load saved prompt history and format
    # those into instructions for GPT
    meta_optimizer = MetaOptimizer()
    meta_optimizer.load()

    # Testing examples
    questions = [
        "How do I create ClearML credentials?",
        "How does clearml ochestration and scheduling work?",
        "Can you clone an entire ClearML Project?",
        "How do I go to the built-in ClearML Serving dashboard?",
        "Where can I find the datalake in ClearML?",
        "What do each of the databases in the ClearML server do?"
    ]

    # Streamlit
    st.title('🎉 promptimyzer')

    st_instructions = st.text_area(label="Instructions Prompt", disabled=True, key="instructions")

    # Get the text areas ready
    for i, question in enumerate(questions):
        st.subheader(f"Q{i}")
        st.text_area(label="Question", value=question, disabled=True, key=f"Q{i}")
        st.text_area(label="Answer", disabled=True, key=f"A{i}")


    st.subheader("Feedback")
    st.number_input("Score", key="score")
    st.text_area(label="General feedback to improve the initial prompt", key="feedback")

    print("End here!")

    # Add the button every time 
    st.button(label="Submit", key="submit_feedback", on_click=main, args=(meta_optimizer,))

    # instructions = meta_optimizer.create_instructions()
    # meta_optimizer.save()
    # meta_optimizer.clearml_callback.flush_tracker(langchain_asset=qa_chain, name="qa_chain")
