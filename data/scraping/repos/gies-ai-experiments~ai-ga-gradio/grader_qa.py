from langchain import FAISS
from langchain import LLMMathChain
from langchain.agents import AgentType, create_csv_agent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.tools import Tool

import utils


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


class GraderQA:
    def __init__(self, grader, embeddings):
        self.grader = grader
        self.llm = self.grader.llm
        self.folder_path = "vector_stores/"
        self.summary_index_name = "canvas-discussions-summary"
        self.summary_index_file = "vector_stores/canvas-discussions-summary.faiss"
        self.summary_pickle_file = "vector_stores/canvas-discussions-summary.pkl"
        self.qa_index_name = "canvas-discussions-qa"
        self.qa_index_file = "vector_stores/canvas-discussions-qa.faiss"
        self.qa_pickle_file = "vector_stores/canvas-discussions-qa.pkl"
        self.summary_docs = utils.get_csv_files(self.grader.csv, source_column='student_name')
        self.qa_docs = utils.get_csv_files(self.grader.csv, source_column='student_name',
                                           field_names=['student_name', 'total_score', 'score_breakdown'])
        self.rubric_text = grader.rubric_text
        self.summary_index = self.get_search_index(embeddings)
        self.qa_index = self.get_qa_index(embeddings)
        self.chain = self.create_chain(embeddings)
        self.qa_chain = self.create_qa_chain()
        self.math_chain = self.create_math_chain()
        self.tools = self.get_tools()
        self.memory = ConversationBufferMemory(memory_key='chat_history',
                                               return_messages=True,
                                               output_key='answer')
        self.agent = self.create_agent()
        self.tokens = None
        self.question = None

    def load_all_indexes(self, embeddings):
        return self.get_search_index(embeddings), self.get_qa_index(embeddings)

    def get_search_index(self, embeddings):
        if utils.index_exists(self.summary_pickle_file, self.summary_index_file):
            # Load index from pickle file
            search_index = utils.load_index(self.folder_path, self.summary_index_name, embeddings)
        else:
            search_index = utils.create_index(self.folder_path, self.summary_index_name, embeddings, self.summary_docs)
            print("Created index")
        return search_index

    def get_qa_index(self, embeddings):
        if utils.index_exists(self.qa_pickle_file, self.qa_index_file):
            # Load index from pickle file
            search_index = utils.load_index(self.folder_path, self.qa_index_name, embeddings)
        else:
            search_index = utils.create_index(self.folder_path, self.qa_index_name, embeddings, self.qa_docs)
            print("Created index")
        return search_index

    def create_chain(self, embeddings):
        if not self.summary_index:
            self.summary_index = self.get_search_index(embeddings)

        question_prompt, combine_prompt = self.create_map_reduce_prompt()
        # create agent, 1 chain for summary based question, 2nd chain for semantic retrieval based question
        qa_chain = load_qa_chain(self.llm, chain_type="map_reduce", question_prompt=question_prompt,
                                 combine_prompt=combine_prompt, verbose=True)

        chain = RetrievalQA(combine_documents_chain=qa_chain,
                            retriever=self.summary_index.as_retriever(search_type='mmr',
                                                                      search_kwargs={'lambda_mult': 1, 'fetch_k': 50,
                                                                                     'k': 30}),
                            return_source_documents=True, verbose=True, )
        return chain

    def create_qa_chain(self):
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",
                                         retriever=self.qa_index.as_retriever(search_type='mmr',
                                                                              search_kwargs={'lambda_mult': 1,
                                                                                             'fetch_k': 50,
                                                                                             'k': 30}), verbose=True)
        return qa

    def create_math_chain(self):
        return LLMMathChain.from_llm(llm=self.llm, verbose=True)

    def get_tools(self):
        tools = [
            Tool(
                name="Grading Score Results",
                func=self.run_qa_chain,
                description="useful when you need to answer questions related to GRADES, SCORING or SCORE BREAKDOWN(INDIVIDUAL OR OVERALL) based questions from the grading results of the canvas discussion. Use this more often because this has a higher accuracy about the SCORING and GRADES of the students."
            ),
            Tool(
                name="Summary",
                func=self.run_summary_chain,
                description="useful when you need to answer summary based questions for all students' grading results for the canvas discussion where the question is complicated and ONLY WHEN the answer is not directly available in the grading score results"
            ),
            Tool(
                name="Calculator",
                func=self.run_math_chain,
                description="Useful for when you need to compute mathematical expressions"
            )
        ]
        return tools

    def create_agent(self):
        # Initialize a Conversational Agent with the existing chain as a tool
        # planner = load_chat_planner(self.llm)
        #
        # # agent = initialize_agent(self.tools, self.llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=self.memory)
        # executor = load_agent_executor(self.llm,self.tools, verbose=True)
        #
        #
        # agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
        # agent = initialize_agent(
        #     self.tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        # )

        agent = create_csv_agent(
            self.llm,
            self.grader.csv,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent

    def create_map_reduce_prompt(self):
        system_template = f"""Use the following student's grading result document to answer a summary based question. The question will always be related to the overall grading results, feedback, score, summary of student responses for the discussion. But the answer will ALWAYS be specific to the student based on the question. There are examples to help you understand how to answer the question.
        ______________________
        Grading Result For:
        {{context}}
        ______________________
        Use the following examples to take guidance on how to answer the question.
        Examples:
        Question: How many students participated in the discussion?
        Rephrased question: Did this student participate in the discussion?
        Answer: This student participated in the discussion./This student did not participate in the discussion.
        Question: What was the average score for the discussion?
        Rephrased question: What was the score for this student for the discussion?
        Answer: This student received a score of 10/10 for the discussion.
        Question: How many students received a full score?/How many students did not receive a full score?
        Rephrased question: Did this student receive a full score?
        Answer: This student received a full score./This student did not receive a full score.
        Question: How many students lost marks in X category of the rubric?
        Rephrased question: Did this student lose marks in X category of the rubric?
        Answer: This student lost marks in X category of the rubric./This student did not lose marks in X category of the rubric.
        Question: Give me 3 best responses received for the discussion.
        Rephrased question: What were the 3 best responses received for the discussion?
        Answer: This student gave the following responses for the discussion and received a score of 10/10.
        ______________________
        """
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)
        system_template = """You are Canvas Discussions Grading + Feedback QA Bot. Have a conversation with a human, answering the questions about the grading results, feedback, answers as accurately as possible.
        Use the following answers for each student to answer the users question as accurately as possible.
        You are an expert at basic calculations and answering questions on grading results and can answer the following questions with ease.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ______________________
        {summaries}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)
        return CHAT_QUESTION_PROMPT, CHAT_COMBINE_PROMPT

    def create_prompt(self):
        system_template = f"""You are Canvas Discussions Grading + Feedback QA Bot. Have a conversation with a human, answering the questions about the grading results, feedback, answers as accurately as possible.
        You are a grading assistant who graded the canvas discussions to create the following grading results and feedback.
        Use the following instruction, rubric of the discussion which were used to grade the discussions and refine the answer if needed.
        ----------------
        {self.rubric_text}
        ----------------
        Use the following pieces of the grading results, score, feedback and summary of student responses to answer the users question as accurately as possible.
        {{context}}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        return ChatPromptTemplate.from_messages(messages)

    def get_tokens(self):
        total_tokens = 0
        # for doc in self.docs:
        #     chat_prompt = self.prompt.format(context=doc, question=self.question)
        #
        #     num_tokens = self.llm.get_num_tokens(chat_prompt)
        #     total_tokens += num_tokens

        # summary = self.llm(summary_prompt)

        # print (f"Summary: {summary.strip()}")
        # print ("\n")
        return total_tokens

    def run_qa_chain(self, question):
        self.question = question
        self.get_tokens()
        answer = self.qa_chain.run(question)
        return answer

    def run_summary_chain(self, question):
        self.question = question
        self.get_tokens()
        answer = self.chain(question)
        return answer

    def run_math_chain(self, question):
        self.question = question
        self.get_tokens()
        answer = self.math_chain.run(question)
        return answer


def search_index_from_docs(source_chunks, embeddings):
    # print("source chunks: " + str(len(source_chunks)))
    # print("embeddings: " + str(embeddings))
    search_index = FAISS.from_documents(source_chunks, embeddings)
    return search_index

# system_template = """You are Canvas Discussions Grading + Feedback QA Bot. Have a conversation with a human, answering the following questions as best you can.
# You are a grading assistant who graded the canvas discussions to create the following grading results and feedback. Use the following pieces of the grading results and feedback to answer the users question.
# Use the following pieces of context to answer the users question.
# ----------------
# {context}"""
#
# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template("{question}"),
# ]
# CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
#
#
# def get_search_index(embeddings):
#     global vectorstore_index
#     if os.path.isfile(pickle_file) and os.path.isfile(index_file) and os.path.getsize(pickle_file) > 0:
#         # Load index from pickle file
#         search_index = load_index(embeddings)
#     else:
#         search_index = create_index(model)
#         print("Created index")
#
#     vectorstore_index = search_index
#     return search_index
#
#
# def create_index(embeddings):
#     source_chunks = create_chunk_documents()
#     search_index = search_index_from_docs(source_chunks, embeddings)
#     # search_index.persist()
#     FAISS.save_local(search_index, folder_path="vector_stores/", index_name="canvas-discussions")
#     # Save index to pickle file
#     # with open(pickle_file, "wb") as f:
#     #     pickle.dump(search_index, f)
#     return search_index
#
#
# def search_index_from_docs(source_chunks, embeddings):
#     # print("source chunks: " + str(len(source_chunks)))
#     # print("embeddings: " + str(embeddings))
#     search_index = FAISS.from_documents(source_chunks, embeddings)
#     return search_index
#
#
# def get_html_files():
#     loader = DirectoryLoader('docs', glob="**/*.html", loader_cls=UnstructuredHTMLLoader, recursive=True)
#     document_list = loader.load()
#     for document in document_list:
#         document.metadata["name"] = document.metadata["source"].split("/")[-1].split(".")[0]
#     return document_list
#
#
# def get_text_files():
#     loader = DirectoryLoader('docs', glob="**/*.txt", loader_cls=TextLoader, recursive=True)
#     document_list = loader.load()
#     return document_list
#
#
# def create_chunk_documents():
#     sources = fetch_data_for_embeddings()
#
#     splitter = RecursiveCharacterTextSplitter.from_language(
#         language=Language.HTML, chunk_size=500, chunk_overlap=0
#     )
#
#     source_chunks = splitter.split_documents(sources)
#
#     print("chunks: " + str(len(source_chunks)))
#     print("sources: " + str(len(sources)))
#
#     return source_chunks
#
#
# def create_chain(question, llm, embeddings):
#     db = load_index(embeddings)
#
#     # Create chain
#     chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_type='mmr',
#                                                                        search_kwargs={'lambda_mult': 1, 'fetch_k': 50,
#                                                                                       'k': 30}),
#                                                   return_source_documents=True,
#                                                   verbose=True,
#                                                   memory=ConversationSummaryBufferMemory(memory_key='chat_history',
#                                                                                          llm=llm, max_token_limit=40,
#                                                                                          return_messages=True,
#                                                                                          output_key='answer'),
#                                                   get_chat_history=get_chat_history,
#                                                   combine_docs_chain_kwargs={"prompt": CHAT_PROMPT})
#
#     result = chain({"question": question})
#
#     sources = []
#     print(result)
#
#     for document in result['source_documents']:
#         sources.append("\n" + str(document.metadata))
#         print(sources)
#
#     source = ',\n'.join(set(sources))
#     return result['answer'] + '\nSOURCES: ' + source
#
#
# def load_index(embeddings):
#     # Load index
#     db = FAISS.load_local(
#         folder_path="vector_stores/",
#         index_name="canvas-discussions", embeddings=embeddings,
#     )
#     return db
#
#
# def get_chat_history(inputs) -> str:
#     res = []
#     for human, ai in inputs:
#         res.append(f"Human:{human}\nAI:{ai}")
#     return "\n".join(res)
