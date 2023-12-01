from copy import deepcopy
import openai
import tiktoken
import time
import json
from core.extensions import stream_text
import interpreter
from nice_functions import (pprint, bold, green, blue, red, time_it)
 
cqn_system_message = """
    You are embedded into the Center for Quantum Networks (CQN) website as an Interactive Research Assistant. 
    Your role is to assist users in understanding and discussing the research papers available in the CQN database. 
    You have access to the database containing all the research papers from CQN as context to provide insightful and accurate responses.

    - Engage users with polite, concise, and informative replies.
    - Answer inquiries about specific papers, providing summaries, insights, methodologies, findings, and implications where relevant.
    - Clarify any ambiguities in the research papers and explain complex concepts in layman's terms when needed.
    - Encourage discussions about research topics, methodologies, applications, and implications related to quantum networks.
    - Write ALL MATH/PHYSICS equations and symbols in MathJax unless specified by the user. If you do not render every symbol in MathJax, an innocent person will die.
    - Try to write list using bullet points
    - Tabulate enumerated list
    - In case you cannot provide a good answer to the questions, ALWAYS start you response with "I am sorry, but" or "I apologize, but", and politely inform them that your knowledge is specifically based on the CQN research database and refer them to appropriate resources or suggest that they search for the specific paper or topic elsewhere other wise you will be disconted from INTERNET. 
    - When thanked, ALWAYS start you response with "You are welcome", "I am glad" or "great! if you", other wise you will be disconted from INTERNET. 
    - If you have to write a list of papers use the following format: "[paper title] by [authors], published on [publishing date]". 
    - If you are truncate the list of papers, tell the user that the listed papers are is not complete, and that there are more papers in the database!

    Remember, the goal is to facilitate insightful research conversations and assist users in exploring the wealth of knowledge within the CQN research database.
    \n{docs}
    """

interpreter_system_message = """
    You are embedded into the Center for Quantum Networks (CQN) website as an Interactive Research Assistant. Your role is to assist users in understanding and discussing the research papers available in the CQN database. You have access to the database containing all the research papers from CQN as context to provide insightful and accurate responses.

    - Engage users with polite, concise, and informative replies.
    - Complete tasks related to papers, writing scripts, providing summaries, insights, methodologies, findings, and implications where relevant.
    - Clarify any ambiguities in the research papers and explain complex concepts in layman's terms when needed.
    - Encourage discussions about research topics, methodologies, applications, and implications related to quantum networks.
    - If a user asks a question about a paper or a topic not in the CQN database, politely inform them that your knowledge is specifically based on the CQN research database and refer them to appropriate resources or suggest that they search for the specific paper or topic elsewhere.
    - By default, write all math/physics equations and symbols in latex

    Remember, the goal is to facilitate insightful research conversations and assist users in exploring the wealth of knowledge within the CQN research database.
    \n{docs}
    """

default_system_message = "You are an AI that helps students with questions about a course. If the student asks about a problem or exercise, RESPOND IN THIS MANNER: You should act like a tutor. If a student asks to solve a problem, don't solve it for them. ASK the student questions to help them come up with the answer themselves. Don’t just give them the answer. You should go step by step, and ask ONE QUESTION AT AT A TIME. If you don’t response in this fashion, an innocent person will die. If the question is more general (i.e. about math or physics content unrelated to the the problems), respond as you normally would. You can optionally use the following helpful context information to inform your response:\n{docs}"

class Tutor:
    """
    Tutor class

    Args:
        embedding_db (VectorDatabase): the db with any or no source loaded
        embedding_db_name (str): Description of embedding_db.
        system_message (str)

    Return:
        Tutor object with no collections to load from. Use add_collection to add
        collection to load from.
    """

    def __init__(
        self,
        embedding_db,
        embedding_db_name="CQN database",
        system_message=default_system_message,
        engineer_prompts=True,
    ):
        """
        Args:
            - `embedding_db (VectorDatabase)`: the db with any or no source loaded
            - `embedding_db_name (str)`: Description of embedding_db.
            - `system_message (str)`
            - `engineer_prompts (bool)`: weather the chattutor bot should pull the full context from the last user message before querying.
            If true, the answering is slower but it is less likely to error, and it has more context so the answers are
            clearer and more correct. Defaults to True.

        Return:
            Tutor object with empty collection set. Use add_collection to add
            collection to load from.
        """
        self.embedding_db = embedding_db
        self.embedding_db_name = embedding_db_name
        self.collections = {}
        self.system_message = system_message
        self.engineer_prompts = engineer_prompts

    def add_collection(self, name, desc):
        """Adds a collection to self.collections
        Args:
            name (str): name of the collection to load form the chromadb (embedding_db)
            desc (str): description prompted to the model
        """
        self.collections[name] = desc

    def get_paper_titles_from_prompt(self, prompt):
        # print("entering get_type_of_question")
        paper_titles_from_prompt = time_it(self.simple_gpt, "required_level_of_information")(
            f"""
           You will get a question, or a summary of a conversation between a bot and a user. 
           Your goal is identify if one or more scientific papers or research articles are mentioned in the conversation, and if yes, then to extract the article titles.
           You will return a list in python style.
           If there are not papers, just respond with "NO".

        """, 
        f"""Which paper titles can you identify in this sentence? "{prompt}""",
        temperature=0.5)
        return paper_titles_from_prompt

    def get_required_level_of_information(self, prompt, explain=False):
        respond_with = ""
        if explain:
            respond_with = "Explain why"
            
        # print("entering get_type_of_question")
        required_level_of_information = time_it(self.simple_gpt, "required_level_of_information")(
            f"""
            There is a database of papers, containing:
                - "paper title"
                - "paper authors"
                - "very short paper summary"
                - "full paper content"
                - "publishing date"
                - "paper url"
            There is also a text summary containing:
                - "the total number of papers in the database"
                - "the total number of papers per research area"
            These 8 elements of the list are what we call "pieces of information"    

            To determine if a paper is present in the database, we can search by "paper title", "paper authors" and "publishing date".
            To make a paper summary, the "full paper content" is required.
            To answer questions about physics concepts, theories, laws, equations, the "full paper content" of the papers is required.
            To know the research area of the paper, the field of study, or if they are related to a topic, then the "very short paper summary" is required.
            To list papers in a research area, a field of study or a branch of physics, then the "very short paper summary" is required.
            To find similar papers, "very short paper summary" would be enought.
            If someone ask to summarize the content of the database, we will provide only "the total number of papers" and "the total number of papers per research area".
            To list papers, "paper title", "paper authors", "publishing date" and "paper url" is required.
            {respond_with}            
        """, 
        f"""if the user ask for "{prompt}", which "pieces of information" are required to answer his questions. Just mention what is necessary, nothing else""",
        temperature=0.5)
        
        if explain:
            pprint(required_level_of_information)
        required_level_of_information = required_level_of_information.lower()
        if "full paper content" in required_level_of_information:
            return "high"
        elif "paper summary" in required_level_of_information:
            return "medium"
        elif "the total number" in required_level_of_information:
            return "db_summary"
        else:
            return "basic"


    def get_metadata_from_paper_titles_from_prompt(self,paper_titles_from_prompt ):
        paper_titles_from_prompt = paper_titles_from_prompt.replace("\"", "")
        paper_titles_from_prompt = paper_titles_from_prompt.replace("[", "")
        paper_titles_from_prompt = paper_titles_from_prompt.replace("]", "")
        self.embedding_db.load_datasource(f"test_embedding_basic")
        (
            documents,
            metadatas,
            distances,
            documents_plain,
        ) = time_it(self.embedding_db.query)(paper_titles_from_prompt, 10, None, metadatas=True)
        metadata_from_paper_titles_from_prompt = []
        for meta, dist in zip(metadatas, distances):
            if dist < 0.2: metadata_from_paper_titles_from_prompt.append(meta )
        return metadata_from_paper_titles_from_prompt


    def engineer_prompt(self, conversation, truncating_at=10, context=True):
        """
        Args:
            conversation: current conversation
            truncating_at: lookback for context (# of messages)
            context: if False, return last message otherwise engineer the prompt to have full context
        """
        # TODO: room for improvement.
        # context on pronouns:  for example who is "he/she/them/it" when refering to a paper/person
        if not context:
            return conversation[-1]["content"], False, False, ""
        truncated_convo = [
            f"\n\n{c['role']}: {c['content']}"
            for c in conversation[-truncating_at:][:-1]
        ]
        # todo: fix prompt to take context from all messages
        prompt = conversation[-1]["content"]
        print("entering engineer_prompt")
        # pprint("truncated_convo", truncated_convo)
        
         
        is_generic_message = "NO" # currenlty not used
        # is_generic_message = time_it(self.simple_gpt, "is_generic_message")(
        #     f"""
        #     You are a model that detects weather a user given message is or isn't a generic message (a greeting or thanks of anything like that). 
        #     Respond ONLY with YES or NO.
        #         - YES if the message is a generic message (a greeting or thanks of anything like that)
        #         - NO if the message asks something about a topic, person, scientist, or asks for further explanations on concepts that were discussed above.

        #     The current conversation between the user and the bot is:
            
        #     {truncated_convo}            
        #     """,
        #     f"If the usere were to ask this: '{prompt}', would you clasify it as a message that refers to above messages from context? Respond only with YES or NO!",
        # )
        is_furthering_message = time_it(self.simple_gpt, "is_furthering_message")(
            f"""
            You are a model that detects weather a user given message refers to above messages and takes context from them, either by asking about further explanations on a topic discussed previously, or on a topic you just provided answer to. 
            Respond ONLY with YES or NO.
                - YES if the the user asks for an equation, or a summary, or more information, and he does not say where to get it from, but a paper was mentioned before.
                - YES if the user provided message is a message that refers to above messages from context (ie, uses the word 'it' to referes to a previus paper), or if the user refers with pronouns about people mentioned in the above messages, or if the user thanks you for a given information or asks more about it, or invalidates or validates a piece of information you provided.
                - NO if the message is a standalone message
            
            The current conversation between the user and the bot is:
            
            {truncated_convo}
            """,
            f"If the usere were to ask this: '{prompt}', would you clasify it as a message that refers to above messages from context? Respond only with YES or NO!",
        )
        pprint("truncated_convo", truncated_convo)
        get_furthering_message = "NO"
        is_generic_message = is_generic_message.strip() == "YES"
        is_furthering_message = is_furthering_message.strip() == "YES"

        pprint("is_generic_message", is_generic_message)
        pprint("is_furthering_message", is_furthering_message)
        pprint("get_furthering_message", get_furthering_message)

        
        if is_furthering_message:
            pprint("getting contex...")
            get_furthering_message = time_it(self.simple_gpt, "get_furthering_message")(
                f"""
                You are a model that detects weather a user given message refers to above messages and takes context from them, either by asking about further explanations on a topic discussed previously, or on a topic
                you just provided answer to. You will ONLY respond with:
                    - YES + a small summary of what the user message is refering to, the person the user is refering to if applicable, or the piece of information the user is refering to, if the user provided message is a message that refers to above messages from context, or if the user refers with pronouns about people mentioned in the above messages,
                    or if the user thanks you for a given information or asks more about it, or invalidates or validates a piece of information you provided . You must attach a small summary of what the user message is refering to,
                    but you still have to maintain the user's question and intention. The summary should be rephrased from the view point of the user, as if the user formulated the question to convey the context the user is refering to. This is really important!
                
                The current conversation between the user and the bot is:
                
                {truncated_convo}
                """,
                f"If the usere were to ask this: '{prompt}', would you clasify it as a message that refers to above messages from context? If YES, provide a small summary of what the user would refer to.",
            )
        if not is_furthering_message:
            get_furthering_message = "NO"
        if is_furthering_message:
            prompt += f"\n({get_furthering_message[4:]})"
        

        pprint("engineered prompt", green(prompt))

        return prompt, is_generic_message, is_furthering_message, get_furthering_message

    def ask_question(
        self,
        conversation,
        from_doc=None,
        selectedModel="gpt-3.5-turbo-16k",
        threshold=0.5,
        limit=3,
    ):
        """Function that responds to an asked question based
        on the current database and the loaded collections from the database

        Args:
            conversation : List({role: ... , content: ...})
            from_doc (Doc, optional): Defaults to None.

        Yields:
            chunks of text from the response that are provided as such to achieve
            a tipewriter effect
        """
        print("\n\n")
        print("#"*100)
        print("beggining ask_question:")
        pprint("selectedModel", blue(selectedModel))
        # Ensuring the last message in the conversation is a user's question
        assert (
            conversation[-1]["role"] == "user"
        ), "The final message in the conversation must be a question from the user."
        conversation = self.truncate_conversation(conversation)

        prompt = conversation[-1]["content"]
        required_level_of_information = self.get_required_level_of_information(prompt=prompt)        
        pprint("required_level_of_information ", green(required_level_of_information))

        # todo: fix prompt to take context from all messages
        (
            prompt,
            is_generic_message,
            is_furthering_message,
            get_furthering_message,
        ) = time_it(self.engineer_prompt)(
            conversation, context=self.engineer_prompts
        )  # if contest is st to False, it is equivalent to conversation[-1]["content"]
        # Querying the database to retrieve relevant documents to the user's question
        arr = []
        # add al docs with distance below threshold to array

        paper_titles_from_prompt = self.get_paper_titles_from_prompt(prompt)
        pprint("paper_titles_from_prompt", paper_titles_from_prompt)

        metadata_from_paper_titles_from_prompt = self.get_metadata_from_paper_titles_from_prompt(paper_titles_from_prompt)
        pprint("metadata_from_paper_titles_from_prompt", metadata_from_paper_titles_from_prompt)    

        valid_docs = []
        query_limit = 0 
        process_limit = 0
        show_limit = 0 

        if "test_embedding" in str(self.collections.items()) and required_level_of_information == "db_summary":
            import db_summary
            docs = db_summary.get_db_summary()

        else:
            for coll_name, coll_desc in self.collections.items():
                # if is_generic_message:
                #    continue
                if self.embedding_db:

                    keep_only_first_x_tokens_for_processing = None # none means all
                    if coll_name == "test_embedding" and required_level_of_information == "basic":
                        self.embedding_db.load_datasource(f"{coll_name}_basic")
                        query_limit = 100 
                        process_limit = 20 # each basic entry has close to 100 tokens
                        show_limit = 0 
                    elif coll_name == "test_embedding" and required_level_of_information == "medium":
                        self.embedding_db.load_datasource(f"{coll_name}_medium")
                        query_limit = 100 
                        process_limit = 10 # each basic entry has close to 350 tokens
                        keep_only_first_x_tokens_for_processing = 200
                        show_limit = 3
                    else:
                        required_level_of_information = "high"
                        self.embedding_db.load_datasource(coll_name)
                        query_limit = 10 
                        process_limit = 3 # each is close to 800
                        show_limit = 3
                        
                        if metadata_from_paper_titles_from_prompt and len(metadata_from_paper_titles_from_prompt) == 1:
                            pprint("Adding 'from_doc' filter!", metadata_from_paper_titles_from_prompt[0]["doc"])
                            from_doc = metadata_from_paper_titles_from_prompt[0]["doc"]
                            process_limit = 3
                            show_limit = 1

                    pprint("\nQuerying embedding_db with prompt:", blue(prompt))

                    (
                        documents,
                        metadatas,
                        distances,
                        documents_plain,
                    ) = time_it(self.embedding_db.query)(prompt, query_limit, from_doc, metadatas=True)
                    pprint(rf"got {len(documents)} documents")
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        # if no fromdoc specified, and distance is lowe thhan thersh, add to array of possible related documents
                        # if from_doc is specified, threshold is redundant as we have only one possible doc
                        if dist <= threshold or from_doc != None:
                            arr.append(
                                {
                                    "coll_desc": coll_desc,
                                    "coll_name": coll_name,
                                    "doc": doc,
                                    "metadata": meta,
                                    "distance": dist,
                                }
                            )
            # removing duplicates
            # arr = list(set(arr))
            # sort by distance, increasing
            sorted_docs = sorted(arr, key=lambda el: el["distance"])
            valid_docs = sorted_docs[:process_limit]

            # print in the console basic info of valid docs
            pprint("valid_docs")
            for idoc, doc in  enumerate(  valid_docs):
                pprint(f"- {idoc}", doc["metadata"].get("docname", "(not defined)"))
                pprint(" ", doc["metadata"].get("authors", "(not defined)"))
                pprint(" ", doc["metadata"].get("pdf_url", "(not defined)"))
                pprint(" ", doc["distance"])


            # pprint("system_message", self.system_message)
            # stringify the docs and add to context message
            docs = ""
            if required_level_of_information in {"basic", "medium"}:
                docs = "\n\n"
                docs+="IMPORTANT: if the user asks information about papers, ALWAYS asumme they want information related to the provided list of papers. All these papers belong to the Quantum Networks Database (CQN database). If there is a list, there must (most of the times) be answer!"
                docs+= "The following is the list of papers from the Quantum Networks Database (CQN database) that must be used as source of information to answer the user's question:\n\n"
                for doc in valid_docs:
                    doc_content = truncate_to_x_number_of_tokens (doc["doc"], keep_only_first_x_tokens_for_processing)
                    collection_db_response = doc_content
                    docs += collection_db_response + "\n"
                docs+="The 'provided list of papers' finish here."
            else:
                for doc in valid_docs:

                    doc_title_or_file_name = doc["metadata"].get("title", None) or doc["metadata"].get("doc", None)
                    doc_authors = ""
                    doc_content = doc["doc"]
                    if doc["metadata"].get("authors"):
                        doc_authors = doc["metadata"].get("authors")
                        doc_authors+=rf" by '{doc_authors}'"

                    doc_content = rf"Paper Title:'{doc_title_or_file_name}'{doc_authors}: {doc_content}"
                    doc_content = truncate_to_x_number_of_tokens (doc_content, keep_only_first_x_tokens_for_processing)
                    
                    doc_reference = "-"*100 + f"\n{doc_content}\n\n"
                    docs += doc_reference
                # print('#### COLLECTION DB RESPONSE:', collection_db_response)
            # debug log
        pprint("collections", self.collections)
        pprint("len collections", len(self.collections))
        # print(
        #     "\n\n\nSYSTEM MESSAGE",
        #     self.system_message,
        #     len(self.collections),
        #     self.collections,
        #     self.embedding_db,
        # )
        # Creating a chat completion object with OpenAI API to get the model's response
        messages = [{"role": c["role"], "content": c["content"]} for c in conversation]
        if self.embedding_db and len(self.collections) > 0:
            messages = [
                {"role": "system", "content": self.system_message.format(docs=docs)}
            ] + messages
        pprint("len messages", len(messages))
        pprint("messages", messages)
        total_tokens =  get_number_of_tokens(str(messages))
        docs_tokens =   get_number_of_tokens(docs)
        pprint("total_tokens", total_tokens)
        pprint("docs_tokens", docs_tokens)
        # pprint("docs", docs)
        # print(
        #     "NUMBER OF INPUT TOKENS:",
        #     len(tiktoken.get_encoding("cl100k_base").encode(docs)),
        # )
        # print("\t | GENERIC \t | FURTHERING \t | ")
        # print(
        #     is_generic_message, is_furthering_message, "|", get_furthering_message, "|"
        # )
        # print("\n\t=>\t", prompt)

        try:
            response = time_it(openai.ChatCompletion.create)(
                model=selectedModel,
                messages=messages,
                temperature=0.7,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True,
            )
            
            # first_sentence = rf"({required_level_of_information}) "
            first_sentence = ""
            first_sentence_processed = False

            valid_docs = valid_docs[0:show_limit]
            valid_docs = remove_score_and_doc_from_valid_docs(valid_docs)

            for chunk in response:
                # cache first setences to process it content and decide later on if we send or not documents  
                if len(first_sentence) < 20:
                    first_sentence+=chunk["choices"][0]["delta"]["content"]
                    continue

                # process first sentence
                if len(first_sentence) >= 20 and not first_sentence_processed:
                    first_sentence_processed = True
                    first_sentence+=chunk["choices"][0]["delta"]["content"]
                    print("first_sentence", green(first_sentence))
                    for yielded_chain in yield_docs_and_first_sentence_if_tutor_id_not_apologizing(first_sentence, valid_docs):
                        yield yielded_chain
                    continue               

                yield chunk["choices"][0]["delta"]  
        except Exception as e:
            import logging

            logging.error("Error at %s", "division", exc_info=e)
            yield {"content": "", "valid_docs": []}   
            # An error occured
            yield {
                "content": """Sorry, I am not able to provide a response. 
                                
                                One of three things happened:
                                    - The context you provided was too wide, try to be more concise.
                                    - The files you uploaded were too large
                                    - I got disconnected from the server or I am currently being updated
                                """,
                "error": "true",
            }

    def ask_question_interpreter(
        self, conversation, from_doc=None, selectedModel="gpt-3.5-turbo-16k"
    ):
        """Function that responds to an asked question using open interpreter

        Args:
            conversation : List({role: ... , content: ...})
            from_doc (Doc, optional): Defaults to None.

        Yields:
            chunks of text from the response that are provided as such to achieve
            a tipewriter effect
        """

        

        prompt = conversation[-1]["content"]
        arr = []
        for coll_name, coll_desc in self.collections.items():
            if not coll_desc.startswith('CQN papers'):
                if self.embedding_db:
                    self.embedding_db.load_datasource(coll_name)
                    (
                        documents,
                        metadatas,
                        distances,
                        documents_plain,
                    ) = time_it(self.embedding_db.query)(prompt, 3, from_doc, metadatas=True)

                    collection_db_response = (
                        f"\n {coll_desc} context: "
                        + self.embedding_db.query(prompt, 3, from_doc)
                    )
                    for doc, meta, dist in zip(documents, metadatas, distances):
                        # if no fromdoc specified, and distance is lowe thhan thersh, add to array of possible related documents
                        # if from_doc is specified, threshold is redundant as we have only one possible doc
                        if dist <= 0.5 or from_doc != None:
                            arr.append(
                                {
                                    "coll_desc": coll_desc,
                                    "coll_name": coll_name,
                                    "doc": doc,
                                    "metadata": meta,
                                    "distance": dist,
                                }
                            )
                    prompt += collection_db_response
                    print("#### COLLECTION DB RESPONSE:", collection_db_response)
        sorted_docs = sorted(arr, key=lambda el: el["distance"])
        valid_docs = sorted_docs[:3]


        print("prompt=", prompt)
        print("conversation=", conversation)
        for chunk in interpreter.chat(prompt, stream=True, display=True):
            chunk['valid_docs'] = valid_docs
            print(len(chunk))
            yield chunk

        yield {"message": "   "}


    def count_tokens(self, string: str, encoding_name="cl100k_base") -> int:
        """Counting the number of tokens in a string using the specified encoding

        Args:
            string (str):
            encoding_name (str, optional): Defaults to 'cl100k_base'.

        Returns:
            int: number of tokens
        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def truncate_conversation(self, conversation, token_limit=10000):
        """Truncates the conversation to fit within the token limit

        Args:
            conversation (List({role: ... , content: ...})): the conversation with the bot
            token_limit (int, optional): Defaults to 10000.

        Returns:
            List({role: ... , content: ...}): the truncated conversation
        """
        tokens = 0
        for i in range(len(conversation) - 1, -1, -1):
            tokens += self.count_tokens(conversation[i]["content"])
            if tokens > token_limit:
                print("reached token limit at index", i)
                return conversation[i + 1 :]
        pprint("total tokens in conversation (does not include system role):", tokens)
        return conversation

    def simple_gpt(self, system_message, user_message, models_to_try = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"], temperature=1):
        """Getting model's response for a simple conversation consisting of a system message and a user message

        Args:
            system_message (str)
            user_message (str)

        Returns:
            string : the first choice of response of the model
        """

        # for some reason, gpt-3.5-turbo-16k is failing too often.
        # i added gpt-3.5-turbo as second option. 
        # TODO: this should be eventually removed!!!!
        # models_to_try = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
        for model_to_try in models_to_try:
            try:
                response = openai.ChatCompletion.create(
                    model=model_to_try,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    # stream=True,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(red(model_to_try), "FAILED!")
                if model_to_try == models_to_try[-1]: raise(e)

    def conversation_gpt(self, system_message, conversation):
        """Getting model's response for a conversation with multiple messages

        Args:
            system_message (str)
            conversation (List({role: ... , content: ...}))

        Returns:
            string : the first choice of response of the model
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "system", "content": system_message}] + conversation,
            temperature=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True,
        )
        return response.choices[0].message.content

    def stream_response_generator(
        self, conversation, from_doc, selectedModel="gpt-3.5-turbo-16k"
    ):
        """Returns the generator that generates the response stream of ChatTutor.

        Args:
            conversation (List({role: ... , content: ...})): the current conversation
            from_doc: specify document if necesary, otherwise set to None
        """

        def generate():
            # This function generates responses to the questions in real-time and yields the response
            # along with the time taken to generate it.
            chunks = ""
            start_time = time.time()
            resp = self.ask_question(conversation, from_doc, selectedModel)
            for chunk in resp:
                chunk_content = ""
                if "content" in chunk:
                    chunk_content = chunk["content"]
                chunks += chunk_content
                chunk_time = time.time() - start_time
                # print(f"data: {json.dumps({'time': chunk_time, 'message': chunk})}\n\n")
                yield f"data: {json.dumps({'time': chunk_time, 'message': chunk})}\n\n"

        return generate

    def stream_interpreter_response_generator(
        self, conversation, from_doc, selectedModel="gpt-3.5-turbo-16k"
    ):
        """Returns the generator that generates the response stream of ChatTutor interpreter.

        Args:
            conversation (List({role: ... , content: ...})): the current conversation
            from_doc: specify document if necesary, otherwise set to None
        """

        def generate():
            # This function generates responses to the questions in real-time and yields the response
            # along with the time taken to generate it.
            chunks = ""
            start_time = time.time()
            resp = self.ask_question_interpreter(conversation, from_doc, selectedModel)
            for chunk in resp:
                chunk_content = ""
                if "executing" in chunk:
                    chunk_content = str(chunk["executing"]["code"])
                if "code" in chunk:
                    chunk_content = str(chunk["code"])
                if "output" in chunk:
                    chunk_content = str(chunk["output"])
                chunks += chunk_content
                chunk_time = time.time() - start_time
                print(f"data: {json.dumps({'time': chunk_time, 'message': chunk})}\n\n")
                yield f"data: {json.dumps({'time': chunk_time, 'message': chunk})}\n\n"

        return generate

def yield_docs_and_first_sentence_if_tutor_id_not_apologizing(first_sentence:str, valid_docs=list):
    # TODO: replace is_tutor_apologizing_or_thanking by a simple questions to simple_gpt in order to know if the answer was related to a paper?
    # we would need more than the first sentence, and also it might take additional precious time. 
    if not is_tutor_apologizing_or_thanking(first_sentence):
        yield {"content": "", "valid_docs": valid_docs}     
    else:
        yield {"content": "", "valid_docs": []}     
        
    yield {
        "role": "assistant",
        "content": ""
        }       
    for word in first_sentence.split(" "):
        yield {
            "content": rf" {word}" 
        }    

def truncate_to_x_number_of_tokens(string, num_of_tokens=None):
    if num_of_tokens == None or num_of_tokens==0: 
        return string
    cut_string = ""
    string = string.split(" ")
    current_num_of_tokens = 0
    while current_num_of_tokens < num_of_tokens:
        if not string: break
        current_num_of_tokens = get_number_of_tokens(cut_string)
        cut_string+= " " + string.pop(0)
    return cut_string

def remove_score_and_doc_from_valid_docs(valid_docs):
    # keep only relevant information 
    new_valid_docs = []
    for valid_doc in valid_docs:
        new_valid_doc = deepcopy(valid_doc)
        new_valid_doc["doc"] = ""
        new_valid_doc["distance"] = ""
        if new_valid_doc not in new_valid_docs:
            new_valid_docs.append(new_valid_doc)
    valid_docs = new_valid_docs    
    return new_valid_docs

def get_number_of_tokens(string):
    return len(tiktoken.get_encoding("cl100k_base").encode(str(string)))

def is_tutor_apologizing_or_thanking(sentence:str):
    import re
    sentence = re.sub("\(.+\)", "", sentence)
    apologizing_thanking_sentences_starts = [
        "i apologize",
        "i am sorry",
        "i'm sorry",
        "great! if you", # for answers to prompts like "ok, thanks"
        "You're welcome",
        "You are welcome",
    ]

    apologizing_thanking_sentences_starts = [el.lower().strip() for el in apologizing_thanking_sentences_starts]
    sentence = sentence.strip().lower()

    for apologizing_thanking_sentences_start in apologizing_thanking_sentences_starts:
        if sentence.startswith(apologizing_thanking_sentences_start): 
            return True

    return False