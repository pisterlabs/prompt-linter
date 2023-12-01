from .rag_v3 import RAGVer3
import openai
import guidance


class RAGVer5(RAGVer3): 
    """
    improving the reader 📖:
    judiciously determine if the query is relevant to the paper or not.
    Uses Chain-of-Thought for reasoning, Microsoft's guidance for determinism.
    """ 

    def __call__(self, query: str, alpha: float = 0.4, k: int = 3) -> str:
        guidance.llm = guidance.llms.OpenAI("text-davinci-003")
        # define the guidance program
        chitchat_detection = guidance(
        """
        title of the paper: {{title}}
        ---
        You are a Question Answering Machine. Your role is to answer the user's question relevant to the query.
        Your role is also to detect if the user's question is irrelevant to the paper. If the user says, for example,
        "hi, how are you doing?", "what is your name?", or "what is the weather today?", then the user's question is irrelevant.
        
        Now, judiciously determine if the following Query is relevant to the paper or not.
    
        Answer in the following format:
        Query: The question to be answered
        Reason: is the Query relevant to the paper? Show your reasoning. Explain why or why not.
        Final Answer: Either conclude with Yes (the Query is relevant) or No (the Query is irrelevant)
        ---
        Query: {{query}}
        Reasoning: {{gen "reasoning" stop="\\nF"}}
        Final Answer:{{#select "answer"}}Yes{{or}}No{{/select}}""")
        out = chitchat_detection(
            title=self.openai_paper['title'],
            query=query,
        )
        answer = out['answer'].strip()
        # save your resources - don't answer if the question is irrelevant
        if answer == 'No':
            answer = "I'm afraid I can't answer your question because: "
            answer += f"{out['reasoning'].split('.')[0]}"
            return answer
        # if the question is relevant, proceed to answer
        results: list[tuple[str, float]] = super().__call__(query, alpha, k)
        excerpts = [res[0] for res in results]
        excerpts = '\n'.join([f'[{i}]. \"{excerpt}\"' for i, excerpt in enumerate(excerpts, start=1)])
        # first, check if the query is answerable 
        # proceed to answer
        prompt = f"""
        user query:
        {query}
        
        title of the paper:
        {self.openai_paper['title']}
        
        excerpts: 
        {excerpts}
        ---
        given the excerpts from the paper above, answer the user query.
        In your answer, make sure to cite the excerpts by its number wherever appropriate.
        Note, however, that the excerpts may not be relevant to the user query.
        """
        # uses gpt-3.5-turbo 
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=[{"role": "user", "content": prompt}])
        answer = chat_completion.choices[0].message.content
        answer += f"\n--- EXCERPTS ---\n{excerpts}"
        return answer