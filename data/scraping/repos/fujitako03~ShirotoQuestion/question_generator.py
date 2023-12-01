from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from openai import ChatCompletion

class QuestionGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, docs):
        topic = self.get_topics(docs)
        issues = self.generate_topic_issues(topic["output_text"])
        missing_issues = self.find_missing_issues(issues, docs)
        question = self.generate_question(missing_issues)

        return question

    def get_topics(self, docs):
        # トピックの抽出
        query = """
        この文章のトピックを一言で表すと何と言えますか
        """
        chain = load_qa_with_sources_chain(self.llm, chain_type="map_reduce")
        topic_text = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        self.topic = topic_text
        return self.topic
    
    def generate_topic_issues(self, topic):
        # 論点の抽出
        template = """
        以下のトピックについて議論をする際に重要な論点を網羅的に羅列してください。
        {topic_text}
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic_text"]
        )
        prompt_text = prompt.format(topic_text=[topic])

        # 
        response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt_text}
            ]
        )
        self.issues = response["choices"][0]["message"]["content"]

        #ChatGPTの回答を出力
        return self.issues
    
    def find_missing_issues(self, issues, docs):
        template = """
        以下の重要な論点のうち、ここまでの議論で取り上げられていないものはどれですか
        {issue_text}
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["issue_text"]
        )
        prompt_text = prompt.format(issue_text=issues)

        chain = load_qa_with_sources_chain(self.llm, chain_type="map_reduce")
        missing_issues = chain({"input_documents": docs, "question": prompt_text}, return_only_outputs=True)
        
        self.missing_issues = missing_issues
        return self.missing_issues
    
    def generate_question(self, missing_issues):
        # プロンプトの生成
        template = """
        あなたは親切なアシスタントです。質問したい内容についてより深く知るための質問を5つ生成してください。
        質問は全て日本語で簡潔にしてください。

        # 質問したい内容
        {missing_issues}

        # フォーマット
        大変興味深い発表ありがとうございます。素人質問で恐縮ですが、
        1. [question1]
        2. [question2]
        3. [question3]
        4. [question4]
        5. [question5]
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["missing_issues"]
        )
        prompt_text = prompt.format(
            missing_issues=missing_issues)

        # 質問の生成
        response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt_text}
            ]
        )

        self.question = response["choices"][0]["message"]["content"]

        #ChatGPTの回答を出力
        return self.question