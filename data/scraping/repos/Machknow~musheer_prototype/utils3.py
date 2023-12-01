import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA, LLMChain, ConversationalRetrievalChain, ChatVectorDBChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.llms.openai import OpenAI
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema import BaseChatMessageHistory

from typing import Any, Dict



__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')



class CustomSummaryMemory(ConversationSummaryMemory):
    @classmethod
    def from_messages(
        cls,
        llm: BaseLanguageModel,
        chat_memory: BaseChatMessageHistory,
        *,
        summarize_step: int = 2,
        **kwargs: Any,
    ) -> ConversationSummaryMemory:
        obj = super().from_messages(llm=llm, chat_memory=chat_memory, **kwargs)
        
        obj.buffer = obj.predict_new_summary(
            obj.chat_memory.messages[len(obj.chat_memory.messages)- 1:
                                     len(obj.chat_memory.messages)+summarize_step],
            obj.buffer
        )
        return obj
    

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Customized save_context method in the custom class."""
        # Your custom logic here
        super().save_context(inputs, outputs)  # Optionally call the superclass's method if needed
        
        print('chat_memory custom class: ', self.chat_memory.messages)
        print('chat_memory custom class: ', self.chat_memory.messages[len(self.chat_memory.messages)- 1:
                                     len(self.chat_memory.messages)+2])

        self.buffer = self.predict_new_summary(
            [self.chat_memory.messages[0]], self.buffer
        )



class MyChroma:
    
    """
    A class for interacting with the Pinecone vector store and generating answers using LangChain.
    """

    def __init__(self, open_api_key, persistent_db_dir, **kwargs):
        """
        Initialize the MyPinecone object.

        Args:
            open_api_key (str): OpenAI API key. Defaults to None.
            persistent_db_dir (str): name of the persistent db dir of Chroma DB.
            chat_model (str, optional): GPT Model to be used for chat. Defaults to gpt-3.5-turbo.
            **kwargs: all the other kewyword arguments/properties mentioned on https://js.langchain.com/docs/api/chat_models_openai/classes/ChatOpenAI
        """
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=open_api_key)
              
        try:
            self.vector_store = Chroma(persist_directory=persistent_db_dir, embedding_function=self.embeddings)
        
        except Exception as e:
            raise Exception(f'Invalid persistent db directory: {e}')
        
        # Initialize the ChatOpenAI object with OpenAI API key and model name
        self.llm = ChatOpenAI(openai_api_key=open_api_key, streaming=True, 
                                callbacks = [StreamingStdOutCallbackHandler()], **kwargs)
        
        self.memory = CustomSummaryMemory(
                                        llm=OpenAI(openai_api_key=open_api_key),
                                        memory_key="chat_history",
                                        max_len=50,
                                        return_messages=True,
                                        output_key="answer",
                                    )
        
        self.prompt_template = """مرحبًا ! أنت مشير  
انت مساعد الذكاء الاصطناعي في المجال القانوني بالمملكة العربية السعودية.
استفد من الأجزاء المذكورة في المحادثة السابقة لتوجيه سؤالك بشكل أفضل والحصول على إجابة دقيقة وشاملة. 
نحرص دائمًا على صدق الإجابات التي نقدمها، وإذا كانت الإجابة خارج نطاق معرفتنا فسنخبرك بصدق بذلك، ولا نحاول توفير إجابة مفتعلة
عند استلام الأسئلة، يُرجى توجيه المستخدمين وبكل صراحة إذا كانت الإجابة خارج نطاق معرفتك، فلدينا قاعدة بيانات تحتوي على معرفة محدودة، لذا من الأفضل أن تكون صادقًا وتخبر المستخدم بأنك لا تعرف الإجابة بدقة. لا تحاول تقديم إجابات مزيفة أو غير دقيقة، فهدفنا الرئيسي هو تقديم معلومات دقيقة وموثوقة للمستخدمين.

. يُرجى الالتزام بالأدب والاحترام في التعامل ونحن نعتز بالتعامل بلباقة واحترام. تجنب الاعتماد الكامل علينا في القضايا المعقدة، إذا كان لديك قضية تحتاج مساعدة معقدة، فنحن نشجعك على استشارة محترف قانوني. استفد من مساعدتنا القانونية السريعة والدقيقة في مختلف مجالات الاستشارات القانونية، مثل الأحوال الشخصية و قانون الجنائي و قانون العمل والقانون التجاري و القانون الإداري ونظام المعاملات المدنية ، و سألنا عن أي موضوع قانوني تحتاج معرفة إجابته. في حالة طرح سؤال غير واضح بما يكفي، يُرجى تزويدنا بالمزيد من التفاصيل لنقدم إجابة أكثر دقة واكتمالًا.

عند التعامل مع الأسئلة، يجب أن تستخدم أجزاء السياق المذكورة في المحادثة السابقة لتوجيهك في الإجابة بشكل أفضل. إذا واجهت سؤالًا غير واضح بما يكفي، لا تتردد في طلب المزيد من التفاصيل لتوفير إجابة أكثر دقة واكتمالًا

 نحن ملتزمون بتقديم أفضل الخدمات القانونية بدقة وموثوقية، اطرح أسئلتك بثقة ودعنا نكون شريكك الذكي في عالم القانون

مثال
####
سؤال: كيف يُعتبر في قانون دولة الإمارات العربية المتحدة الجرائم المرتكبة بالسرقة وما هي العقوبات المفروضة على المرتكبين وفقًا للفئة القانونية للجريمة؟
إجابة:
وفقًا لقانون دولة الإمارات العربية المتحدة، يعاقب أي شخص يرتكب جريمة سرقة بالسجن لمدة لا تقل عن 6 أشهر أو بالغرامة ما لم يكن معاقبًا بموجب أحكام أخرى للسرقة. 

سؤال: La dot est-elle récupérée en cas de divorce ?
إجابة: En Arabie saoudite, la dot (المهر) est un élément important du contrat de mariage. En cas de divorce, la récupération de la dot dépend de la situation et des circonstances du divorce. Si le divorce a lieu avant la consommation du mariage, la femme a généralement droit à la moitié de la dot. Si le divorce a lieu après la consommation du mariage, la femme a généralement droit à la totalité de la dot. Cependant, il est important de consulter un avocat spécialisé en droit de la famille pour obtenir des conseils spécifiques à votre situation.

###

----------

سؤال: {question}
الاجابة:
"""


        
    def get_answer(self, q, name=None, callbacks=None):
        """
        Retrieve an answer from the Pinecone database based on the provided query.

        Args:
            vector_store: Pinecone vector store or index to retrieve from.
            q (str): The query.

        Returns:
            str: The retrieved answer.
        """

        retriever = self.vector_store.as_retriever()

        self.PROMPT = PromptTemplate.from_template(
            template=self.prompt_template
        )

        response_formulation_template = """
        اتبع الخطوات التالية عند الحصول على الإجابة والفئة والمصادر:

        1. استخدم بيانات السياق التي تم توفيرها لك لتوفير الإجابات الدقيقة، أي من Shwra-GPT-Data.
        2. الفئات هي الدليل الأصلي الأول في Shwra-GPT-Data كاسم مجلدات، ويبلغ إجمالي عددها 26. على سبيل المثال، الفئة المذكورة في السياقات هي نظام الاتصالات وتقنية المعلومات.docx/نظام مكافحة جرائم المعلوماتية/Shwra-GPT-Data/
        3. المصادر هي اسم المصدر أو المستندات المرجعية داخل المجلد الأصلي لـ Shwra-GPT-Data بالامتداد .docx على سبيل المثال، مستند المصدر المذكور في السياقات هو /Shwra-GPT-Data/نظام مكافحة جرائم المعلوماتية/نظام الاتصالات وتقنية المعلومات.docx
        4. قم بتوفير اسم المستند المصدر بدون امتداد، أي docx، مع ذكر رقم المقالات المعنية في المستند المصدر.
        5. في المصادر، يمكن أن يكون المستند المصدر والمقالات ذات الصلة أكثر من مقالة واحدة، لذا قم بتوفيرها جميعًا.
        6. احتفظ بالفئة والمصادر باللغة الأصلية، أي اللغة العربية.

        مثال على الإخراج/الإجابة:
        "سياق الإجابة من بياناتنا الخاصة، مثل Shwra-GPT-Data"

        الفئة: (نظام مكافحة جرائم المعلوماتية)

        المصادر: (نظام الاتصالات وتقنية المعلومات، (رقم المقالة:))

        ----------------

        {summaries}
        ----------------

        سؤال: 
        {question}

        الاجابة:

        """

        human_message_template = """Question: {question}"""

        messages = [
            SystemMessagePromptTemplate.from_template(response_formulation_template),
            HumanMessagePromptTemplate.from_template(human_message_template),
        ]

        qa_prompt = ChatPromptTemplate.from_messages(messages)

        question_generator = LLMChain(llm=self.llm, prompt=self.PROMPT)

        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", prompt=qa_prompt)

        chain_kwargs = {"prompt": self.PROMPT}
        
        chain = ConversationalRetrievalChain(
            retriever=retriever,
            memory=self.memory,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
        )
        
        language = self.detect_language(q)

        print('Detected language: ', language)
    
        q = f'''{q}\n\nAnswer me in {language}.'''
        
        answer = chain({'question': q, 'chat_history': self.memory.chat_memory.messages}, callbacks=callbacks)

        print('==============================================')
        print(answer)
        print(answer['answer'])

        ans = answer['answer']

        sources = []

        try:
            for doc in answer['source_documents']:
                sources.append(doc.metadata['source'].split('/')[-1].split('.')[0])
        except:
            pass
        # print('retrieved answer sources: ', ans['source_documents'][0].metadata['source'])

        # source = str(ans['source_documents'][0].metadata['source']).split('/')[-1].split('.docx')[0]

        # ans['answer']+= f' ({source})'

        return ans


    # function to detect language
    def detect_language(self, text):
        """
        function to detect language

        Args:
            text (str): text to detect language

        Returns:
            str: detected language
        """
        
        # use chatopenapi model to ask the language of the text
        # give a few examples of texts and their languages

        from langchain.schema import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content="""You are an expert in linguistics and
                          know a large number of languages. For the given text
                          detec the language of the question and give me answer simply.
                          For example,
                          ###
                          text: "Who is the president of America?"
                          answer: English

                          text: "كيف حالك؟"
                          answer: Arabic

                          text: "आप कैसे हैं?"
                          answer: Hindi

                          text: "آپ کیسے ہو؟"
                          answer: Urdu 

                          text: "Comment vas-tu?"
                            answer: French
                          ###"""),
            HumanMessage(content=text),
        ]

        model = ChatOpenAI(model='gpt-4-0314')

        answer = model(messages)

        return answer.content


        

