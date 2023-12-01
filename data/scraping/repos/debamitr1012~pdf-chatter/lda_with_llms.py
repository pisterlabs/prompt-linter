import gensim
import nltk
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pypdf import PdfReader
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
def preprocess(text, stop_words):
    result = []
    for token in simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            result.append(token)
    return result
def get_topic_lists_from_pdf(file, num_topics, words_per_topic):
    loader = PdfReader(file)
    documents= []
    for page in loader.pages:
        documents.append(page.extract_text())
    nltk.download('stopwords')
    stop_words = set(stopwords.words(['english','spanish']))
    processed_documents = [preprocess(doc, stop_words) for doc in documents]
    dictionary = corpora.Dictionary(processed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in processed_documents]
    lda_model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=15
        )
    topics = lda_model.print_topics(num_words=words_per_topic)
    topics_ls = []
    for topic in topics:
        words = topic[1].split("+")
        topic_words = [word.split("*")[1].replace('"', '').strip() for word in words]
        topics_ls.append(topic_words)
    return topics_ls
def topics_from_pdf(llm, file, num_topics, words_per_topic):
    list_of_topicwords = get_topic_lists_from_pdf(file, num_topics,
                                                  words_per_topic)
    string_lda = ""
    for list in list_of_topicwords:
        string_lda += str(list) + "\n"
    template_string = '''Describe the topic of each of the {num_topics}
        double-quote delimited lists in a simple sentence and also write down
        three possible different subthemes. The lists are the result of an
        algorithm for topic discovery.
        Do not provide an introduction or a conclusion, only describe the
        topics. Do not mention the word "topic" when describing the topics.
        Use the following template for the response.
        1: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>
        2: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>
        ...
        n: <<<(sentence describing the topic)>>>
        - <<<(Phrase describing the first subtheme)>>>
        - <<<(Phrase describing the second subtheme)>>>
        - <<<(Phrase describing the third subtheme)>>>
        Lists: """{string_lda}""" '''
    prompt_template = ChatPromptTemplate.from_template(template_string)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({
        "string_lda" : string_lda,
        "num_topics" : num_topics
        })
    return response
openai_key = "sk-<your key here>"
llm = OpenAI(openai_api_key=openai_key, max_tokens=-1)
file = "C:/Users/91983/Downloads/pdf_chat-master/the-metamorphosis.pdf"
num_topics = 6
words_per_topic = 30
summary = topics_from_pdf(llm, file, num_topics, words_per_topic)
print(summary)
file = "C:/Users/91983/Downloads/pdf_chat-master/Hilbert.pdf"
summary = topics_from_pdf(llm, file, num_topics, words_per_topic)
print(summary)