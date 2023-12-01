# coding=utf-8
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import logging
import tiktoken
import time
import os
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
openai_api_key = os.environ.get("OPENAI_API_KEY")
# Summarization parameters
OPENAI_MODEL_NAME: str = "text-davinci-003"
OPENAI_TOKENIZER_NAME: str = "cl100k_base"
OPENAI_MODEL_TEMPERATURE: float = 0.3  # 0 is fully deterministic, 1 is most random
OPENAI_MODEL_MAX_TOKENS: int = 500  # langchain automatically sets to max for OPENAI_MODEL_NAME
ANSWER_PROMPT_TEMPLATE: str = (
    "Answer a question by using information from a text."
    "Use as few words as possible."
    "Only include any information from the text that relates to the question."
    "The question is '{question}'"
    "Ignore any information that appear to be website artifacts."
     "Only include information that is found in the text."
    "If the answer cannot be found in the text, just say that you don't know. Don't try to make up an answer or use other sources of information."
    "Format your answer in long and complete sentences. Do not include fragments."
    "The text is '{text}'"
    # "The sources are '{source}'"
)

# Summarization helpers
encoder = tiktoken.get_encoding(OPENAI_TOKENIZER_NAME)

answer_prompt = PromptTemplate(template=ANSWER_PROMPT_TEMPLATE, input_variables=["question", "text"])

def answer_follow_up_question(question: str, references) -> str:
    """
    Arguments:
        question {str} -- Question
        text {str} -- Text

    Returns:
        str -- Answer
    """
    # Initialize OpenAI LLM with langchain
    llm = OpenAI(
        model_name=OPENAI_MODEL_NAME,
        temperature=OPENAI_MODEL_TEMPERATURE,
        max_tokens=OPENAI_MODEL_MAX_TOKENS,
        openai_api_key=openai_api_key,
    )
    text1 = [Document(page_content=i['summary']) for i in references]
    sources =[i['title'] for i in references]
    # text1 = [Document(page_content=i['summary']) for i in references]
    # Creating langchain summarize chain
    # chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=text1,question=question)
    return answer
# TODO make the references into single sentences, make k larger

def main():
    # reference1 = "European markets live updates: News from WEF, data and earningsSkip NavigationwatchliveMarketsPre-MarketsU.S. MarketsCurrenciesCryptocurrencyFutures & CommoditiesBondsFunds & ETFsBusinessEconomyFinanceHealth & ScienceMediaReal EstateEnergyClimateTransportationIndustrialsRetailWealthLifeSmall BusinessInvestingPersonal FinanceFintechFinancial AdvisorsOptions ActionETF StreetBuffett ArchiveEarningsTrader TalkTechCybersecurityEnterpriseInternetMediaMobileSocial MediaCNBC Disruptor 50Tech GuidePoliticsWhite HousePolicyDefenseCongressEquity and OpportunityCNBC TVLive TVLive AudioBusiness Day ShowsEntertainment ShowsFull EpisodesLatest VideoTop VideoCEO InterviewsCNBC DocumentariesCNBC PodcastsCNBC WorldDigital OriginalsLive TV ScheduleWatchlistInvesting ClubTrust PortfolioAnalysisTrade AlertsVideoEducationPROPro NewsPro LiveSubscribeSign InMenuMake ItSelectUSAINTLwatchliveSearch quotes, news & videosWatchlistSIGN INCreate free accountMarketsBusinessInvestingTechPoliticsCNBC TVWatchlistInvesting ClubPROMenuLIVE UPDATESUpdated Tue, Jan 17 20233:16 AM ESTShareShare Article via FacebookShare Article via TwitterShare Article via LinkedInShare Article via EmailEuropean markets mixed as economic concerns dominate DavosElliot SmithHolly EllyattThis is CNBC's live blog covering European markets.European markets were muted on Tuesday, with concerns about the global economy high on the agenda at the World Economic Forum in Davos this week.European marketsThe pan-European Stoxx 600 hovered around the flatline in early trade, with autos adding 0.5% while retail stocks dropped by a similar amount.CNBC will be speaking to a range of delegates at the forum on Tuesday, including the leaders of Spain, Latvia, Lithuania and Poland and the CEOs of Unilever, UBS, Allianz, Swiss Re and many others. Follow our coverage here.Concerns over the direction of the global economy, persistent inflation, fragmentation and sluggish growth are high on the agenda, as well as the war in UkraineInvestors will also be digesting a slew of "
    # article1 = {"title": "", "url": "", "date": "", "content": "someone with a football", "summary": reference1}
    # article2 = {"title": "", "url": "", "date": "", "content": "someone not with a football", "summary": reference1}
    # article3 = {"title": "", "url": "", "date": "", "content": "someone not with a soccer", "summary": reference1}
    # references = [article1, article2, article3]
    # question = "What happened to Mars Markets?"
    # answer = answer_follow_up_question(question, references)
    # print(answer)

    reference1 = "European markets live updates: News from WEF, data and earningsSkip NavigationwatchliveMarketsPre-MarketsU.S. MarketsCurrenciesCryptocurrencyFutures & CommoditiesBondsFunds & ETFsBusinessEconomyFinanceHealth & ScienceMediaReal EstateEnergyClimateTransportationIndustrialsRetailWealthLifeSmall BusinessInvestingPersonal FinanceFintechFinancial AdvisorsOptions ActionETF StreetBuffett ArchiveEarningsTrader TalkTechCybersecurityEnterpriseInternetMediaMobileSocial MediaCNBC Disruptor 50Tech GuidePoliticsWhite HousePolicyDefenseCongressEquity and OpportunityCNBC TVLive TVLive AudioBusiness Day ShowsEntertainment ShowsFull EpisodesLatest VideoTop VideoCEO InterviewsCNBC DocumentariesCNBC PodcastsCNBC WorldDigital OriginalsLive TV ScheduleWatchlistInvesting ClubTrust PortfolioAnalysisTrade AlertsVideoEducationPROPro NewsPro LiveSubscribeSign InMenuMake ItSelectUSAINTLwatchliveSearch quotes, news & videosWatchlistSIGN INCreate free accountMarketsBusinessInvestingTechPoliticsCNBC TVWatchlistInvesting ClubPROMenuLIVE UPDATESUpdated Tue, Jan 17 20233:16 AM ESTShareShare Article via FacebookShare Article via TwitterShare Article via LinkedInShare Article via EmailEuropean markets mixed as economic concerns dominate DavosElliot SmithHolly EllyattThis is CNBC's live blog covering European markets.European markets were muted on Tuesday, with concerns about the global economy high on the agenda at the World Economic Forum in Davos this week.European marketsThe pan-European Stoxx 600 hovered around the flatline in early trade, with autos adding 0.5% while retail stocks dropped by a similar amount.CNBC will be speaking to a range of delegates at the forum on Tuesday, including the leaders of Spain, Latvia, Lithuania and Poland and the CEOs of Unilever, UBS, Allianz, Swiss Re and many others. Follow our coverage here.Concerns over the direction of the global economy, persistent inflation, fragmentation and sluggish growth are high on the agenda, as well as the war in UkraineInvestors will also be digesting a slew of "
    article1 = {"title": "", "url": "", "date": "", "content": "someone with a football", "summary": reference1}
    article2 = {"title": "", "url": "", "date": "", "content": "someone not with a football", "summary": reference1}
    article3 = {"title": "", "url": "", "date": "", "content": "someone not with a soccer", "summary": reference1}
    references = [article1, article2, article3]
    question = "What happened to Mars Markets?"
    answer = answer_follow_up_question(question, references)
    print(answer)

if __name__ == "__main__":
    main()