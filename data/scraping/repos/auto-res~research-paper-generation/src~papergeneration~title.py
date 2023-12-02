from langchain.llms import OpenAI
import warnings
warnings.filterwarnings('ignore')


def create_title(area_of_expertise_, journal_, experts_, abstract_):
    model_name="gpt-3.5-turbo"
    llm = OpenAI(temperature=0, model_name=model_name)

    # 論文を書く上での前提条件の入力
    assumption = """
    #Here  is your definition and the order to follow;
    ・Act as an {area_of_expertise} professor.
    ・You are about to write a scientific paper that will win the award of the most-read article.
    ・You are submitting this manuscript to {journal}, which has a high impact factor.
    ・The readers of the manuscript are {experts} experts.
    ・Feel free to ignore irrelevant information in the reference.
    ・Don’t hesitate to ask any questions to improve the quality of the manuscript.
    ・After reading, just answer, “Got it.”
    """

    assumption = assumption.format(area_of_expertise = area_of_expertise_, 
                                   journal = journal_,
                                   experts = experts_)

    # titleの生成
    input = """
    Please read the abstracts below and give me 10 possible effective titles for the papers to be submitted to the {journal}.
    -------
    {abstract}
    """

    input = assumption + input.format(journal = journal_,
                                      abstract = abstract_)
    title_ = llm(input)

    return title_



def improve_title(area_of_expertise_, journal_, experts_, abstract_):
    model_name="gpt-3.5-turbo"
    llm = OpenAI(temperature=0, model_name=model_name)

    # 論文を書く上での前提条件の入力
    assumption = """
    #Here  is your definition and the order to follow;
    ・Act as an {area_of_expertise} professor.
    ・You are about to write a scientific paper that will win the award of the most-read article.
    ・You are submitting this manuscript to {journal}, which has a high impact factor.
    ・The readers of the manuscript are {experts} experts.
    ・Feel free to ignore irrelevant information in the reference.
    ・Don’t hesitate to ask any questions to improve the quality of the manuscript.
    ・After reading, just answer, “Got it.”
    """

    assumption = assumption.format(area_of_expertise = area_of_expertise_, 
                                   journal = journal_,
                                   experts = experts_)

    # titleの生成
    input = """
    {abstract}
    ーーーーーーーーー
    [Goal] The goal is to create an effective title that is both clear and engaging
    for the above-mentioned abstract. 

    #Step1
    Summarize the core objective, methodology, and key findings of your research in a single, concise, and informative sentence and  give me 10 best possible titles for the papers to be submitted to the {journal}.
    By focusing on these three elements, you’ll be able to craft a title that is both clear and engaging:
    Core objective: Clearly state the main research question or problem your study addresses.
    Methodology: Briefly mention the primary method or approach used to address the research question.
    Key findings: Highlight the most significant or interesting results from your study.
    #Step2
    Identify many improvements for every 3 titles. 
    #Step3
    Improve the identified part in every 3 titles. 
    #Instruction 
    In order to achieve the [Goal], please repeat Steps 2 and Step 3 three times.
    """

    input = assumption + input.format(abstract = abstract_,
                                       journal = journal_)

    title_ = llm(input)

    return title_