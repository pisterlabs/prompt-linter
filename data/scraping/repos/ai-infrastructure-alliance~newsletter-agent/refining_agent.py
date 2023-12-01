import guidance


class RefiningAgent:

  def __init__(self, gpt, logger):
    self.gpt = gpt
    self.logger = logger

  def refine_summary(self, title, summaries):
    summary_1 = summaries[0]
    summary_2 = summaries[1]
    summary_3 = summaries[2]
    self.logger.info(f"[Refining] Refining summary for {title}...")
    refiner = guidance('''
    {{#system~}}
    You are an engineer with a strong passion for the latest developments in AI.
    You constantly read the news, and you are fascinated by the pace of development
    of AI and the power of new AI-driven solutions.
    {{~/system}}
  
    {{#user~}}
    I have written three versions of the section summary for a news article 
    with a title of "{{title}}".
    I want you to pick the best one of these three. Please choose the one that is
    the most exciting and relevant to your interests. Pick the summary that will
    most likely make you to click on the link and read the full article.

    Summary 1: {{summary_1}}
    Summary 2: {{summary_2}}
    Summary 3: {{summary_3}}

    As an answer, please explain which summary you prefer and why.
    {{~/user}}
      
    {{#assistant~}}
    {{gen 'explanation' temperature=0.5 max_tokens=1000}}
    {{~/assistant}}
  
    {{#user~}}
    Now, please return only the text of the summary that you like the most.
    {{~/user}}
  
    {{#assistant~}}
    {{gen 'new_summary' temperature=0.5 max_tokens=500}}
    {{~/assistant}}
    ''',
                       llm=self.gpt)

    result = refiner(title=title,
                     summary_1=summary_1,
                     summary_2=summary_2,
                     summary_3=summary_3)
    self.logger.info(f"[Refiner] Refined the summary for the article: {title}")
    return result["new_summary"]
