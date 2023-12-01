import guidance
import random


class ReviewingAgent:

  def __init__(self, gpt, logger):
    self.gpt = gpt
    self.logger = logger

  def _compare_two_news(self, news_1, news_2):
    self.logger.info(
      f"[Reviewer] Running a comparison: '{news_1.title[:10]}...' VS '{news_2.title[:10]}...'"
    )
    title_1 = news_1.title
    title_2 = news_2.title
    summary_1 = news_1.summary
    summary_2 = news_2.summary
    experts = guidance('''
    {{#system~}}
    You are an engineer with a strong passion for the latest developments in AI.
    You constantly read the news, and you are fascinated by the pace of development
    of AI and the power of new AI-driven solutions. 
    {{~/system}}
  
    {{#user~}}
    You are going to read two pieces of news below and tell me, which one of them
    you find more interesting, exciting and important. Explain step be step why you 
    think so.
    
    1. {{title_1}}
    {{summary_1}}
  
    2. {{title_2}}
    {{summary_2}}
    {{~/user}}
  
    {{#assistant~}}
    {{gen 'reasoning' temperature=0 max_tokens=500}}
    {{~/assistant}}
  
    {{#user~}}
    Now, give me just a short answer to the question above. 
    Write only a number, either 1 or 2.
    {{~/user}}
  
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=500}}
    {{~/assistant}}
    ''',
                       llm=self.gpt)

    result = experts(title_1=title_1,
                     summary_1=summary_1,
                     title_2=title_2,
                     summary_2=summary_2)
    self.logger.info("[Reviewer] Comparison finished.")
    return result["answer"], result["reasoning"]

  # CONTSRAINT: `news` should have even length
  # INPORTANT: news are modified inplace here!
  def run_one_round(self, news):
    random.shuffle(news)
    news.sort(key=lambda x: x.score, reverse=True)
    pairs = [news[i:i + 2] for i in range(0, len(news), 2)]
    bad_answers = []
    for pair in pairs:
      answer, reason = self._compare_two_news(pair[0], pair[1])
      try:
        n = int(float(answer))
        if n == 1:
          pair[0].score += 1
        else:
          pair[1].score += 1
      except:
        self.logger.warn(f"[Reviewer] Get unexpected answer: {answer}")
        bad_answers.append(answer)
    news.sort(key=lambda x: x.score, reverse=True)
    self.logger.info("[Reviewer] Round is finished.")
    return bad_answers
