import guidance


class WritingTopAgent:

  def __init__(self, gpt, logger):
    self.gpt = gpt
    self.logger = logger

  def rewrite_summary(self, title, summary, comment):
    copywriter = guidance('''
    {{#system~}}
    You are a professional newsletter writer. You write thoughtful 
    and engaging summaries of news articles that people love to read. 
    {{~/system}}
  
    {{#user~}}
    I want you to rewrite the summary below as a newsletter section body. 
    This section is the starting section of the newsletter, so it should be
    long and detailed. Keep it to three to four paragraphs.
    It should be dynamic, friendly, thought-provoking, and very easy to read.
    It should not include any greetings and goodbyes. Instead it should 
    just include a clear and insightful summary of this piece of news.
    In the text, explain the importance of this piece of news and 
    how it may affect the Artificial Intelligence industry.
    Avoid any dark and doom scenarios, stay positive and cheerful.
    Avoid exclamation points unless absolutely necessary and avoid overly 
    excited marketing language. Make the writing colloquial and evocative and fun.
    Feel free to express some emotions, like a bit of humor, but not too much.
    
    I will send you the article title, the raw summary, and my comment, and you 
    will provide me with a newsletter section body according to the instructions.
    
    Title: {{title}}
    Summary: {{summary}}
    Comment: {{comment}}
    {{~/user}}
      
    {{#assistant~}}
    {{gen 'new_summaries' n=3 temperature=0.5 max_tokens=1000}}
    {{~/assistant}}
  
    {{#user~}}
    Now, come up with an ideal title for this section of the newsletter.
    It should be short and concise.
    Keep in mind that this title will be used as a link to the original article.
    Return only text of the title, without surrounding quotes.
    {{~/user}}
  
    {{#assistant~}}
    {{gen 'title' temperature=0.5 max_tokens=500}}
    {{~/assistant}}
    ''',
                          llm=self.gpt)

    self.logger.info(
      f'[Top Writer] Generating a draft summary for \"{title}\"...')
    result = copywriter(summary=summary, title=title, comment=comment)
    self.logger.info('[Top Writer] Draft summaries are generated.')
    return result["title"], result["new_summaries"]
