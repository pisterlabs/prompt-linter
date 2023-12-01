from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import dotenv
import os

# Load environment variables
dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

# Constants

output_format = """
- Salutation (with no name, you can use something generic like 'buddy') + warm and original welcome + email number update + funny short satirical pun
\n💡May I suggest:
  \n[Emoji] + Action Name
  \n[Emoji] + Action Name
\n[Emoji] + Action Name
"""

# Custom Chain


class SayHiChain(LLMChain):
    def __init__(self):
        llm = OpenAI(openai_api_key=OPEN_API_KEY,
                     temperature=0.8, model_name="gpt-3.5-turbo")
        prompt = PromptTemplate(
            input_variables=["unreadEmails", "eventsTodayMainCalendar", "drafts"],
            template=f"""You are a personal assistant. You must create a fun welcome message to the user saying how many emails they got.
            Something like 'Hey user, you have {{unreadEmails}} unread emails today!, and please note that you have {{eventsTodayMainCalendar}} events today on your main calendar.'
            And also do not forget to mention that you have {{drafts}} email drafts.
            When you finish with the update, you should also offer the user a list of possible actions that they can do with you.
            The actions should be related to the your assistant task. You could offer to reply or create some drafts. Try to be very helpfull as a good personal assistant
            The should be presented as a list, 1 option per line. and no more than 3  
     
            Plase use this output format:
            {output_format}
            """
        )
        super().__init__(llm=llm, prompt=prompt, verbose=True)
