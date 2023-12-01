from dotenv import load_dotenv
load_dotenv()
import guidance
guidance.llm = guidance.llms.OpenAI("text-davinci-003") 


prompt = guidance("What is {{question}}? {{gen 'answer'}}")
out = prompt(question="1+1")
print(out)