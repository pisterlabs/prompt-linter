from dotenv import load_dotenv
import guidance

load_dotenv()


guidance.llm = guidance.llms.OpenAI("text-davinci-003")

program= guidance(
    """ what is the top ten most common commands  used in the {{os}} os.
    {{#block hidden=True}}
    A few Example of fruits are:
    1-Apple
    2-Mango
    {{gen 'Example' n=1 stop='\\n' max_tokens=50 temperature=0}}
    {{/block}}

     Here are top comman commands are:
     {{#geneach 'commands' num_iterations=10}}
     [{{@index}}]: "{{gen 'that' stop='"'}}" --Description: "{{gen 'description' stop='"'}}"
     {{/geneach}}

  

   """
)

result=program(os="Linux")
print(result)
print("===")
print(result["Example"])