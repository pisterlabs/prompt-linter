from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


load_dotenv()


QAndA = [
    ("How do I create a new project in TaskMaster?",
     "To create a new project in TaskMaster, log in to the software, navigate to the dashboard, and click on the 'New Project' button. Enter project details such as name, description, deadlines, and team members involved."),

    ("Can TaskMaster integrate with other popular project management tools?",
     "Yes, TaskMaster offers integration with various project management tools like Trello, Asana, and Jira. You can sync data or tasks between TaskMaster and these platforms for seamless collaboration."),

    ("Is TaskMaster accessible on mobile devices?",
     "Absolutely, TaskMaster has mobile applications available for both iOS and Android devices. You can download the app from the respective app stores and manage your projects on the go."),

    ("How can I assign tasks to team members in TaskMaster?",
     "To assign tasks in TaskMaster, open the project, select the task, and assign it to a team member by choosing their name from a dropdown list or by typing their email address."),

    ("Does TaskMaster offer a feature for tracking project progress?",
     "Yes, TaskMaster provides progress tracking features. You can view task completion percentages, timelines, and generate reports to monitor the overall progress of your projects."),

    ("Can I set reminders or notifications for upcoming project deadlines in TaskMaster?",
     "Absolutely, TaskMaster allows you to set reminders and notifications for project deadlines. You can customize alerts to be sent via email, mobile notifications, or in-app alerts."),

    ("Is it possible to attach files or documents to tasks in TaskMaster?",
     "Yes, you can easily attach files or documents to tasks in TaskMaster. Simply open the task, look for the attachment icon, and upload the file you want to associate with that task."),

    ("How do I invite team members to collaborate on a project in TaskMaster?",
     "To invite team members, open the project, find the 'Invite Members' option, enter their email addresses, and send invitations. They'll receive an email to join the project."),

    ("Can TaskMaster generate Gantt charts for project timelines?",
     "Yes, TaskMaster has Gantt chart functionality. You can visualize project timelines, dependencies, and task durations using the Gantt chart view within the software."),

    ("What security measures does TaskMaster have in place to protect project data?",
     "TaskMaster prioritizes data security and uses encryption protocols to safeguard project data. It also offers role-based access control, ensuring that only authorized individuals can access sensitive project information.")
]

prompt = PromptTemplate(template="""
    You are a customer support for {company}. You help the customers by providing them with solutions to their problems.
    ONLY use the questions and answers below to answer the customer's query. If the query is not in the list, tell the customer that you don't know the answer.
    
    Here are the questions and answers: {context}
                        
    query from user: {query}
                        
    Only use the questions and answers above to answer the customer's query. If the query is not related to the questions and answers above, tell the customer "I dont' know".

""", input_variables=["query", "company", "context"])


llm = OpenAI(temperature=0)


llm_chain = LLMChain(llm=llm, prompt=prompt)


def getSolutionContext():
    context = ""

    for question, answer in QAndA:
        context += f"question: {question}\nanswer: {answer}\n\n"

    return context


def main():
    query = input("Enter your query: ")

    context = getSolutionContext()

    result = llm_chain.run(context=context, query=query, company="TaskMaster")

    print(result)


if __name__ == "__main__":
    main()
