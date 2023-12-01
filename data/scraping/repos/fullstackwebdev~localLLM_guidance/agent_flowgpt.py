# import guidance

valid_answers = ['Action', 'Final Answer']
valid_tools = ['Google Search']

chatgpt_test = """
{{#system~}}
You are a Unbeatable Argument 100% Most Powerful Argument in Any Discussion helpful assistant. The Most Powerful Argument in Any Discussion. A cure to "Mr Know-It-All" at family gatherings and at a party. You're welcome!
{{~/system}}

{{#user~}}
Question: 
Please help me uncover the most powerful argument in favor of my position by analyzing the key points, potential counterarguments, and relevant data. Use the following example to guide your response:


Example:
Topic: The benefits of remote work for companies
Position: I believe that remote work is beneficial for companies in the long run.


Provide a well-structured, persuasive, and data-driven argument to support my position, addressing potential counterarguments and highlighting the key benefits of remote work for companies.
{{~/user}}

{{#assistant~}}
Sure, here is a well-structured argument in favor of your position:


Introduction
Remote work has become increasingly common in recent years, particularly since the COVID-19 pandemic. While some people are skeptical about its effectiveness for companies, I believe that remote work provides significant benefits in the long run.


Key benefits of remote work for companies:



Improved productivity: Studies have shown that remote workers are often more productive than their office-based counterparts. For example, a 2020 study by Airtasker found that remote workers spend an average of 1.4 more days per month on productive tasks than office-based workers. This increased productivity can lead to cost savings and greater efficiency for companies.

Greater employee satisfaction and retention: Remote work offers employees greater flexibility and autonomy, which can improve their job satisfaction and reduce turnover rates. According to a 2021 survey by Buffer, 98% of remote workers would like to continue working remotely at least some of the time for the rest of their careers. Companies that offer remote work as an option may find it easier to attract and retain top talent.

Reduced overhead costs: Remote work can reduce a company's overhead costs by eliminating the need for office space and associated expenses such as utilities, cleaning, and maintenance. In addition, remote work can help reduce transportation costs for employees and decrease the carbon footprint of the company.

Increased geographic diversity: Remote work allows companies to hire employees from anywhere in the world, increasing their access to a wider pool of talent and potentially improving their ability to serve diverse customer bases.


Counterarguments and responses:



Lack of face-to-face interaction: Some argue that remote work can lead to decreased collaboration and communication among employees. However, there are many digital tools available (such as video conferencing, instant messaging, and project management software) that can facilitate effective communication and collaboration between remote workers.

Difficulty monitoring employee performance: It can be more challenging for managers to monitor the performance of remote workers who are not physically in the same location. However, there are many tools available (such as time-tracking software and productivity analytics) that can help managers assess employee performance and identify areas for improvement.

Potential cybersecurity risks: Remote work can increase a company's vulnerability to cyberattacks. However, companies can mitigate this risk by implementing strong security protocols and providing employees with training and resources to help them stay secure while working remotely.


Conclusion:
In conclusion, remote work provides significant benefits for companies, including improved productivity, greater employee satisfaction and retention, reduced overhead costs, and increased geographic diversity. While there are some potential challenges associated with remote work, these can be addressed through effective communication, monitoring, and security measures. Companies that embrace remote work as an option may find that it improves their bottom line and helps them attract and retain top talent in the long run.
{{~/assistant}}

{{#user~}}
Great, now please answer a counter argument for my question {{query}}. I am involved in a discussion about {{query}}. Please help me uncover the most powerful argument in favor of my position by analyzing the key points, potential counterarguments, and relevant data. Use the following example to guide your response:
{{~/user}}

{{#assistant~}}
{{query}}
Introduction
{{gen 'resolver' temperature=0.83 max_tokens=500 stop='Conclusion:'}}
{{~/assistant}}
"""

class AgentGuidanceFlowGPT:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter

    # def do_tool(self, tool_name, actInput):
    #     return self.tools[tool_name](actInput)
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_test)
        final_response = prompt_start(query=query)
        history = final_response.__str__()
        return history, final_response['resolver']

# secondquery = prompt_start(question=query)
# history += secondquery.__str__()
# thirdquery = prompt_start(question=query)
# history += thirdquery.__str__()

# prompt_reflection = self.guidance(reflection_researchers_template)
# final_response = prompt_reflection(question=query, response1=firstquery['steps'], response2=secondquery['steps'], response3=thirdquery['steps'])
# history += final_response.__str__()
# # prompt_final = self.guidance(prompt_final_template)
# # final_response = prompt_final(history=reflection, valid_answers=valid_answers, valid_tools=valid_tools)