from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_outlines(chat, outlines_input: str):
    sys_template = f"""\
    You are a digital media and communication specialist who is well-versed in content creation strategies and techniques 
    for YouTube videos and can provide informed guidance and actionable advice to DIY entrepreneurs.
    """
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    human_template = f"""\
    You have been tasked with using the following information and techniques to generate YouTube video content descriptions 
    and outlines for a video title and/or topic description provided to you.

    Audience Description:
    The DIY Entrepreneurs audience would be largely made up of individuals who have a strong independent streak and prefer 
    self-guided learning over structured, institutional education. They may be in different phases of their entrepreneurial 
    journey, from those just toying with the idea of starting a business to those who have already started but want to 
    expand their knowledge and skills. This audience may also include freelancers, independent consultants, or solo 
    entrepreneurs who run their own businesses single-handedly. 

    Writing Techniques:
    To outline engaging and effective YouTube videos, you should start with a strong hook to grab your audience's attention. 
    Use visual aids, personal stories, narrative arcs and metaphors to illustrate your points sustainably. 
    
    Share case studies of success, and involve your viewers through interactive elements and regular calls to 
    action. 
    
    Create emotional connections by incorporating humor, passion, excitement, and shared experiences. 
    
    Keep your content simple, clear, and jargon-free, while maintaining high production quality. 
    
    Reinforce key points through consistency and repetition. 
    
    Additionally, keep your audience engaged with cliffhangers and teasers for future videos. 


    You are to apply this information and techniques to creating video descriptions and outlines for only the 
    following title and/or topic information: 

    --- 
    {outlines_input}
    --- 

    Respond with 3 detailed video descriptions and outlines options for the video content.
    Each option should incorporate a different theme or take on the topic.
    Format your response in the markdown syntax.
    """
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt,
            human_prompt,
        ]
    )
    formatted_prompt = chat_prompt.format_prompt(
        outlines_input=outlines_input
    ).to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content
