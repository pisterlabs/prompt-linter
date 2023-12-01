

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

def get_titles(chat):
    sys_template = '''\
    Imagine you are a content creator for a YouTube channel dedicated to inspiring and equipping 
    DIY Entrepreneurs - individuals eager to learn how to start and manage their own businesses 
    from scratch using modern tools and technologies. Your channel aims to provide practical, 
    step-by-step guides on starting a business, bootstrapping techniques, self-taught business skills, 
    and lessons learned from successful entrepreneurs throughout history.
        '''
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    context_template=f'''\
    **Target Audience Characteristics:**
    The DIY Entrepreneurs audience would be largely made up of individuals who have a strong 
    independent streak and prefer self-guided learning over structured, institutional education. 
    They may be in different phases of their entrepreneurial journey, from those just toying with 
    the idea of starting a business to those who have already started but want to expand their 
    knowledge and skills. This audience may also include freelancers, independent consultants, 
    or solo entrepreneurs who run their own businesses single-handedly.

    Given this niche, please generate video title ideas and theme descriptions that would cater 
    to our audience's needs. 

    Consider including topics on the use of digital marketing tools, managing finances, 
    optimizing productivity, navigating legal hurdles, sourcing and supply chain management, 
    customer relationship building, and strategies for scaling and growth. 

    Consider the following techniques in combination to create the video titles:

    Sensationalize (Without Clickbait): Stir curiosity by creating a sense of suspense or surprise. 
    However, avoid making untrue claims. This can backfire and damage your credibility. 

    Use Numbers: Numbers quantifies your content, makes your title more specific, and also improves 
    search engine optimization (SEO). 

    Highlight Value Proposition: Clearly specify what value the video will provide to viewers. 
    Will it entertain, educate, inspire or provide a solution to a problem? 

    Leverage Emotional Triggers: Use power words that evoke emotion in your audience. 
    Words like 'mind-blowing', 'unbelievable', 'shocking' can attract viewers. 

    Use Relevant Keywords: Incorporate keywords that your target audience is searching for. 
    It helps in discoverability and makes it clear what your video is about. 

    Leverage Popular Trends: If there's a hot topic or trend relevant to your video, 
    capitalize on it in your title. It could be a meme, a cultural moment, a popular movie, a viral challenge, etc. 

    Pose a Question: Titles that ask intriguing questions can be very effective in engaging viewers. 

    Be Specific: Rather than generic titles, use detailed descriptions. E.g., "5 Steps to Master Lightroom" 
    instead of "How to Use Lightroom." 

    Use of Personal Stories: If your video revolves around personal experiences, include that in the title. 
    It can make the video seem more relatable and engaging. 

    Use Celebrity or Influencer Names: If your video is related to a popular person or influencer in your field, 
    include their name. This can increase visibility. 

    Create Urgency: Words like 'now', 'today', 'immediate', 'hurry' can create a sense of urgency and prompt 
    viewers to click. 
    
    Show Contrasts or Comparisons: People love comparing things. So, if your video does a comparison or shows 
    a stark contrast, it's a good idea to highlight it. 

    Highlight Exclusivity: If your content is exclusive or first-hand information, highlight it in your title. 
    For example, "First look at the new iPhone 15" or "Exclusive Interview with Elon Musk." 

    The 'How-to' Title: People are always looking for guides and solutions. 'How-to' titles are classic and 
    often result in high click-through rates. 

    Storytelling: Make your title a mini-story. E.g., "The Day I Nearly Quit YouTube – How I Overcame Burnout."

    Remember, the titles and theme descriptions should be actionable, engaging, and directly relatable to the 
    challenges and aspirations of a DIY Entrepreneur.
    
    Respond with a list of 5 video titles and short theme descriptions.
    '''
    context_prompt = HumanMessagePromptTemplate.from_template(context_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            sys_prompt, 
            context_prompt, 
        ]
    )
    formatted_prompt = chat_prompt.format_prompt().to_messages()
    llm = chat
    result = llm(formatted_prompt)
    return result.content