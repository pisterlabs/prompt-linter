import re
from langchain.llms import Ollama


def extract_items(input_string):
    # Find the text inside the << >>
    content = re.search(r'<<(.+?)>>', input_string)

    if content:
        content = content.group(1)
    else:
        return []

    # Split the content by the | separator and remove whitespace
    items = [item.strip() for item in content.split('|')]

    # Remove the quotes from each item
    items = [re.sub(r'^"|"$', '', item) for item in items]

    return items


def slide_data_gen(topic):
    llm = Ollama(model="dolphin2.1-mistral",
                 temperature="0.4")

    slide_data = []

    point_count = 5

    slide_data.append(extract_items(llm(f"""
    You are a text summarization and formatting specialized model that fetches relevant information

    For the topic "{topic}" suggest a presentation title and a presentation subtitle it should be returned in the format :
    << "title" | "subtitle >>

    example :
    << "Ethics in Design" | "Integrating Ethics into Design Processes" >>
    """)))

    slide_data.append(extract_items(llm(f"""
    You are a text summarization and formatting specialized model that fetches relevant information
            
    For the presentation titled "{slide_data[0][0]}" and with subtitle "{slide_data[0][1]}" for the topic "{topic}"
    Write a table of contents containing the title of each slide for a 7 slide presentation
    It should be of the format :
    << "slide1" | "slide2" | "slide3" | ... | >>
            
    example :
    << "Introduction to Design Ethics" | "User-Centered Design" | "Transparency and Honesty" | "Data Privacy and Security" | "Accessibility and Inclusion" | "Social Impact and Sustainability" | "Ethical AI and Automation" | "Collaboration and Professional Ethics" >>          
    """)))

    for subtopic in slide_data[1]:

        data_to_clean = llm(f"""
        You are a content generation specialized model that fetches relevant information and presents it in clear concise manner
                
        For the presentation titled "{slide_data[0][0]}" and with subtitle "{slide_data[0][1]}" for the topic "{topic}"
        Write the contents for a slide with the subtopic {subtopic}
        Write {point_count} points. Each point 10 words maximum.
        Make the points short, concise and to the point.
        """)

        cleaned_data = llm(f"""
        You are a text summarization and formatting specialized model that fetches relevant information and formats it into user specified formats
        Given below is a text draft for a presentation slide containing {point_count} points , extract the {point_count} sentences and format it as :
                    
        << "point1" | "point2" | "point3" | ... | >>
                    
        example :
        << "Foster a collaborative and inclusive work environment." | "Respect intellectual property rights and avoid plagiarism." | "Uphold professional standards and codes of ethics." | "Be open to feedback and continuous learning." >>

        -- Beginning of the text --
        {data_to_clean}
        -- End of the text --         
        """)

        slide_data.append([subtopic] + extract_items(cleaned_data))

    return slide_data
