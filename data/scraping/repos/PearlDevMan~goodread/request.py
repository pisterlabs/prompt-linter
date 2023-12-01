
import streamlit as st
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import anthropic
import json
import os
# Set the URL of your Flask app
FLASK_APP_URL = 'http://176.37.67.205:5000'

def convert_reviews_to_string(reviews):
    review_string = ""
    for review in reviews:
        review_string += review["Review"] + " | \n"
    return review_string.strip()

# Define your file download function
def download_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return data

def promptlist(): 
    prompts = [   '''Prompt: Act as an expert marketer and market research analyst. Analyze the provided reviews to uncover key insights and compelling language to use in copy and messaging. Follow these steps:

        Read through all reviews carefully, multiple times if needed. Look for common themes, recurring questions or concerns, expressions of delight or frustration. Copy the actual language used verbatim whenever possible.
        Questions to guide analysis:

            What outcomes or results are customers hoping to achieve?
            What needs or problems are customers looking to solve?
            What specific benefits do customers mention?
            What words or phrases do customers use frequently to describe their experience?''',

        ''' Organize the insights and language into categories:

        Pain points: What problems or annoyances do customers frequently mention? What do they wish was different or better?

        Desired outcomes: What are customers ultimately trying to achieve or hoping to experience? What needs do they want fulfilled?

        Questions: What do customers wonder or ask about frequently regarding the product or service? What are they unsure or confused about?

        Favorite features: What specific attributes or capabilities do customers call out as valuable and pleasing? What creates delight?

            Identify the most compelling and insightful language to use in copy and messaging. Consider messages and wording that speak directly to the primary pain points, desired outcomes, and questions. Mirror the words and sentiments of customers.''',

        '''Potential headlines:

            What sentences or short phrases would make striking headlines?
            What types of headlines would resonate most?

            Compile examples of copy, headlines, and messaging based on these insights. For each, note the rationale for why that particular copy or message was chosen. Explain how it connects to the review analysis.''',
            
        '''Bullet point benefits: What are the 5-10 most appealing benefits or advantages mentioned?  What product attributes, features, or capabilities come up most often as positives?''',

        '''Inspirations for sales copy:

            What stories, examples, or anecdotes help to illustrate key benefits?
            What imagery or sensory language is used to describe the experience?
            What emotions are expressed in a genuine, authentic way?

            Make recommendations for how these insights and examples of copy could be tested to determine resonance with the target audience. Discuss ways both qualitative feedback and quantitative metrics could be used to optimize the copy and messaging over time based on reviews.''',

        '''Messages to improve:

            What objections, downsides, or disadvantages are mentioned in critical or lower-star reviews?
            What areas of confusion or unanswered questions can be addressed?
            What language choices seem inauthentic or overly salesy? How could this be improved?''',

        '''Perform word frequency analysis on the reviews. Identify frequently used words or phrases and consider their significance in relation to the book. Take note of any keywords or buzzwords that could be useful for marketing purposes.''']
    return prompts


def getQA(data):
    
    answer = ""
    prompts = promptlist()
    c = anthropic.Client(api_key='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')
    reviews = convert_reviews_to_string(data)
    for i in range(1, len(prompts)):
        resp = c.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompts[0]} {reviews} {prompts[i]}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-v1.3-100k",
            max_tokens_to_sample=10512,
        )
        answer += resp["completion"]
    return answer



def create_pdf(text, title):
    filename = title + ".pdf"
    lines = text.split("\n")
    c = canvas.Canvas(filename, pagesize=letter)

    # Set the font and font size
    c.setFont("Helvetica", 12)

    # Set the line height
    line_height = 14

    # Set the margin
    margin = inch

    # Loop through the lines and draw them on the canvas
    y = letter[1] - margin
    for line in lines:
        # Split the line into words
        words = line.split()

        # Start with an empty line
        line_text = ""

        # Loop through the words and add them to the line until it's too long
        for word in words:
            if c.stringWidth(line_text + " " + word) < letter[0] - (2 * margin):
                line_text += " " + word
            else:
                # Check if there is enough space on the current page for the next line
                if y - line_height < margin:
                    # If there isn't enough space, create a new page
                    c.showPage()
                    y = letter[1] - margin

                # Draw the line on the canvas
                c.drawString(margin, y, line_text.strip())

                # Move to the next line
                y -= line_height
                line_text = ""

        # Draw any remaining text on the current page
        if line_text:
            # Check if there is enough space on the current page for the next line
            if y - line_height < margin:
                # If there isn't enough space, create a new page
                c.showPage()
                y = letter[1] - margin

            # Draw the line on the canvas
            c.drawString(margin, y, line_text.strip())

            # Move to the next line
            y -= line_height

    # Save the PDF file
    c.save()
    return filename



def promain(data):    
    return create_pdf(getQA(data), "review")


# Streamlit app code
def main():
    st.title("Goodreads Reviews - Please wait up to 5 minutes for results to show")

    # Create an input field for the URL
    url = st.text_input("Enter the URL")
    headers = {'Content-Type': 'application/json'}

    payload = {'url':url}

    # Create a button to execute the Selenium API
    if st.button("Execute"):
        # Send a POST request to the Flask app input route
        response = requests.post(f'{FLASK_APP_URL}/execute_selenium_code', json=payload)
        
        # Display the response from the API
        if response.status_code == 200:
            data = response.json()
            st.write("API Response:")
            st.write(data)
            filename = promain(data)
            # Call your download function
            file_data = download_file(filename)
            # Set up the file download
            st.download_button(
                label='Download PDF',
                data=file_data,
                file_name=filename,
                mime='application/pdf'
            )
            # Delete the file after download is complete
            os.remove(f'/{filename}')
        else:
            st.write("Error executing Selenium code")

if __name__ == '__main__':
    main()
