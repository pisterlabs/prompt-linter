from flask import Flask, render_template, request
import openai

app = Flask(__name__)

# Define your OpenAI API key here
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

@app.route('/')
def index():
    return render_template('index.html', search_results=None, article=None)  # Initialize search_results and article as None

@app.route('/search', methods=['POST'])
def search():
    try:
        search_query = request.form.get('search_input')
        selected_language = request.form.get('language-select')
        
        # Define the system message and user message for the OpenAI chat
        search_query = f"Is {search_query} related to Sciences, Math, or Technical Subjects?"
        
        # Create a conversation with system and user messages
        conversation = [
            {"role": "system", "content": "You are a helpful assistant that provides information."},
            {"role": "user", "content": search_query}
        ]
        
        # Generate a response from OpenAI's GPT-3 model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )
        
        # Extract and check if the response indicates the topic is Sciences, Math, or Technical Subjects
        answer = response['choices'][0]['message']['content'].strip().lower()
        
        if answer == "yes":
            # If the topic is related, generate an article with headings
            article_content = generate_article(search_query)
            
            # Render the HTML template with search results and article
            return render_template('index.html', search_results=article_content, article=True)
        else:
            # If the topic is not related, display a message
            return render_template('index.html', search_results="This topic is not related to Sciences, Math, or Technical Subjects.", article=False)
    
    except Exception as e:
        error_message = 'Error: ' + str(e)
        return render_template('error.html', error_message=error_message)

def generate_article(topic):
    # Define the headings for the article
    headings = [
        "Introduction/Theory",
        "History",
        "Principles and Laws",
        "Formulas and Calculations",
        "Experiments and Demonstrations",
        "Case Studies",
        "Implications and Applications",
        "Controversies and Debates",
        "Future Directions",
        "Summary and Review"
    ]

    # Initialize the article content with the topic
    article_content = f"<h2>{topic}</h2>"

    # Generate content for each heading
    for heading in headings:
        article_content += f"<h3>{heading}</h3>"
        article_content += "<p>Content for this section goes here...</p>"

    return article_content

if __name__ == '__main__':
    app.run(port=8080, debug=True)
