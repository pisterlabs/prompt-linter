from flask import Flask, request, render_template
import openai
import constants

text_test = "This is a test"
num_words = len(text_test.split(" "))
print(f"Num of words from the test is: {num_words}")
# persistent chats with davinci


openai.api_key = constants.api_key

app = Flask(__name__)

# Store the conversation history
conversation_history = {}

def count_tokens(text):
    num_words = len(text.split(" "))
    add_fifty_percent = len(text.split(" ")) / 2
    num_of_tokens = num_words + add_fifty_percent
    return num_of_tokens


@app.route("/", methods=["GET", "POST"])
def start_here():
    if request.method == "POST":
        user_id = request.remote_addr  # Simplified user identification (use a better method in production)
        text_question = request.form.get("question")

        # Retrieve the existing conversation history, if it exists
        chat_history = conversation_history.get(user_id, "")

        # Append the new question to the history
        chat_history += f"\nHuman: {text_question}\nAI:"
        request_text = chat_history
        print(f"Chat history is: {request_text}")
        # count the tokens
        # request_tokens = count_tokens(request_text)


        # Make a request to the OpenAI API using the text-davinci-003 model
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=chat_history,
                max_tokens=1500,
                temperature=0.7,
                # get three responses.
                n = 3,
                stop="\nHuman:"
            )

            response_text = response

            # Extract one AI response and update the chat history
            ai_response = response.choices[0].text.strip()
            chat_history += ai_response

            ai_response02 = response.choices[1].text.strip()
            ai_response03 = response.choices[2].text.strip()


            conversation_history[user_id] = chat_history


             # response_tokens = count_tokens(response_text)
            total_tokens = 999 # request_tokens + response_tokens

            # Send the responses back to the user
            return render_template("index.html", textQuestion = "",
                                                 textAnswer = ai_response,
                                                 textAnswer02 = ai_response02,
                                                 textAnswer03 = ai_response03,
                                                 tokenUsage = total_tokens,
                                                 queryUsage = chat_history
                                   )

        except openai.error.OpenAIError as e:
            print(f"OpenAIError occurred: {e.__class__.__name__} - {e}")
            return render_template("index.html", textQuestion=text_question, textAnswer=f"OpenAIError occurred: {e.__class__.__name__} - {e}")
        except Exception as e:
            print(f"Unexpected error: {e.__class__.__name__} - {e}")
            return render_template("index.html", textQuestion=text_question, textAnswer=f"Unexpected error: {e.__class__.__name__} - {e}")

        print(f"Request Tokens: {request_tokens}, Response Tokens: {response_tokens}, Total Tokens: {total_tokens}")



    return render_template("index.html", textQuestion="", textAnswer="")


if __name__ == "__main__":
    app.run(debug=True)