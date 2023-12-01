import streamlit as st
from pymongo import MongoClient
import re
# from streamlit_extras.app_logo import add_logo
from poe_api_wrapper import PoeApi


try:

    Poeclient = PoeApi("CN6Yyu36OUZAxL1N-ytjvg==")

except Exception as e:
    print(e)
    print("error in connecting to POe")



def get_chatgpt_answer(question):
    # bot_list = ["chinchilla", "gpt3_5","chinchilla_instruct", "capybara","acouchy","llama_2_7b_chat","llama_2_13b_chat","llama_2_70b_chat","code_llama_7b_instruct","code_llama_13b_instruct","code_llama_34b_instruct","upstage_solar_0_70b_16bit"]

    bot_list = ["chinchilla"]
    for bot in bot_list:

        try:
            for chunk in Poeclient.send_message(bot, question):
                pass
            return chunk["text"]
        except Exception as e:
            print(e)
            # raise e

    return get_cohere_answer(question)
import cohere
co = cohere.Client('PQ50WPjjMsFSzUhZlMQaGTlS30MyRs9YkbuKfhHh')

def get_cohere_answer(question):
    """
    This function uses the Cohere API to get an answer for the question
    """

    response = co.generate(
  model='command-nightly',
  prompt="""
Error Example: a cookie header was received [${jndi:ldap://log4shell-generic-8molyf0ab2aqtscsyugh${lower:ten}.w.nessus.org/nessus}=${jndi:ldap://log4shell-generic-8molyf0ab2aqtscsyugh${lower:ten}.w.nessus.org/nessus};] that contained an invalid cookie.

Regex: `a cookie header was received \[.*?\] that contained an invalid cookie\.`

Solution: This regex will help you identify instances where a cookie header is received with an invalid cookie. You should then inspect the actual cookie content to determine the cause of the error.

---

Error Example: servlet.service() for servlet [dispatcherservlet] in context with path [] threw exception

Regex: `servlet\.service\(\) for servlet \[.*?\] in context with path \[\] threw exception`

Solution: This regex will help you identify instances where a servlet named "dispatcherservlet" in a specific context path throws an exception. You should investigate the servlet configuration and the exception stack trace to diagnose and resolve the issue.

---

Error Example: discoveryclient_rdd-async-service/p1049433.prod.cloud.fedex.com:rdd-async-service:6521 - was unable to send heartbeat!

Regex: `discoveryclient_.*? - was unable to send heartbeat!`

Solution: This regex will help you identify instances where a component with a name starting with "discoveryclient_" was unable to send a heartbeat. You should check the configuration and network connectivity of the component to ensure it can send heartbeats as expected.

---

Error Example: exception processing errorpage[errorcode=0, location=/error]

Regex: `exception processing errorpage\[.*?\]`

Solution: This regex will help you identify instances where an exception is raised during the processing of an error page. You should investigate the error page configuration and the specific error code to understand and address the issue.

---
please suggest a regex for the following error and a brief solution in a single line to solve the following error:


just provide the two values in the following format:

Regex: <regex>

Solution: <solution>

Error Example:""" + question,
  max_tokens=244,
  temperature=0,
  k=0,
  stop_sequences=["---"],
  return_likelihoods='NONE')
# print('Prediction: {}'.format(response.generations[0].text))
    return response.generations[0].text

# Access MongoDB connection details from Streamlit secrets
db_uri = st.secrets["db_uri"]
db_database = st.secrets["db_database"]
db_collection = st.secrets["db_collection"]

client = MongoClient(db_uri)


dblist = client.list_database_names()
print("the list of databases are: ", dblist)

if db_database in dblist:
    print("The database exists.")

    db = client[db_database]  # Connect to the specified database

else:
    print("The database does not exist.")

    db = client[db_database]  # Connect to the specified database


# Ensure the collection exists, create it if it doesn't
collection_name = db_collection
if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)

collection = db[collection_name]


def check_known_errors(user_input):
    # Check if the regex pattern matches any document in the MongoDB collection
    matched_errors = []
    for error_doc in collection.find():
        regex_pattern = error_doc["ErrorRegex"]
        if re.search(regex_pattern, user_input, re.IGNORECASE):
            matched_errors.append(error_doc["ErrorSolution"])
    return matched_errors


def add_new_error(user_input, solution):
    # Insert a new document into the MongoDB collection
    error_doc = {"ErrorRegex": user_input, "ErrorSolution": solution}
    collection.insert_one(error_doc)


def if_submit_button_clicked(new_error_input, new_error_solution):

    if new_error_input and new_error_solution:
        add_new_error(new_error_input, new_error_solution)
        st.success("New error added to the database!")

def main():
    # Streamlit app
    st.set_page_config(
        page_title="Error Message Analyzer",
        page_icon="https://lh3.googleusercontent.com/YtXTsa-6SaaMl02-OUo8iRztlX5Thu4aCLavunIV1M5hm9y4ySTPpMjpY44fL4ayz7Se",
    )

    # Containers
    top = st.container()



    # User input for error message
    with top:

        # Add a logo image
        logo_image = 'https://lh3.googleusercontent.com/YtXTsa-6SaaMl02-OUo8iRztlX5Thu4aCLavunIV1M5hm9y4ySTPpMjpY44fL4ayz7Se'  # Replace 'path_to_your_logo_image.png' with the actual path to your logo image

        # Custom header with logo
        header_html = f"""
        <div style="display: flex; align-items: center; padding: 10px;">
            <img src={logo_image} alt="Logo" width="50" height="50" style="margin-right: 10px;">
            <h1>Error Message Analyzer</h1>
        </div>
        """

        st.markdown(header_html, unsafe_allow_html=True)

        # Add title
        # st.title("Error Message Analyzer")

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        st.text_area(
            "Enter your error message:", st.session_state.user_input, key="user_input"
        )

        if st.button("Check Error") or st.session_state.user_input:
            matched_errors = check_known_errors(st.session_state.user_input)
            if matched_errors:
                st.success("Known Error Detected")
                for error_solution, i in zip(
                    matched_errors, range(len(matched_errors))
                ):
                    # Display the error solution in a good format
                    st.write(f"Solution {i+1}:", error_solution)
            else:
                st.header("New Error Detected")
                st.write("This error is not in the database. Would you like to add it?")

                # # get the answer from chatgpt
                # answer = get_chatgpt_answer("please suggest a regex and a solution for this error: " + st.session_state.user_input)

                # # Display the answer
                # st.write("suggested regex and solution: " + answer)

                # add a spinner while waiting for the answer
                with st.spinner(
                    "Please wait while we suggest a regex and a solution for this error ..."
                ):
                    if "chatgpt_answer" not in st.session_state:
                        st.session_state.chatgpt_answer = ""
                        # get the answer from chatgpt

                        try:

                            answer = get_chatgpt_answer(
                            f"""please suggest a regex for the following error and a brief solution to solve the following error:

{st.session_state.user_input}

just provide the two values in the following format:
Regex: <regex>

Solution: <solution>


don't provide any other information in the message, just the regex and the solution. Thank you!"""
                        )
                            st.session_state.chatgpt_answer = answer
                            # print("suggested regex and solution: \n " + st.session_state.chatgpt_answer )
                        except Exception as e:

                            print("error" + str(e))

                            st.write("Error in auto-suggesting a regex and a solution for this error. Please provide a regex and a solution manually.")

                    # Display the answer
                    # st.write("suggested regex and solution: " + answer)



                if "new_error_input" not in st.session_state:
                    st.session_state.new_error_input = ""
                if "new_error_solution" not in st.session_state:
                    st.session_state.new_error_solution = ""

                # try: to extract the regex and solution from the answer

                try:
                    regex = re.search(
                        r"Regex:(.*)\n", st.session_state.chatgpt_answer
                    ).group(1)
                    solution = re.search(
                        r"Solution: (.*)", st.session_state.chatgpt_answer
                    ).group(1)

                    # remove the code block from the regex from the start and end if exists
                    regex = re.sub(r"`", "", regex)

                    st.session_state.new_error_input = regex
                    st.session_state.new_error_solution = solution
                except Exception as e:
                    print("error" + str(e))

                    # print("the chatgpt answer is: ", st.session_state.chatgpt_answer)

                    # st.write("Error in auto-suggesting a regex and a solution for this error. Please provide a regex and a solution manually.")

                    if len(st.session_state.chatgpt_answer) > 0:
                        st.write("suggested regex and solution: \n " + st.session_state.chatgpt_answer)


                    # print("error in extracting the regex and solution from the answer")

                new_error_input = st.text_input(
                    "Error Regex (Edit if necessary):",
                    key="new_error_input",
                )
                new_error_solution = st.text_area(
                    "Error Solution (Provide a solution for the error):",
                    key="new_error_solution",
                )

                if st.button("Submit"):
                    if_submit_button_clicked(new_error_input, new_error_solution)

    st.write("Please reload the page to check another error.")

if __name__ == "__main__":
    main()
