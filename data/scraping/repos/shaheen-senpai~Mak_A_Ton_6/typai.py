import os
import pandas as pd
import openai
from dotenv import load_dotenv
from functions import get_primer,format_question,run_request,format_response
import matplotlib.pyplot as plt


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

my_plot = None
path = "movies.csv"
question = "generate a scatter plot of the data." 
df = pd.DataFrame()
primer1,primer2 = "",""





def load_csv():
    global df,primer1,primer2
    df = pd.read_csv(path)
    print(df)
    primer1,primer2 = get_primer(df,"df")
    
def sent_df():
    return df


load_csv()





try:
    # Format the question 
    question_to_ask = format_question(primer1, primer2, question)   
    print(question_to_ask)
    print()
    # Run the question
    answer=""
    answer = run_request(question_to_ask, "gpt-3.5-turbo", key=openai.api_key)
    # the answer is the completed Python script so add to the beginning of the script to it.
    answer = primer2 + answer
    answer = format_response(answer)
    answer = answer + "\nmy_plot = plt.gcf()\n" + "my_plot.savefig('my_plot1.png')\n"
    
    print(answer,"\n end")
    # plot_area = 
    # exec(answer)

    # Write the generated script to a new file
    with open('generated_script.py', 'w') as f:
        f.write(answer)

    # Now execute the generated script from the new file
    os.system('python generated_script.py')
    

    # exec(answer)      
except Exception as e:
    if type(e) == openai.error.APIError:
        print("OpenAI API Error. Please try again a short time later. (" + str(e) + ")")
    elif type(e) == openai.error.Timeout:
        print("OpenAI API Error. Your request timed out. Please try again a short time later. (" + str(e) + ")")
    elif type(e) == openai.error.RateLimitError:
        print("OpenAI API Error. You have exceeded your assigned rate limit. (" + str(e) + ")")
    elif type(e) == openai.error.APIConnectionError:
        print("OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings. (" + str(e) + ")")
    elif type(e) == openai.error.InvalidRequestError:
        print("OpenAI API Error. Your request was malformed or missing required parameters. (" + str(e) + ")")
    elif type(e) == openai.error.AuthenticationError:
        print("Please enter a valid OpenAI API Key. (" + str(e) + ")")
    elif type(e) == openai.error.ServiceUnavailableError:
        print("OpenAI Service is currently unavailable. Please try again a short time later. (" + str(e) + ")")               
    else:
        print("Unfortunately the code generated from the model contained errors and was unable to execute.")

    

