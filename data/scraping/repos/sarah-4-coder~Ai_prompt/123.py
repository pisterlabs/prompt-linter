import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
data = pd.read_csv("Ai_prompt\data.csv")


def plotter(data, plot_type, time_start, time_end, column_name):
    req_data = data[(data['Year'] >= time_start) & (data['Year'] <= time_end)]
    if "point" in plot_type.lower():
        sns.pointplot(x=req_data["Year"], y=req_data[column_name])
    if "bar" in plot_type.lower():
        sns.barplot(x=req_data["Year"], y=req_data[column_name])
    if "pie" in plot_type.lower():
        colors = sns.color_palette('pastel')[0:5]
        plt.pie(req_data["Year"], labels=req_data[column_name], colors=colors)

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title(f'Bar Plot of {column_name}')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig('1.png')

    plt.show()



# Set your OpenAI API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

# Define the prompt
prompt = "Given the following statement, identify the categories for column_name, time_start, time_end, and plot_type:\n\n"\
         "\"Prepare a bar plot for the column agriculture between the time period of 1985 and 1989 from the data.\""

# Call the OpenAI API to get the categories
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=50,  # Adjust as needed to capture the required information
    temperature=0.6,
    stop=None
)
prompt1 = input("Enter prompt")
prompt_fin = "Given the following statement, identify the categories for column_name, time_start, time_end, and plot_type:\n\n" \
             "\"" + prompt1 + '"'

# Call the OpenAI API to get the categories
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt_fin,
    max_tokens=50,  # Adjust as needed to capture the required information
    temperature=0.6,
    stop=None
)

categories = response.choices[0].text.strip().split('\n')
column_name = categories[0][13:]
column_name = column_name.replace(" ", "")
time_start = int(categories[1][12:])
time_end = int(categories[2][10:])
plot_type = categories[3][11:]
plot_type = plot_type.lower()
plot_type = plot_type.replace(" ", "")
if 'plot' not in plot_type:
    plot_type = plot_type + 'plot'

plotter(data, plot_type, time_start, time_end, column_name)