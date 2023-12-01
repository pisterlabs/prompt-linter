import pandas as pd
import openai

# Load the data
# add correct file name and sheet name
df = pd.read_excel('filename.xlsx', sheet_name='sheet name')
print(df.head())

# Define the range of rows to process
start_row = 0  # Change this to the index of the first row you want to process
end_row = 75  # Change this to the index of the last row you want to process

# Slice the dataframe to only include the rows you want to process
df = df.iloc[start_row:end_row]

# Initialize OpenAI API
openai.api_key = 'add key here'

# Function to extract keywords and assign to categories
def extract_keywords_and_assign_to_categories(feedback):
    keywords = extract_keywords(feedback)
    for category in categories:
        if any(keyword in keywords for keyword in category['keywords']):
            category['feedback'].append(feedback)
            return
    # If no match found, create a new category
    new_category = {'keywords': keywords, 'feedback': [feedback]}
    categories.append(new_category)

# Function to extract keywords using AI model
def extract_keywords(feedback):
    prompt = f"This is a piece of feedback from a student: \"{feedback}\". Please identify the main suggestion or theme in this feedback and express it as a single word or short phrase."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=60
    )
    return response.choices[0].text.strip()

# Initialize an empty list to store categories
categories = []

# Process each row and store the feedback in categories
for index, row in df.iterrows():
    feedback = row['Start']  # Replace 'Start' with the column containing feedback
    extract_keywords_and_assign_to_categories(feedback)

# Print the categories and their feedback
for category in categories:
    print("Category Keywords:", category['keywords'])
    print("Feedback:")
    for feedback in category['feedback']:
        print(feedback)
    print("-" * 50)

# Create a new dataframe to store the categorized feedback
categorized_feedback_df = pd.DataFrame(columns=['Category', 'Feedback'])

# Add the categorized feedback to the dataframe
for category in categories:
    category_name = ', '.join(category['keywords'])
    feedback_list = category['feedback']
    for feedback in feedback_list:
        categorized_feedback_df = pd.concat([categorized_feedback_df, pd.DataFrame({'Category': [category_name], 'Feedback': [feedback]})])

# Save the categorized feedback to an Excel file
# Choose created file name and sheet name
with pd.ExcelWriter('file name.xlsx') as writer:
    categorized_feedback_df.to_excel(writer, sheet_name='sheet name')
