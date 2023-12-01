import csv
import openai
from tqdm import tqdm

# Set up OpenAI API credentials
openai.api_key = ''

# List to store previously generated names
generated_names = []

# Function to predict category based on meal names
def predict_category(meal_data, engine):
    categories = []
    for data in tqdm(meal_data, desc="Naming Mealplans"):
        retries = 0
        while retries < 5:
            try:
                # Include previously generated names in the prompt
                prompt = f"Previously generated names: {', '.join(generated_names)}. Based on the recipes [{data['Breakfast 1']},... what would be a unique and random name for this meal plan?"
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=50,
                    n=1,
                    stop=None,
                )
                predicted_category = response.choices[0]['text'].strip().lower()
                
                # Append the new name to the list
                generated_names.append(predicted_category)
                
                categories.append(predicted_category)
                break  # Break the retry loop if the request is successful
            except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                # Print error and continue to the next set of meal names
                print(f"Error predicting category for '{data['Names']}': {e}")
                categories.append(None)
                break
            except Exception as e:
                # Print error and continue to the next set of meal names
                print(f"An error occurred predicting category for '{data['Names']}': {e}")
                categories.append(None)
                break
        else:
            # If maximum retries reached, append None as the predicted category
            categories.append(None)

    return categories

# Read meal data from CSV
meal_data = []
with open('meal_plans_20.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        meal_data.append(row)

# Predict categories based on meal names
predicted_categories = predict_category(meal_data, "text-davinci-003")

# Update the 'Filter' column in the input CSV
with open('meal_plans_20.csv', 'r') as input_file, open('meal_plans_final_v1_names3.csv', 'w', newline='') as output_file:
    reader = csv.DictReader(input_file)
    fieldnames = reader.fieldnames + ['Names']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for row, category in zip(reader, predicted_categories):
        row['Names'] = category
        writer.writerow(row)
