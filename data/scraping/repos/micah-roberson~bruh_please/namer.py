import csv
import openai
from tqdm import tqdm

# Set up OpenAI API credentials
openai.api_key = ''

# Function to predict category based on meal names
def predict_category(meal_data, engine):
    categories = []
    for data in tqdm(meal_data, desc="Naming Mealplans"):
        retries = 0
        while retries < 5:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=f"Based on the recipes [{data['Breakfast 1']},{data['Breakfast 2']},{data['Lunch 1']},{data['Lunch 2']},{data['Lunch 3']},{data['Lunch 4']},{data['Dinner 1']},{data['Dinner 2']},{data['Dinner 3']},{data['Dinner 4']}] what would be a unique and random name for this meal plan?",
                    temperature=0.3,
                    max_tokens=50,
                    n=1,
                    stop=None,
                )
                predicted_category = response.choices[0]['text'].strip().lower()
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
with open('meal_plans_20_per_filter.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        meal_data.append(row)

# Predict categories based on meal names
predicted_categories = predict_category(meal_data, "text-davinci-003")

# Update the 'Filter' column in the input CSV
with open('meal_plans_20_per_filter.csv', 'r') as input_file, open('meal_plans_final_v1_names2.csv', 'w', newline='') as output_file:
    reader = csv.DictReader(input_file)
    fieldnames = reader.fieldnames + ['Names']
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()

    for row, category in zip(reader, predicted_categories):
        row['Names'] = category
        writer.writerow(row)