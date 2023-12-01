import openai

print("Script started")

openai.api_key = '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'

try:
    fine_tunes = openai.FineTune.list()
    print("Fetched fine-tune jobs")

    if fine_tunes and fine_tunes.get('data'):
        for fine_tune in fine_tunes['data']:
            print(f"Fine-tune ID: {fine_tune['id']}, Status: {fine_tune.get('status')}, Model: {fine_tune.get('model')}, Fine-tuned Model: {fine_tune.get('fine_tuned_model')}, Created At: {fine_tune.get('created_at')}")
    else:
        print("No fine-tune jobs found")

except Exception as e:
    print(f"An error occurred: {e}")
