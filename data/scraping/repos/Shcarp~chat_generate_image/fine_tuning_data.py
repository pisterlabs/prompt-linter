from openai import OpenAI

api_key = f""

client = OpenAI(api_key=api_key)

MODEL = 'ft:gpt-3.5-turbo-1106:onbrand-inc::8Jyggva9'

def main():
    file = client.files.create(
        file=open('data.jsonl', 'rb'),
        purpose='fine-tune',
    )

    print(file)

    res = client.fine_tuning.jobs.create(
        training_file=file.id,
        model=MODEL,
    )

    print(res)

# ftjob-lgrpxwfbCZ7k31X9p1Qxvtiq

if __name__ == "__main__":
    main()
