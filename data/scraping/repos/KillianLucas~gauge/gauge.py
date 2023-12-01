import replicate
import openai
import requests
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live

def evaluate_answer(query, answer):

    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        max_tokens=1000,
        messages=[
            {
                "role": "system", 
                "content": "You are tasked with evaluating the correctness of an answer to a given query. Consider the context, facts, and logic in your evaluation. Please provide your assessment in the following format: 'Explanation: [Your explanation here]', followed by 'Score: [A score between 0 and 10 to rate the accuracy of the answer]'. For example, 'Explanation: The answer is factually correct. Score: 10'"
            },
            {
                "role": "user", 
                "content": f"The query was: '{query}'. The provided answer was: '{answer}'."
            }
        ]
    )

    text = response["choices"][0]["message"]["content"].strip()
    parts = text.split('Score: ')

    if len(parts) != 2:
        return {"error": "Unexpected response format"}

    explanation = parts[0].replace("Explanation: ", "").strip()
    score = parts[1].strip()

    try:
        score = int(score)
    except ValueError:
        return {"error": "Unexpected score format"}

    return {"explanation": explanation, "score": score}

class Gauge:
    models = [
        {"type": "replicate", "name": "Vicuna", "id": "vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b", "price_per_second": 0.0023},
        {"type": "replicate", "name": "OASST", "id": "oasst-sft-1-pythia-12b:28d1875590308642710f46489b97632c0df55adb5078d43064e1dc0bf68117c3", "price_per_second": 0.0023},
        {"type": "huggingface", "name": "Starcoder", "id": "bigcode/starcoder", "price_per_second": 0.0023},
        {"type": "huggingface", "name": "GPT-4-All", "id": "nomic-ai/gpt4all-j", "price_per_second": 0.0023},
        {"type": "huggingface", "name": "Bloom", "id": "elastic/distilbert-base-uncased-finetuned-conll03-english", "price_per_second": 0.0023},
        {"type": "replicate", "name": "Replit Code", "id": "replit-code-v1-3b:83ac76c4dcf42ecb9a62dc97dd4049cf19007aeaa7571e3272c5d14530db0cf7", "price_per_second": 0.0023},
    ]

    console = Console()

    def run(self, model, query):
        self.console.print(f"[bold yellow]Running model: {model['name']}[/bold yellow]")
        full_output = ""
        start = time.time()
        total_time = 0
        total_cost = 0

        try:
            with Live(self.console, refresh_per_second=4, auto_refresh=False) as live:
                if model["type"] == "replicate":
                    output = replicate.run(
                        f"replicate/{model['id']}",
                        input={"prompt": f"Q: {query}\nA:"}
                    )
                    for item in output:
                        if "Q:" in item:
                            break
                        full_output += item
                        live.update(f"[bold green]Response: {full_output}[/bold green]")
                        live.refresh()

                elif model["type"] == "huggingface":
                    API_URL = f"https://api-inference.huggingface.co/models/{model['id']}"
                    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}

                    response = requests.post(API_URL, headers=headers, json={"inputs": query})
                    full_output = response.json()[0]["generated_text"]
                    live.update(f"[bold green]Response: {full_output}[/bold green]")
                    live.refresh()

                total_time = time.time() - start
                total_cost = model["price_per_second"] * total_time
        except Exception as e:
            self.console.print(f"[bold red]An error occurred: {e}[/bold red]")

        if full_output.strip() == "":
          full_output = "None"

        return full_output, total_time, total_cost

    def evaluate(self, query):
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("Model")
        results_table.add_column("Response")
        results_table.add_column("Score")
        results_table.add_column("Explanation")
        results_table.add_column("Latency")
        results_table.add_column("Cost")

        for model in self.models:
            answer, latency, cost = self.run(model, query)
            self.console.print(f"[bold cyan]Evaluating model: {model['name']}[/bold cyan]")
            
            evaluation = evaluate_answer(query, answer)
            
            if "error" not in evaluation:
                results_table.add_row(
                    model["name"], 
                    answer,
                    str(evaluation["score"]),
                    evaluation["explanation"],
                    str(latency),
                    str(cost)
                )

        self.console.print(results_table)

gauge = Gauge()