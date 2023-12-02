import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.Model.list()

for model in response["data"]:
    print(model["id"])
 #   if "training_accuracy" in model:
 #       print(f"Accuracy: {model['training_accuracy']:.2f}")
 #   else:
 #       #print("Accuracy information not available.")
 #   if "parameters" in model:
 #       print(f"Params: {model['parameters']}")
 #   else:
 #       print("Parameter information not available.")
    print("-" * 30)