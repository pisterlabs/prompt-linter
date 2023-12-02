# import guidance

valid_answers = ['Action', 'Final Answer']
valid_tools = ['Google Search']

chatgpt_test = """
{{#system~}}
You are a full stack developer. You take in universal data and generate a JSON schema.
{{~/system}}

{{#user~}}
Question: 
Please generate a Zod Schema for the following JSON.  
```js
{
  "steps": [
    "Open Visual Studio Code.",
    "Press `Ctrl + backtick (grave accent)` to open the integrated terminal.",
    "Click on the gear icon located at the top right corner of the terminal panel or right-click inside the terminal.",
    "In the dropdown menu, click on \"Configure Terminal Settings\" or \"Configure Terminal Profile\".",
    "This will open the `settings.json` file with the terminal profile settings.",
    "Look for the \"fontSize\" property in the JSON file.",
    "Modify the \"fontSize\" value to your desired font size. For example, \"fontSize\": 14.",
    "Save the `settings.json` file."
  ]
}
```

{{~/user}}

{{#assistant~}}
```js
const z = require('zod');

// Define the schema
const schema = z.object({
  name: z.string().min(3).max(50),
  age: z.number().int().min(0).max(150),
  email: z.string().email(),
  address: z.string().nullable(),
});
```

{{~/assistant}}

{{#user~}}
Convert the following into JSON:
{{query}}
{{~/user}}

{{#assistant~}}
Sure, here's the data converted to JSON
```js
{{gen 'resolver' temperature=0.83 max_tokens=500 stop='Conclusion:'}}
```
{{~/assistant}}
"""

class UniversalAnythingToJSON:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_test)
        final_response = prompt_start(query=query)
        history = final_response.__str__()
        return history, final_response['resolver']