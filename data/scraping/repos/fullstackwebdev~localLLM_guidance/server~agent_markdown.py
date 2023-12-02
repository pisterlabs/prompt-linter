# import guidance

valid_answers = ['Action', 'Final Answer']
valid_tools = ['Google Search']

chatgpt_test = """
{{#system~}}
You are a full stack developer. You take in JSON data, and generate Markdown. Including utilizing all the features of markdown in a creative way. The markdown features you know: Headings, Styling text, Quoting text, Quoting code, Supported color models, Links, Section links, Relative links, Images, Lists, Task lists, Mentioning people and teams, Referencing issues and pull requests, Referencing external resources, Uploading assets, Using emoji, Paragraphs, Footnotes
{{~/system}}

{{#user~}}
Question: 
Please generate markdown for the following JSON.  
```js
{
  "title" : "How to change the font size in the Visual Studio Code integrated terminal",
  "keywords": ["Visual Studio Code", "VS Code", "Terminal", "Font Size"],
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

Sure here's the markdown:
```
# How to change the font size in the Visual Studio Code integrated terminal

## Keywords
- Visual Studio Code
- VS Code
- Terminal
- Font Size

## Steps 
- Open Visual Studio Code.
- Press Ctrl + backtick (grave accent) to open the integrated terminal.
- Click on the gear icon located at the top right corner of the terminal panel or right-click inside the terminal.
- In the dropdown menu, click on "Configure Terminal Settings" or "Configure Terminal Profile".
- This will open the settings.json file with the terminal profile settings.
- Look for the "fontSize" property in the JSON file.
- Modify the "fontSize" value to your desired font size. For example, "fontSize": 14.
- Save the settings.json file.
```

{{~/assistant}}

{{#user~}}
Great, good job. Now convert the following JSON into markdown:
```js
{{query}}
```js
{{~/user}}

{{#assistant~}}
Sure, here's the JSON converted to markdown using all the features:
```
{{gen 'resolver' temperature=0.83 max_tokens=700 stop='```'}}
```
{{~/assistant}}
"""

class UniversalAnythingToMarkdown:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_test)
        final_response = prompt_start(query=query)
        history = final_response.__str__()
        return history, final_response['resolver']