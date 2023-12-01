# gradio gr_app.py
# open http://localhost:7860
import gradio as gr

from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
import os

google_api_key = os.getenv("GOOGLE_API_KEY")
llm = GooglePalm(google_api_key=google_api_key, temperature=0.0)

TEMPLATE = """
You are an experienced technical writer able to explain complicated systems in simple words.
Improve the documentation below. Return the result as markdown. Add context and improve description too:


Documentation:
```
{text}
```
"""


def handle(template, byte_string):
    text = byte_string.decode("utf-8")
    # Create a LangChain prompt template that we can insert values to later
    prompt = PromptTemplate(input_variables=["text"], template=template)
    final_prompt = prompt.format(text=text)
    return text, llm(final_prompt)


demo = gr.Interface(
    fn=handle,
    inputs=[
        gr.Textbox(
            lines=2, value=TEMPLATE, placeholder="Prompt here...", label="Prompt"
        ),
        gr.File(
            file_count="single",
            file_types=[".md"],
            container=True,
            show_label=True,
            type="binary",
        ),
    ],
    outputs=[
        gr.Markdown(label="Original"),
        gr.Code(label="Output", language="markdown", interactive=True),
    ],
)

if __name__ == "__main__":
    demo.launch(show_api=False)
