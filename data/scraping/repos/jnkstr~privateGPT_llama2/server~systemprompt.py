from langchain.prompts import PromptTemplate

prompt_template = """Verwenden Sie die folgenden Informationen, um die Frage des Benutzers zu beantworten.
Wenn Sie die Antwort nicht wissen, sagen Sie einfach, dass Sie es nicht wissen, und versuchen Sie nicht, eine Antwort zu erfinden.

Kontext: {context}
Frage: {question}

Geben Sie nur die unten stehende hilfreiche Antwort zurück und sonst nichts.
Hilfreiche Antwort in deutsch:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"])
