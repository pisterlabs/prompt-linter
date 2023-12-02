import openai


role_prompt = """
You are an expert in medicine with a specific focus on explaining a diagnosis. You know how to analyze a set of sentences to generate an explanation for a medical condition.You will generate an accurate yet constrained explanation based on the information provided.
You will receive the set of diseases and the set of sentences containing relevant information for the explanation. Then you will generate an explanation for why * correct disease * is the most likely disease based on these key reasons. If there is enough information, explain why the other diseases are not the most likely disease.
DO NOT conclude anything outside of the information provided. DO NOT suggest anything. Stick to the information provided in the Information and nothing more. DO NOT infer any new information. Avoid any other response more than the explanation itself. If you are not sure about the explanation, you can say that you are not sure.
"""


def generate_gpt_explanation(diseases, correct_disease, information):
    openai.api_key = "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"

    prompt = f"Diseases: {diseases} \n correct_disease: {correct_disease}, \n information: {information} \n Explain why the symptoms suggest the disease is {correct_disease} rather than the others."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    # explanation = response.choices[0].text.strip()
    explanation = response["choices"][0]["message"]["content"]
    return explanation
