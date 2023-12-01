def warehouse_course_requisites(course_id: str):
    from src.utils.init_firestore import init_firestore

    db = init_firestore()

    course_doc = db.collection("courses").document(course_id).get()
    course_prerequisites_raw = course_doc.to_dict()["prerequisites_raw"]

    if not course_prerequisites_raw or len(course_prerequisites_raw) == 0:
        return False

    requisites = __structure_course_requisites(course_prerequisites_raw)
    if requisites is None:
        return False

    course_doc.reference.set({
        "prerequisites": requisites
    }, merge=True)

    return True


def __structure_course_requisites(requisites: str):
    import openai
    openai.api_key = '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'

    try:
        with open("src/prompts/extract_requisites", "r") as prompt_file:
            requisites_prompt = prompt_file.read()

        # Prompt the model
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": requisites_prompt},
                {"role": "user", "content": requisites}
            ],
            temperature=0
        )

        # Assuming the response is in text form and is valid YAML
        json_response = response.choices[0].message.content
        json_response = json_response.replace("```json", "").replace("```", "")

        import json
        # Parse the YAML response
        parsed_struct = json.loads(json_response)

        if "prerequisites" not in parsed_struct:
            return None

        return parsed_struct["prerequisites"]

    except Exception as e:
        print(e)
        return None
