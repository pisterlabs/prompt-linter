import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
def get_ids(drugs):
    for i, d in enumerate(drugs):
        # replace space with +
        drugs[i] = drugs[i].replace(" ", "+")
        # remove any characters after the brackets
        drugs[i] = drugs[i].split("(")[0]

    url = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={name}&search=0"

    ids = []

    for drug in drugs:
        response = requests.get(url.format(name=drug))
        data = response.json()
        try:
            ids.append(data["idGroup"]["rxnormId"][0])
        except:
            pass

    return ids

def get_interactions(ids):
    url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={rxcuis}"

    response = requests.get(url.format(rxcuis="+".join(ids)))
    data = response.json()

    try:
        # pretty print
        pairs = []
        for group in data["fullInteractionTypeGroup"]:
            for interaction in group["fullInteractionType"]:
                for pair in interaction["interactionPair"]:
                    pairs.append({
                        "description": pair["description"],
                        "comment": interaction["comment"],
                        "source": group["sourceName"]
                    })
    except:
        return []

    return pairs, data["fullInteractionTypeGroup"]


def describe_interactions(drugs):
    ids = get_ids(drugs)
    interactions = get_interactions(ids)

    if len(interactions) == 0:
        return {
            "pairs": [],
            "interactionGroup": [],
            "drugs": drugs
        }

    return {
        "pairs": interactions[0],
        "interactionGroup": interactions[1],
        "drugs": drugs
    }


def explain_interactions(interactions):
    drugs = interactions["drugs"]
    pairs = interactions["pairs"]
    interactionGroup = interactions["interactionGroup"]

    if len(pairs) == 0:
        return "No interactions found between {drugs}".format(drugs=" and ".join(drugs))

    human_message = "Interactions between {drugs}:\n\n".format(drugs=" and ".join(drugs))

    for pair in pairs:
        human_message += "Description: {description}\n".format(description=pair["description"])
        human_message += "Comment: {comment}\n".format(comment=pair["comment"])
        human_message += "Source: {source}\n\n".format(source=pair["source"])
        human_message += "------------------\n\n"


    messages = [
        SystemMessage(
            content="You will be given some drugs and interactions between them.\n"
                    "You should explain those interactions (with the outcome result of each interaction ONLY IF POSSIBLE) in an understandable way.\n"
                    "Your answer should be short and concise.\n"
                    "Answer in bullet points.\n"
                    "Your answers will be displayed in app that will be used by pharmacists and patients, so make sure to make your answers clear.\n"
                    "Mention the outcome or effect of each interaction ONLY if possible. Never say that you don't know.\n"
                    "If something is not mentioned, just ignore it.\n"
                    "Never give any advice.\n"
                    "Never skip any interaction.\n"
                    "Never mention any sources.\n"
                    "Never write anything that is not related to the interactions.\n\n"
                    "Be simple, short and precise without losing any important information.\n"
                    "NO SOURCES, NO ADVICE"
        ),
        HumanMessage(
            content=human_message
        )

    ]

    print(human_message)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    explanation = model.predict_messages(messages)

    return explanation.content
