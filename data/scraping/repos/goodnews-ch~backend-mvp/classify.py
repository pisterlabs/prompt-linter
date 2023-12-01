import cohere 
from cohere.classify import Example 
co = cohere.Client('3Gadb4V5oKd2YIwc6rz7Oxw6LOYFTxFSbg0nxy7k')

TOXIC_WEIGHT = 3
FINETUNED_WEIGHT = 0.75
FINANCE_WEIGHT = 1

def calculate_score(text_input):
    toxicity_conf, finetuned_conf, finance_conf = find_confidences(text_input)
    return TOXIC_WEIGHT * toxicity_conf['TOXIC'] + FINETUNED_WEIGHT * finetuned_conf['0'] + FINANCE_WEIGHT * finance_conf['NEGATIVE']

def find_confidences(text_input):

    response_toxicity = co.classify( 
    model='cohere-toxicity', 
    inputs=[text_input])

    response_finetuned = co.classify(
        model='223de49b-5243-4a5d-97ef-bfd04baba559-ft',
        inputs=[text_input]
    )

    response_finance = co.classify(
        model='finance-sentiment',
        inputs=[text_input]
    )

    toxicity_conf = populate_dict(response_toxicity)
    finetuned_conf = populate_dict(response_finetuned)
    finance_conf = populate_dict(response_finance)

    return (toxicity_conf, finetuned_conf, finance_conf)


def populate_dict(response):
    conf = dict()
    conf[response.classifications[0].confidence[0].label] = response.classifications[0].confidence[0].confidence
    conf[response.classifications[0].confidence[1].label] = response.classifications[0].confidence[1].confidence
    return conf
