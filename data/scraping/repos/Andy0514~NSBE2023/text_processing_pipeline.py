import cohere
import toxicity
import toxic_sentence_transformation
import sentiment


co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z')

def process_text(input_string):

    # Use summarize if the input sentence is long
    if len(input_string) > 250:
        response = co.summarize(
            model='summarize-xlarge',
            length='medium',
            format='paragraph',
            temperature=0.3,
            abstractiveness='low'
        ).summary
    else:
        response = input_string

    # Split the input into sentences
    response_arr = response.split(".")
    if (response_arr[-1] == ""):
        response_arr = response_arr[:-1]
    print(response_arr)

    # Run toxicity filter to filter out sentences that are inappropriate
    # or non-inclusive
    tox = toxicity.toxicity_filter(response_arr)

    # For sentences that are toxic, replace them with another sentence.
    for i in range(len(tox)):
        if tox[i][1] == "toxic":
            response_arr[i] = toxic_sentence_transformation.detox(tox[i][0])
        else:
            response_arr[i] = tox[i][0]

    # perform sentiment analysis
    sentence_sentiment = sentiment.sentiment_analysis(response_arr)

    return response_arr, sentence_sentiment





