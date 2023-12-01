import cohere

def get_sentiment(transcript):
    co = cohere.Client('hiYAEG7MADWhQuqA2ivVdwoSUNieZ2Sj2gV0ran9') #This is your trial API key
    response = co.classify(model='d39dcfad-05e3-49eb-b3ba-f70a15f3930a-ft', inputs=transcript)
    negative_sentiment_count = 0
    positive_sentiment_count = 0
    negative_sentiment = 0
    positive_sentiment = 0
    ind = 0
    for classification in response.classifications:
        ind += 1
        if classification.prediction == "negative":
            negative_sentiment += classification.confidence
            negative_sentiment_count += 1
        else:
            positive_sentiment += classification.confidence
            positive_sentiment_count += 1

    # print(f"\nnegative sentiment:{negative_sentiment}\npositive sentiment:{positive_sentiment}\n")
    # print(f"\nnegative sentiment count:{negative_sentiment_count}\npositive sentiment count:{positive_sentiment_count}\n")

    total_count = negative_sentiment_count + positive_sentiment_count
    neg_weighted_val = negative_sentiment * negative_sentiment_count / total_count
    pos_weighted_val = positive_sentiment * positive_sentiment_count / total_count

    cumulative_val = (pos_weighted_val - neg_weighted_val) / 100
    compound_sentiment = ((cumulative_val + 1) * 100) / 2

    return round(compound_sentiment)