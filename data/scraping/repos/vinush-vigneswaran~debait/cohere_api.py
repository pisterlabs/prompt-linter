import cohere


def generate(prompt, api_key) -> str:
    '''
    Connects to cohere API and returns AI response given a prompt text.
    '''
    num_tries = 0
    out = ""

    while num_tries < 2:
        try:
            co = cohere.Client(api_key)
            response = co.generate(
                model='large',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                k=0,
                p=0.75,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=["--"],
                return_likelihoods='NONE')
            out = response.generations[0].text
            status = 200
            error_msg = ""
            break

        except cohere.CohereError as e:
            status = e.http_status
            error_msg = e.message
            #header = e.headers
            num_tries += 1
    return status, error_msg, out

def classify(input, api_key):
    '''
    Connects to cohere API and returns fine-tuned utils classification
    '''
    max = 0
    num_tries = 0
    while num_tries < 2:
        try:
            co = cohere.Client(api_key)
            response = co.classify(
                model='9e2e2d1c-2c28-466c-8302-9c69dad99124-ft',
                inputs=[input])

            confidence = response.classifications[0].confidence
            for data in confidence:
                if data.confidence > max:
                    max = data.confidence
                    best_label = data.label
            status = 200
            error_msg = ""
            break

        except cohere.CohereError as e:
            status = e.http_status
            error_msg = e.message
            best_label = "Call to API failed"
            #header = e.headers
            num_tries += 1

    return status, error_msg, best_label


# def classify(input, api_key):
#     '''
#     Connects to cohere API and returns fine-tuned utils classification
#     '''
#     max = 0
#     co = cohere.Client(api_key)
#     response = co.classify(
#         utils='9e2e2d1c-2c28-466c-8302-9c69dad99124-ft',
#         inputs=[input])
#
#     confidence = response.classifications[0].confidence
#     for data in confidence:
#         if data.confidence > max:
#             max = data.confidence
#             best_label = data.label
#
#
#     return best_label

# out = classify("high ideals are all well and good, but not when they come at the expense of the present. Our world is marred by war, famine, and poverty; billions of people are struggling simply to live from day to day. Our dreams of exploring space are a luxury they cannot afford!")
# print(out)