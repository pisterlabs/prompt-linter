import cohere
from cohere.classify import Example

def get_class(response: str) -> str:
    examples = [Example("I don\'t care", "negative"),
                Example("go to hell", "negative"),
                Example("don\'t be a coward", "negative"),
                Example("you petty little bitch", "negative"),
                Example("why so rude", "neutral"),
                Example("But that\'s not my fault", "neutral"),
                Example("where\'s the classroom", "neutral"),
                Example("thank you very much", "positive"),
                Example("I really appreciate it", "positive"),
                Example("I like your dress", "positive"),
                Example("you did good", "positive"),
                Example("The order came 5 days early", "positive"),
                Example("The item exceeded my expectations", "positive"),
                Example("I ordered more for my friends", "positive"),
                Example("I would buy this again", "positive"),
                Example("I would recommend this to others", "positive"),
                Example("The package was damaged", "negative"),
                Example("The order is 5 days late", "negative"),
                Example("The order was incorrect", "negative"),
                Example("I want to return my item", "negative"),
                Example("The item\'s material feels low quality", "negative"),
                Example("The product was okay", "neutral"),
                Example("I received five items in total", "neutral"),
                Example("I bought it from the website", "neutral"),
                Example("I used the product this morning", "neutral"),
                Example("The product arrived yesterday", "neutral")]
    co = cohere.Client('mDgdZhuZohzzJnVmOnygreJzaxHyvUD4MdPcEoOq')

    response_class = co.classify(
        model='large',
        inputs=[response],
        examples=examples,
    )
    sentence = str(response_class.classifications[0])
    result = sentence.index('prediction: ')
    result_class = sentence[result + 13: result + 21]
    return result_class

def reasonable_answer(class_start: str, class_response: str) -> str:
    if class_start == class_response and class_response != "negative":
        print(f"Your response is {class_response}, "
              f"same as the speaker, this is reasonable.")
    else:
        if class_start == "negative" and class_response == "positive":
            return(f"You are too kind, try speaking up fpr yourself in the future.")
        elif class_start == class_response == "negative":
            return(f"You stood up for yourself. \n"+
            "But maybe there is a more subtle way.")
        elif class_start == "negative" and class_response == "neutral\"":
            return(f"You are a little too kind.")
        elif class_start == "positive" and class_response == "negative":
            return(f"You are too mean! Try to be more polite.")
        elif class_start == "positive" and class_response == "neutral\"":
            return(f"This is reasonable, but maybe you should show more kindness.")
        elif class_start == "neutral\"" and class_response == "positive":
            return(f"You are kind! Good job.")
        elif class_start == "neutral\"" and class_response == "negative":
            return(f"That's not so kind! Be kind to others.")


if __name__ == '__main__':
    start_sentence = "You can’t even get that answer? You’re so dumb!"
    print("You can’t even get that answer? You’re so dumb!")
    class_start = get_class(start_sentence)

    response = input("Enter your response: ")
    result_class = get_class(response)
    print("Your response is", result_class)

    reasonable_answer(class_start, result_class)
