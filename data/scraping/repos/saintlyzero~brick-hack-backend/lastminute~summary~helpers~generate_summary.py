# from data import OOP, OS_CN
from keys import API_KEY
import cohere


def generate_summary(text):
    co = cohere.Client(API_KEY)
    # return generate_summary(co).summary
    return co.summarize(model='summarize-xlarge', text=text, length='long', extractiveness='medium', temperature=0.25)


def main():
    return generate_summary()

if __name__ == '__main__':
    print(main())