import cohere
import streamlit as st

co = cohere.Client('4SXKn5PZBWm6elQFZ2JCum7JchNaqxyCG2F9y2mz')



def mental_support(problem):
    text_file = open("training.txt", "r")
    data = text_file.read()

    response = co.generate(
        model='xlarge',
        prompt=data + " " + problem + "\nStrategy: ",
        max_tokens=100,
        temperature=1,
        k=0,
        p=0.7,
        frequency_penalty=0.1,
        presence_penalty=0,
        stop_sequences=["--"])
    # st.write('Predict: {}'.format(response.generations[0].text))

    response = response.generations[0].text
    response = response.replace("\n\n--", "").replace("\n--", "").strip()
    st.write(response)
    return response


def main():

    problem = st.text_input('What seems to be the problem?\n', '')
    # problem = input("Enter problem:")
    response = mental_support(problem)
    print(response)


if __name__ == "__main__":
    main()