from langchain.prompts import PromptTemplate



if __name__ == '__main__':
    prompt = PromptTemplate.from_template('Suggest youtube channel name based on company that creates videos on {content}')

    prompt.format(content='AI')
    print(prompt)
