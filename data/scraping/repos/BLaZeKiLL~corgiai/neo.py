import random

from langchain.graphs import Neo4jGraph


def connect_neo_graph(url: str, username: str, password: str) -> Neo4jGraph:
    return Neo4jGraph(url=url, username=username, password=password)


def create_vector_index(driver: Neo4jGraph, dimension: int) -> None:
    index_query = "CALL db.index.vector.createNodeIndex('top_questions', 'Question', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass
    index_query = "CALL db.index.vector.createNodeIndex('top_answers', 'Answer', 'embedding', $dimension, 'cosine')"
    try:
        driver.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass


def get_random_question(driver: Neo4jGraph, tag: str) -> str:
    question_query = '''
    MATCH (question:Question) -[r:TAGGED]-> (t:Tag {name:$tag})
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..1] as answers
        RETURN reduce(str='', answer IN answers | str + 
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    } 
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n' + answerTexts AS text
    SKIP $offset LIMIT 1
    '''

    return driver.query(question_query, {
        "tag": tag,
        "offset": random.randint(1, 2000)
    })[0]['text']
    