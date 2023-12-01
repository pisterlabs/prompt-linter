from langchain.graphs import Neo4jGraph

from langchain.schema.embeddings import Embeddings

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


def create_constraints(driver: Neo4jGraph) -> None:
    driver.query(
        "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE (q.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT answer_id IF NOT EXISTS FOR (a:Answer) REQUIRE (a.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE (u.id) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE (t.name) IS UNIQUE"
    )
    driver.query(
        "CREATE CONSTRAINT site_name IF NOT EXISTS FOR (s:Site) REQUIRE (s.name) IS UNIQUE"
    )


def insert_data(data: dict, embeddings: Embeddings, driver: Neo4jGraph) -> None:

    count = len(data["items"])

    print(f'Generating embeddings for {count} items')

    i = 0

    for question in data["items"]:
        question_text = f'{question["title"]}\n{question["body_markdown"]}'

        question["embedding"] = embeddings.embed_query(question_text)

        for answer in question["answers"]:
            # We include the question text as we want vectors generated for answers against the questions I guess
            answer_text = f'{question_text}\n{answer["body_markdown"]}'

            answer["embedding"] = embeddings.embed_query(answer_text)
        
        i += 1

        print(f'Embedding progress : {i}/{count}')

    import_query = '''
    UNWIND $questions AS q
    MERGE (question:Question {id:q.question_id}) 
    ON CREATE SET question.title = q.title, question.link = q.link, question.score = q.score,
        question.favorite_count = q.favorite_count, question.creation_date = datetime({epochSeconds: q.creation_date}),
        question.body = q.body_markdown, question.embedding = q.embedding
    MERGE (site:Site {name:$site})
    MERGE (question)-[:SITE]->(site)
    FOREACH (tagName IN q.tags | 
        MERGE (tag:Tag {name:tagName}) 
        MERGE (question)-[:TAGGED]->(tag)
    )
    FOREACH (a IN q.answers |
        MERGE (question)<-[:ANSWERS]-(answer:Answer {id:a.answer_id})
        SET answer.is_accepted = a.is_accepted,
            answer.score = a.score,
            answer.creation_date = datetime({epochSeconds:a.creation_date}),
            answer.body = a.body_markdown,
            answer.embedding = a.embedding
        MERGE (answerer:User {id:coalesce(a.owner.user_id, "deleted")}) 
        ON CREATE SET answerer.display_name = a.owner.display_name,
                      answerer.reputation= a.owner.reputation
        MERGE (answer)<-[:PROVIDED]-(answerer)
    )
    WITH * WHERE NOT q.owner.user_id IS NULL
    MERGE (owner:User {id:q.owner.user_id})
    ON CREATE SET owner.display_name = q.owner.display_name,
                  owner.reputation = q.owner.reputation
    MERGE (owner)-[:ASKED]->(question)
    '''

    driver.query(import_query, {
        "questions": data["items"],
        "site": data["site"]
    })