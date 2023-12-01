# imports
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
import os
import openai
import numpy as np
from openai.embeddings_utils import (
    get_embedding,
    cosine_similarity,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
from database import load_courses_from_db, load_last_viewed_courses_from_db
from flask import session

# Set your OpenAI API key here
openai.api_key = os.environ['OpenAi_API']

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"

# Database connection setup
db_connection_string = os.environ['DB_CONNECTION_STRING']

engine = create_engine(
    db_connection_string,
    connect_args={
        "ssl": {
            "ssl_ca": "/etc/ssl/cert.pem"
        }
    }
)

courses_dict = load_courses_from_db()
courses_df = pd.DataFrame(courses_dict)

# Use the SQLAlchemy engine to execute the query and retrieve the data
with engine.connect() as conn:
    result_1 = conn.execute(text("SELECT embedding FROM courses LIMIT 427"))

    # Convert the result to a list of embedding strings
    embedding_strings_1 = [row[0] for row in result_1]

# Convert the embedding strings to lists of floats
embeddings_list_of_lists = []
for embedding_str in embedding_strings_1:
    # Assuming the embeddings are stored as space-separated values in the 'embedding' column
    embedding_values_1 = [float(value) for value in embedding_str.split()]
    embeddings_list_of_lists.append(embedding_values_1)

with engine.connect() as conn:
    result_2 = conn.execute(text("SELECT embedding FROM courses LIMIT 427, 427"))

    # Convert the result to a list of embedding strings
    embedding_strings_2 = [row[0] for row in result_2]

for embedding_str in embedding_strings_2:
    # Assuming the embeddings are stored as space-separated values in the 'embedding' column
    embedding_values_2 = [float(value) for value in embedding_str.split()]
    embeddings_list_of_lists.append(embedding_values_2)

with engine.connect() as conn:
    result_3 = conn.execute(text("SELECT embedding FROM courses LIMIT 854, 427"))

    # Convert the result to a list of embedding strings
    embedding_strings_3 = [row[0] for row in result_3]

for embedding_str in embedding_strings_3:
    # Assuming the embeddings are stored as space-separated values in the 'embedding' column
    embedding_values_3 = [float(value) for value in embedding_str.split()]
    embeddings_list_of_lists.append(embedding_values_3)

with engine.connect() as conn:
    result_4 = conn.execute(text("SELECT embedding FROM courses LIMIT 1281, 427"))

    # Convert the result to a list of embedding strings
    embedding_strings_4 = [row[0] for row in result_4]

for embedding_str in embedding_strings_4:
    # Assuming the embeddings are stored as space-separated values in the 'embedding' column
    embedding_values_4 = [float(value) for value in embedding_str.split()]
    embeddings_list_of_lists.append(embedding_values_4)


def print_recommendations_from_strings():
    session_id = session.get('session_id')
    course_codes = courses_df["course_code"].tolist()

    last_viewed_courses = load_last_viewed_courses_from_db()
    if last_viewed_courses:
        last_viewed_course_codes = [course['course_code'] for course in last_viewed_courses]

    # get embeddings for all strings
    embeddings = embeddings_list_of_lists

    # Check for empty embeddings and remove them
    valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) > 0]
    embeddings = [emb for emb in embeddings if len(emb) > 0]
    course_codes = [course_codes[i] for i in valid_indices]

    with engine.connect() as conn:
        query = text("""
            SELECT `course_code`
            FROM `sessions`
            WHERE `ID` = :session_id
                AND (`activity` = 'clicked' OR `activity` = 'favorited')
            ORDER BY `timestamp` DESC
            LIMIT 1;
        """)

        result = conn.execute(query, {"session_id":session_id})
        row = result.fetchone()

    if row is None:
        # Handle the case where no data was found
        return []

    starting_course_1 = row[0]

    index_of_source_string_1 = courses_df[courses_df["course_code"] == starting_course_1].index[0]

    last_viewed_indices = courses_df[courses_df['course_code'].isin(last_viewed_course_codes)].index
    last_viewed_indices_set = set(last_viewed_indices)

    # get the embedding of the source string
    query_embedding_1 = embeddings[index_of_source_string_1]
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding_1, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors_1 = indices_of_nearest_neighbors_from_distances(distances)
    indices_of_nearest_neighbors_1 = [index for index in indices_of_nearest_neighbors_1 if index not in last_viewed_indices_set]
    indices_of_5_nearest_neighbors_1 = indices_of_nearest_neighbors_1[:9]

    course_codes_of_nearest_neighbors_1 = [course_codes[i] for i in indices_of_5_nearest_neighbors_1]

    #with engine.connect() as conn:
    #    query = text("""
    #        SELECT `course_code`
    #        FROM `sessions`
    #        WHERE `ID` = :session_id
    #            AND (`activity` = 'clicked' OR `activity` = 'favorited')
    #        ORDER BY `timestamp` DESC
    #        LIMIT 1,1;
    #    """)
#
    #    result = conn.execute(query, {"session_id":session_id})
    #    row = result.fetchone()
#
    #if row is None:
    #    # Handle the case where no data was found
    #    course_codes_of_nearest_neighbors_1 = course_codes_of_nearest_neighbors_1
    #    course_codes_tuple_1 = tuple(course_codes_of_nearest_neighbors_1)
#
    #    def load_similar_courses_from_db():
    #        with engine.connect() as conn:
    #            query = "SELECT course_name, course_code, language, aims, content, Degree, ECTS, school, tests, block, lecturers FROM courses WHERE course_code IN :similar_course_codes ORDER BY CASE"
    #    
    #            for i, code in enumerate(course_codes_of_nearest_neighbors_1, start=1):
    #                query += f" WHEN :code{i} THEN {i}"
    #                    
    #            query += " END"
    #                    
    #            # Execute the dynamically generated query
    #            query_params = {'similar_course_codes': course_codes_tuple_1}
    #            query_params.update({f'code{i}': code for i, code in enumerate(course_codes_of_nearest_neighbors_1, start=1)})
    #                    
    #            result = conn.execute(text(query), query_params)
    #            courses = []
    #            columns = result.keys()
    #            for row in result:
    #                result_dict = {column: value for column, value in zip(columns, row)}
    #                courses.append(result_dict)
    #            return courses
#
    #    similar_courses_1 = load_similar_courses_from_db()
    #    return similar_courses_1
#
    #starting_course_2 = row[0]
#
    #index_of_source_string_2 = courses_df[courses_df["course_code"] == starting_course_2].index[0]
#
    ## get the embedding of the source string
    #query_embedding_2 = embeddings[index_of_source_string_2]
    ## get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    #distances = distances_from_embeddings(query_embedding_2, embeddings, distance_metric="cosine")
    ## get indices of nearest neighbors (function from embeddings_utils.py)
    #indices_of_nearest_neighbors_2 = indices_of_nearest_neighbors_from_distances(distances)
    #indices_of_nearest_neighbors_2 = [index for index in indices_of_nearest_neighbors_2 if index not in last_viewed_indices_set]
    #indices_of_4_nearest_neighbors_2 = indices_of_nearest_neighbors_2[:4]
#
    #course_codes_of_nearest_neighbors_2 = [course_codes[i] for i in indices_of_4_nearest_neighbors_2]

    course_codes_of_nearest_neighbors = course_codes_of_nearest_neighbors_1 #+ course_codes_of_nearest_neighbors_2

    course_codes_tuple = tuple(course_codes_of_nearest_neighbors)

    def load_similar_courses_from_db():
        with engine.connect() as conn:
            query = "SELECT course_name, course_code, language, aims, content, Degree, ECTS, school, tests, block, lecturers FROM courses WHERE course_code IN :similar_course_codes ORDER BY CASE"
    
            for i, code in enumerate(course_codes_of_nearest_neighbors, start=1):
                query += f" WHEN :code{i} THEN {i}"
                    
            query += " END"
                    
            # Execute the dynamically generated query
            query_params = {'similar_course_codes': course_codes_tuple}
            query_params.update({f'code{i}': code for i, code in enumerate(course_codes_of_nearest_neighbors, start=1)})
                    
            result = conn.execute(text(query), query_params)
            courses = []
            columns = result.keys()
            for row in result:
                result_dict = {column: value for column, value in zip(columns, row)}
                courses.append(result_dict)
            return courses

    similar_courses = load_similar_courses_from_db()

    return similar_courses

def ai_search_results(query):
    query_embedding = get_embedding(query, engine=EMBEDDING_MODEL)
    course_embeddings = embeddings_list_of_lists
    course_codes = courses_df["course_code"].tolist()

    df = pd.DataFrame({'embeddings': course_embeddings})

    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))

    top_similarity_row_numbers = df.nlargest(6, 'similarity').index.tolist()

    # find courses based on the indexes
    similar_course_codes = [course_codes[i] for i in top_similarity_row_numbers]
    similar_course_codes_tuple = tuple(similar_course_codes)

    def load_search_courses_from_db():
            with engine.connect() as conn:
                query = "SELECT course_name, course_code, language, aims, content, Degree, ECTS, school, tests, block, lecturers FROM courses WHERE course_code IN :similar_course_codes ORDER BY CASE"
        
                for i, code in enumerate(similar_course_codes, start=1):
                    query += f" WHEN :code{i} THEN {i}"
                
                query += " END"
                
                # Execute the dynamically generated query
                query_params = {'similar_course_codes': similar_course_codes_tuple}
                query_params.update({f'code{i}': code for i, code in enumerate(similar_course_codes, start=1)})
                
                result = conn.execute(text(query), query_params)
                courses = []
                columns = result.keys()
                for row in result:
                    result_dict = {column: value for column, value in zip(columns, row)}
                    courses.append(result_dict)
                return courses

    similar_courses = load_search_courses_from_db()
    return similar_courses