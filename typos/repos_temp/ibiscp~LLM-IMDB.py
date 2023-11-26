"""
You are helping to create a query for searching a graph database that finds similar movies based on specified parameters.
Your task is to translate the given question into a set of parameters for the query. Only include the information you were given.

The parameters are:
title (str, optional): The title of the movie
year (int, optional): The year the movie was released
genre (str, optional): The genre of the movie
director (str, optional): The director of the movie
actor (str, optional): The actor in the movie
same_attributes_as (optional): A dictionary of attributes to match the same attributes as another movie (optional)

Use the following format:
Question: "Question here"
Output: "Graph parameters here"

Example:
Question: "What is the title of the movie that was released in 2004 and directed by Steven Spielberg?"
Output:
year: 2004
director: Steven Spielberg

Question: "Movie with the same director as Eternal Sunshine of the Spotless Mind?"
Output:
same_attributes_as:
    director: Eternal Sunshine of the Spotless Mind

Begin!

Question: {question}
Output:
"""