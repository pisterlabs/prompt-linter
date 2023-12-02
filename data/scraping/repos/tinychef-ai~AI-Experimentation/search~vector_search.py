import json

import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import csv

embeddings = OpenAIEmbeddings()


class RecipeIndex:
    db = None

    # index_name1 and index_name2 are deprecated as they were created using
    # openai first generation embedding models
    # index_name1 = "faiss_index"
    # index_name2 = "faiss_index2"
    index_name3 = "faiss_index3"
    index_name4 = "faiss_index4"
    index_name5 = "faiss_index5"
    index_name6 = "faiss_index6"
    index_name7 = "faiss_index7"
    index_name8 = "faiss_index8"
    index_name9 = "faiss_index9"
    index_name10 = "faiss_index10"
    index_name11 = "faiss_index11"
    index_name12 = "faiss_index12"

    def create12(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = " recipe name: " + df["name"] + \
                              " \n\ncuisines:\n " + df["cuisines"] + \
                              " \n\nmeal types:\n " + df["mealTypes"] + \
                              " \n\ndiet preferences:\n " + df["diet_preferences"] + \
                              " \n\nmajor ingredients:\n " + df["majorIngredients"] + \
                              " \n\ndish types:\n " + df["dishTypes"] + \
                              " \n\ndescriptors:\n " + df["descriptors"] + \
                              " \n\ndescription:\n " + df["description"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name12)

    def create11(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = df["name"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name11)

    def create10(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = df["name"] + \
                              ", " + df["cuisines"] + \
                              ", " + df["mealTypes"] + \
                              ", " + df["dishTypes"] + \
                              ", " + df["descriptors"] + \
                              ", " + df["diet_preferences"] + \
                              ", " + df["majorIngredients"] + \
                              ", " + df["description"]
        df["index_content"] = df["index_content"].str.replace('\n', ' ')
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name10)

    def create9(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = df["name"] + \
                              ", " + df["cuisines"] + \
                              ", " + df["mealTypes"] + \
                              ", " + df["dishTypes"] + \
                              ", " + df["descriptors"] + \
                              ", " + df["diet_preferences"] + \
                              ", " + df["majorIngredients"]
        df["index_content"] = df["index_content"].str.replace('\n', ' ')
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name9)

    def create8(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = df["name"] + \
                              ", " + df["cuisines"] + \
                              ", " + df["mealTypes"] + \
                              ", " + df["dishTypes"] + \
                              ", " + df["descriptors"] + \
                              ", " + df["diet_preferences"] + \
                              ", " + df["majorIngredients"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name8)

    def create7(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = "recipe name: " + df["name"] + \
                              " cuisines: " + df["cuisines"] + \
                              " meal types: " + df["mealTypes"] + \
                              " dist types: " + df["dishTypes"] + \
                              " descriptors: " + df["descriptors"] + \
                              " diet preferences: " + df["diet_preferences"] + \
                              " major ingredients: " + df["majorIngredients"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name7)

    def create6(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = "recipe name: " + df["name"] + \
                              " cuisines: " + df["cuisines"] + \
                              " meal types: " + df["mealTypes"] + \
                              " dist types: " + df["dishTypes"] + \
                              " descriptors: " + df["descriptors"] + \
                              " diet preferences: " + df["diet_preferences"] + \
                              " major ingredients: " + df["majorIngredients"] + \
                              " description: " + df["description"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name6)

    def create5(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[(df["status"] == "active") & (df["languagesSupported"] == "en")]
        df["index_content"] = "recipe name: " + df["name"] + \
                              " cuisines: " + df["cuisines"] + \
                              " meal types: " + df["mealTypes"] + \
                              " diet preferences: " + df["diet_preferences"] + \
                              " major ingredients: " + df["majorIngredients"] + \
                              " description: " + df["description"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name5)

    def create4(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[df["status"] == "active"]
        df["index_content"] = "recipe name: " + df["name"] + \
                              " cuisines: " + df["cuisines"] + \
                              " meal types: " + df["mealTypes"] + \
                              " diet preferences: " + df["diet_preferences"] + \
                              " major ingredients: " + df["majorIngredients"] + \
                              " description: " + df["description"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name4)

    def create3(self, recipe_file):
        df = pd.read_csv(recipe_file)
        df = df[df["status"] == "active"]
        df["index_content"] = df["name"]
        df = df[["name", "index_content"]]

        loader = DataFrameLoader(df, page_content_column="index_content")
        documents = loader.load()

        self.db = FAISS.from_documents(documents, embeddings)
        self.db.save_local(self.index_name3)

    def load(self, index_name):
        self.db = FAISS.load_local(index_name, embeddings)

    def query(self, query):
        if self.db is None:
            raise Exception("Need to create or load index before querying")

        responses = self.db.similarity_search_with_score(query, k=10)
        recipe_names = []
        scores = []
        for response in responses:
            recipe_names.append(response[0].metadata["name"])
            scores.append(response[1])

        return recipe_names, scores


def parse_expanded_queries(expanded_queries_file):
    expanded_queries = {}

    with open(expanded_queries_file) as input_file:
        for line in input_file:
            json_obj = json.loads(line.strip())
            query = json_obj["query"].strip().lower()
            if json_obj["parse_success"]:
                expanded_query = json_obj["expanded_query"]
                expanded_queries[query] = expanded_query

    return expanded_queries


def generate_search_results(index_name, search_result_file1, search_result_file2,
                            expanded_queries_file=None):
    recipe_index = RecipeIndex()
    recipe_index.load(index_name)

    expanded_queries = {}
    if expanded_queries_file is not None:
        expanded_queries = parse_expanded_queries(expanded_queries_file)

    with open(search_result_file1) as input_file:
        with open(search_result_file2, "w") as output_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)

            headers = next(reader, None)
            output_header = ["Search_Text"]
            for i in range(1, 11):
                output_header.append("Result_" + str(i))
            output_header.append("Actual Search Text")

            writer.writerow(output_header)
            for row in reader:
                query = row[0].strip().lower()
                emb_search_query = "recipe search query: " + query
                if query in expanded_queries.keys():
                    emb_search_query += " description: " + expanded_queries[query]

                recipe_names, scores = recipe_index.query(emb_search_query)
                output_row = [query]
                output_row.extend(recipe_names)
                output_row.append(emb_search_query)
                writer.writerow(output_row)


def index_recipes():
    recipe_index = RecipeIndex()
    # recipe_index.create3("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create4("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create5("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create6("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create7("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create8("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create9("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create10("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    # recipe_index.create11("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")
    recipe_index.create12("/Users/agaramit/Downloads/Recipe Tag Dump - Sheet1.csv")


def test_recipe_searches():
    recipe_index = RecipeIndex()
    # recipe_index.load(RecipeIndex.index_name12)
    # query_list = ["vegetarian soups",
    #               "soups",
    #               "vegetarian soup",
    #               "soup",
    #               "south indian breakfast",
    #               "paneer",
    #               "barfee",
    #               "barfi",
    #               "burfi",
    #               "korma",
    #               "veg korma",
    #               "veg kurma",
    #               "veg kurmaa",
    #               "non veg korma",
    #               "bina oven ka pizza",
    #               "non veg dish",
    #               "non vegetarian"]

    # query_list = ["Non veg recipe ideas for a flavorful and diverse range of dishes using meat, poultry, seafood, and eggs",
    #               "Explore a variety of barfee recipes, a delectable Indian sweet confection made with milk, sugar, and various flavorings, offering a delightful treat for any occasion",
    #               "Unleash your culinary creativity with a range of recipes that embody the spirit of 'sher aur kya chahiye,' offering a delightful selection of dishes that go beyond expectations",
    #               "Discover a delightful assortment of recipes inspired by the lively and playful spirit of 'bum bum bole,' offering a burst of flavors and culinary fun for everyone",
    #               "Explore a vibrant collection of recipes to celebrate the festive spirit of Baisakhi, featuring traditional Punjabi dishes and refreshing flavors that capture the essence of this joyous occasion.",
    #               "Embark on a culinary journey inspired by the enchanting lyrics of 'In Aankhon Ki Masti Mein,' and discover a delightful array of recipes that evoke a sense of indulgence and charm"]

    query_list = [
        "A delightful and refreshing recipe featuring the vibrant flavors of citrus and the essence of a bird-like dish with a hint of orange."]

    for query in query_list:
        print(query, recipe_index.query("recipe search query: " + query))


def call_generate_search_results():
    # generate_search_results(RecipeIndex.index_name1,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss_sample.csv")

    # generate_search_results(RecipeIndex.index_name4,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss4_sample.csv")

    # generate_search_results(RecipeIndex.index_name5,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss5_sample.csv")

    # generate_search_results(RecipeIndex.index_name9,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss9_sample.csv")

    # generate_search_results(RecipeIndex.index_name11,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss11_sample.csv")

    # generate_search_results(RecipeIndex.index_name5,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss5_1_sample.csv",
    #                         "recipe search query: ")

    # generate_search_results(RecipeIndex.index_name5,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss5_2_sample.csv",
    #                         "most popular recipes for recipe search query: ")
    #
    generate_search_results(RecipeIndex.index_name5,
                            "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
                            "/Users/agaramit/Downloads/Search Results - openai_faiss5_expq1_sample.csv",
                            "/Users/agaramit/Downloads/expanded_queries.json")

    # generate_search_results(RecipeIndex.index_name12,
    #                         "/Users/agaramit/Downloads/Search Results - Skill_app_searches_1_sample.csv",
    #                         "/Users/agaramit/Downloads/Search Results - openai_faiss12_expq1_sample.csv",
    #                         "/Users/agaramit/Downloads/expanded_queries.json")


if __name__ == "__main__":
    # index_recipes()
    # test_recipe_searches()
    call_generate_search_results()
