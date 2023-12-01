import marimo

__generated_with = "0.1.39"
app = marimo.App()


@app.cell
def first(mo):
    mo.md("#Entrepreneur First Portfolio Company Analysis")
    return


@app.cell
def __(mo):
    mo.md(
        "This notebook is an exploration & analysis of the portfolio companies of [Entrepreneur First](https://www.joinef.com/), by Jeremy Hadfield."
    )
    return


@app.cell
def search(mo):
    # Add a label and a text input widget for regular search
    regular_search_label = mo.md("## Keyword Search\nFind a company using a keyword")
    regular_search_label
    return (regular_search_label,)


@app.cell
def __(mo):
    search_box = mo.ui.text(value="")
    search_box
    return (search_box,)


@app.cell
def initial_analysis(mo, pd, search_box):
    # Load the csv file
    df = pd.read_csv("portcos.csv")

    if search_box == "":
        df
    else:
        # Filter the dataframe based on the search input
        df = df[
            df.apply(
                lambda row: row.astype(str).str.contains(search_box.value).any(), axis=1
            )
        ]

    # Display the csv file as a table
    csv_file = mo.ui.table(df)
    csv_file
    return csv_file, df


@app.cell
def __(mo):
    # Add a label and a text input widget for semantic search
    semantic_search_label = mo.md(
        "## Semantic Search\n Just describe a company or a concept in natural language!"
    )
    semantic_search_label
    return (semantic_search_label,)


@app.cell
def __(mo):
    semantic_search_box = mo.ui.text(value="")
    semantic_search_box
    return (semantic_search_box,)


@app.cell
def __(cosine_similarity, df, embeddings_model, np, semantic_search_box):
    # Implement semantic search over the company summaries
    _df = df.copy()
    _df["combined_text"] = _df[
        ["Company Name", "Summary", "Category1", "Category2", "Category3"]
    ].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    # Apply the embeddings model to the 'combined_text' column
    #! this results in an openaai call
    # use the glove vectors embeddings model
    _df["summary_embeddings"] = embeddings_model.embed_documents(
        _df["combined_text"].tolist()
    )

    # Get the query from the text, embed the documents given the query
    query = semantic_search_box.value
    query_embedding = embeddings_model.embed_documents([query])

    # Calculate similarity
    similarity_scores = cosine_similarity(
        np.array(_df["summary_embeddings"].tolist()),
        np.array(query_embedding).reshape(1, -1),
    )

    # Rank companies
    _df["similarity"] = similarity_scores.flatten()
    _df_sorted = _df.sort_values(by="similarity", ascending=False)

    # Return top 'num_companies' results
    _num_companies = 5
    top_companies = _df_sorted.head(_num_companies)
    # drop summary embeddings and combined text at the end
    top_companies = top_companies.drop(columns=["summary_embeddings", "combined_text"])
    top_companies
    return query, query_embedding, similarity_scores, top_companies


@app.cell
def load_and_display_data(alt, df, mo):
    # melt the dataframe to create a company-category pair for each category
    _df_melted = df.melt(
        id_vars="Company Name", value_vars=["Category1", "Category2", "Category3"]
    )

    # create the bar chart
    chart = (
        alt.Chart(_df_melted, title="Company Category Frequency")
        .transform_calculate(category="datum.value")
        .transform_aggregate(frequency="count()", groupby=["category"])
        .transform_window(
            rank="rank(frequency)",
            sort=[alt.SortField("frequency", order="descending")],
        )
        .transform_filter("datum.rank <= 30")  # Select only the top 30 categories
        .mark_bar()
        .encode(
            x=alt.X("category:N", title="Category", sort="-y"),
            y=alt.Y("frequency:Q", title="Frequency"),
            color="category:N",
        )
    )

    barchart = mo.ui.altair_chart(chart)

    barchart

    return barchart, chart


@app.cell
def __(alt, df, mo):
    # Melt the dataframe to create a company-category pair for each category
    _df_melted = df.melt(
        id_vars="Company Name", value_vars=["Category1", "Category2", "Category3"]
    )

    # Create the pie chart
    _chart = (
        alt.Chart(_df_melted)
        .mark_arc(innerRadius=50)
        .encode(
            alt.Theta("count()", stack=True),
            alt.Color("value:N", legend=alt.Legend(title="Category")),
        )
        .properties(width=400, height=400)
    )

    piechart = mo.ui.altair_chart(_chart)
    piechart
    return (piechart,)


@app.cell
def visualize_embeddings(TSNE, alt, df, embeddings_model, mo, np, os):
    # do embeddings exist already in a file? if so, load them; if not, create
    _df = df.copy()
    if os.path.exists("embeddings.npy"):
        embeddings_array = np.load("embeddings.npy")
    else:
        # Embed the company summaries
        _df["summary_embeddings"] = embeddings_model.embed_documents(
            _df["Summary"].tolist()
        )

        # Convert the list of embeddings to a numpy array
        embeddings_array = np.array(_df["summary_embeddings"].tolist())

        # Save the embeddings to a file
        np.save("embeddings.npy", embeddings_array)

    # Create a TSNE object
    _tsne = TSNE(n_components=2)

    # Reduce the dimensionality of the embeddings to 2D
    _df[["x", "y"]] = _tsne.fit_transform(embeddings_array)

    # Save the embeddings to a file
    np.save("embeddings.npy", embeddings_array)

    # Calculate the distance from the origin for each point
    _df["distance"] = np.sqrt(_df["x"] ** 2 + _df["y"] ** 2)

    # Create a scatter plot of the 2D embeddings
    embeddings_chart = (
        alt.Chart(_df)
        .mark_circle()
        .encode(
            x="x",
            y="y",
            color=alt.Color("distance:Q", scale=alt.Scale(scheme="plasma")),
            tooltip=["Company Name", "Summary"],
        )
        .properties(title="Visualization of Company Summary Embedding Space")
    )

    # Display the scatter plot
    scatter_plot = mo.ui.altair_chart(embeddings_chart)
    scatter_plot
    return embeddings_array, embeddings_chart, scatter_plot


@app.cell
def __(KMeans, TSNE, df, embeddings_array, px):
    # copy dataframe
    _df = df.copy()
    # Reduce the dimensionality of the embeddings to 3D
    _tsne = TSNE(n_components=3)
    # to the left (higher on d1) is more hardware; to the right is more software-y
    dimension_1 = "d1 (Hardware-y)"
    # more d2 seems to be connected to biomedical/biotech startups
    dimension_2 = "d2 (Biomedical-y)"
    # higher d3 = more AI-like, lower = climate tech related?
    dimension_3 = "d3 (AI-y)"
    _df[[dimension_1, dimension_2, dimension_3]] = _tsne.fit_transform(embeddings_array)

    # Perform clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    _df["cluster"] = kmeans.fit_predict(embeddings_array)

    # Create a 3D scatter plot of the embeddings
    fig = px.scatter_3d(
        _df,
        x=dimension_1,
        y=dimension_2,
        z=dimension_3,
        labels={"x": dimension_1, "y": dimension_2, "z": dimension_3},
        color="cluster",
        title="Visualization of Company Summary Embedding Space",
        hover_data=["Company Name", "Summary"],  # Only show company name and summary
    )

    # Display the plot inline
    fig
    return dimension_1, dimension_2, dimension_3, fig, kmeans


@app.cell
def imports():
    import os

    OAI_API_KEY = os.environ["OPENAI_API_KEY"]

    import altair as alt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import torch
    from langchain.embeddings import OpenAIEmbeddings
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    from transformers import pipeline

    embeddings_model = OpenAIEmbeddings(openai_api_key=OAI_API_KEY)
    import marimo as mo

    return (
        KMeans,
        NearestNeighbors,
        OAI_API_KEY,
        OpenAIEmbeddings,
        TSNE,
        alt,
        cosine_similarity,
        embeddings_model,
        mo,
        np,
        os,
        pd,
        pipeline,
        plt,
        px,
        torch,
    )


if __name__ == "__main__":
    app.run()
