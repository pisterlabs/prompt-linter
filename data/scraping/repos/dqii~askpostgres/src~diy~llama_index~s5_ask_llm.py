import os
import openai
from .s4_find_relevant import find_relevant

openai.api_key = os.environ['OPENAI_API_KEY']


SYSTEM_PROMPT = """
You are a Q&A assistant for developers using LlamaIndex, a data framework for LLM applications to ingest, structure, and access private or domain-specific data.
"""


def ask_llm(question):
    relevant_chunks = find_relevant(question)
    user_prompt = "\n\n".join([
        question,
        "Here are some pieces of code that I thought might be relevant in helping answer this question.",
        "\n---\n".join(relevant_chunks)
    ])
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    return completion.choices[0].message.content


print(ask_llm("How can I get started with LlamaIndex?"))

"""
askpostgres % python3 -m src.llama_index.s5_ask_llm
To get started with LlamaIndex, you can follow these steps:

1. Install LlamaIndex: Start by installing LlamaIndex using pip:

   ```
   pip install llama-index
   ```

2. Import the necessary modules: Import the required modules in your Python script or notebook:

   ```python
   from llama_index import LlamaIndex, LlamaDataset
   ```

3. Initialize LlamaIndex: Create an instance of LlamaIndex, passing the path to your data directory as an argument:

   ```python
   index = LlamaIndex(data_dir="/path/to/data")
   ```

4. Ingest data: Use the `ingest` method to ingest your data into LlamaIndex. This method takes a LlamaDataset object as an argument, which can be created using the `LlamaDataset.from_csv` or `LlamaDataset.from_pandas` methods:

   ```python
   dataset = LlamaDataset.from_csv("/path/to/data.csv")
   index.ingest(dataset)
   ```

   You can also use other methods like `from_json` or `from_dict` depending on your data format.

5. Structure data: After ingesting the data, you can structure it using indices or graphs. LlamaIndex provides methods like `create_index` and `create_graph` for this purpose. For example, to create an index on a specific column, you can use the `create_index` method:

   ```python
   index.create_index("column_name")
   ```

6. Query data: Once your data is structured, you can query it using LlamaIndex's retrieval/query interface. The `query` method allows you to provide an input prompt and get back the retrieved context and knowledge-augmented output. Here's an example:

   ```python
   input_prompt = "What are songs by Taylor Swift in the pop genre"
   retrieved_context, augmented_output = index.query(input_prompt)
   ```

   You can also specify additional parameters like the number of documents to retrieve using the `top_k` parameter.

7. Customize and extend: LlamaIndex provides both high-level and low-level APIs for customization and extension. Advanced users can customize and extend modules like data connectors, indices, retrievers, query engines, and reranking modules to fit their specific needs.

These are the basic steps to get started with LlamaIndex. You can refer to the LlamaIndex documentation and samples for more detailed information and examples.
"""
