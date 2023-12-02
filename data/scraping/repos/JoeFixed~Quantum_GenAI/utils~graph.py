import openai
import json
import re
import networkx as nx
import streamlit as st
from pyvis.network import Network
from utils.translation_utils import perform_translation, load_translation_models
from config import Settings
API_KEY = Settings().GPT_API_KEY
openai.api_key = API_KEY


# (
#     model_ar_en,
#     tokenizer_ar_en,
#     model_en_ar,
#     tokenizer_en_ar,
# ) = load_translation_models()


def create_graph_network_new_approach1(context, model_en_ar,tokenizer_en_ar):
    def plot_graph(kg):
        G = nx.DiGraph()
        if len(kg) > 2:
            G.add_edges_from(
                (
                    perform_translation(source, model_en_ar,tokenizer_en_ar ),
                    perform_translation(target, model_en_ar,tokenizer_en_ar),
                    {"relation": perform_translation(relation, model_en_ar,tokenizer_en_ar)},
                )
                for source, relation, target in kg
            )

            # Convert the networkx graph to a pyvis graph
            nt = Network(notebook=True, height="750px", width="100%")
            nt.from_nx(G)

            # Customize the appearance (optional)
            nt.toggle_physics(True)

            # Save to an HTML file
            html_file_path = "temp_graph.html"
            nt.show(html_file_path)

            # Display the pyvis graph in Streamlit
            st.components.v1.html(open(html_file_path, "r").read(), height=800)
        else:
            pass

    def strict_output(
        system_prompt,
        user_prompt,
        output_format,
        default_category="",
        output_value_only=False,
        model="gpt-3.5-turbo",
        temperature=0,
        num_tries=2,
        verbose=False,
    ):
        error_msg = ""
        for i in range(num_tries):
            output_format_prompt = f"\nYou are to output the following in json format: {output_format}. Do not put quotation marks or escape character \\ in the output fields."

            response = openai.chat.completions.create(
                temperature=temperature,
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt + output_format_prompt + error_msg,
                    },
                    {"role": "user", "content": str(user_prompt)},
                ],
            )
            res = response.choices[0].message.content.replace("'", '"')
            res = re.sub(r"(\w)\"(\w)", r"\1'\2", res)
            try:
                output = json.loads(res)
                if isinstance(output, list):
                    return output
                else:
                    return [output]
            except Exception as e:
                error_msg = f"\n\nResult: {res}\n\nError message: {str(e)}"
        return {}

    res = strict_output(
        system_prompt=(
            "You are a knowledgeable entity tasked with analyzing the provided domain content. "
            "Construct a knowledge graph that identifies key entities, events, and relationships "
            "present in the provided domain narrative. Your graph should represent "
            "the interconnections between these components in a way that's comprehensive and clear. "
            "The knowledge graph output should be in the form of a list of relations, each consisting of "
            "[object_1, relation, object_2]."
        ),
        user_prompt=context,
        output_format={
            "Knowledge Graph": "List of relations of the form [object_1, relation, object_2]"
        },
    )
    if res and "Knowledge Graph" in res[0]:
        kg = res[0]["Knowledge Graph"]
        plot_graph(kg)  # Plot the knowledge graph
    else:
        print("The response doesn't contain a 'Knowledge Graph' property.")

    
