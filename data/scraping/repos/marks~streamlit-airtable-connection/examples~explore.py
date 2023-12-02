import streamlit as st
import pandas as pd
import json

from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from streamlit_airtable import AirtableConnection

import explore_helpers

# Initiate connection to Airtable using st.connection
airtable_conn = st.connection(
    "your_connection_name", type=AirtableConnection
)

# Retrieve list of bases, and create a dict of base id to name
bases_list = airtable_conn.list_bases()
bases_id_to_name = {base["id"]: base["name"] for base in bases_list["bases"]}

# Add a sidebar to select a base
with st.sidebar:
    st.markdown("## Configuration")

    # with st.form("Configuration"):
    selected_base_id = st.selectbox(
        "Which base would you like to explore?",
        options=list(bases_id_to_name.keys()),
        format_func=lambda base_id: bases_id_to_name[base_id],
        help="If you don't see a base in the list, make sure your personal access token has access to it.",
    )
    openai_api_key = st.text_input(
        "(Optional) What is your OpenAI API key?",
        value="",
        type="password",
        help="(Optional) You can find your API key at https://platform.openai.com/account/api-keys. If not supplied, you will not be able to use the AI features on this page.",
    )

# Main content pane
with st.container():
    st.info(
        "Welcome to the demo of the Airtable connector for Streamlit. The [source code for the connector and this example can be found on Github](https://github.com/marks/streamlit-airtable-connection). You can also watch a [demo video](https://share.support.airtable.com/RBuJRnyL) and access the [underlying sample Airtable base](https://airtable.com/appdbRXgibDiQkNQN/shrIO0m8oyeQJTf9T) from [thesquirrelcensus.com](https://www.thesquirrelcensus.com/).",
        icon="👋",
    )

    st.markdown("# Airtable Base Explorer")
    # st.markdown(f"## `{bases_id_to_name[selected_base_id]}`")
    base_schema = airtable_conn.get_base_schema(base_id=selected_base_id)

    st.markdown(
        f"You're exploring the base named [{bases_id_to_name[selected_base_id]}](https://airtable.com/{selected_base_id})."
    )

    st.markdown("### Base schema")

    # Two columns: list of tables and graph of relationships
    col1, col2 = st.columns(2)

    # Show the number and a list of tables in the base
    col1.write(f"**{len(base_schema['tables'])} tables**")
    # Create a dataframe of the table names and ids; and display number of
    tables_df = pd.DataFrame(
        [
            {
                "name": table["name"],
                "id": table["id"],
                "deep_link": f"https://airtable.com/{selected_base_id}/{table['id']}",
            }
            for table in base_schema["tables"]
        ]
    )
    col1.dataframe(
        tables_df,
        column_config={"deep_link": st.column_config.LinkColumn()},
        hide_index=True,
    )

    col1.download_button(
        "Download base schema as JSON",
        json.dumps(base_schema),
        f"base-schema-{selected_base_id}.json",
        "application/json",
    )

    # Show a graph of the relationships between tables
    graph = explore_helpers.create_graph_from_base_schema(selected_base_id, base_schema)
    col2.write(
        f"**Table relationships**\n\nNodes are table names and edges are _field names_ of the linked record fields. Both are clickable links to the Airtable UI for that item."
    )
    col2.graphviz_chart(graph)

    st.divider()
    st.markdown("### Table schemas")

    table_schema_tabs = st.tabs([f"{table['name']}" for table in base_schema["tables"]])

    # Show the full schema for each table in an expander
    for i, table_schema in enumerate(base_schema["tables"]):
        this_tab = table_schema_tabs[i]
        this_tab.write(
            f"**{len(table_schema['fields'])} fields** belonging to table `{table_schema['id']}`:"
        )
        fields_df = pd.DataFrame(
            [
                {
                    "name": item["name"],
                    "id": item["id"],
                    "type": item["type"],
                    "choices": [
                        choice["name"] for choice in item["options"].get("choices", [])
                    ]
                    if item["type"] in ["singleSelect", "multipleSelects"]
                    else None,
                    "linked_table_id": item["options"]["linkedTableId"]
                    if item["type"] in ["multipleRecordLinks"]
                    else None,
                    "deep_link": f"https://airtable.com/{selected_base_id}/{table_schema['id']}/{item['id']}",
                }
                for item in table_schema["fields"]
            ]
        )
        this_tab.dataframe(
            fields_df,
            column_config={"deep_link": st.column_config.LinkColumn()},
            hide_index=True,
        )

        col1, col2 = this_tab.columns(2)

        col1.download_button(
            "Download list of fields as CSV",
            fields_df.to_csv(index=False).encode("utf-8"),
            f"table-schema-{selected_base_id}-{table_schema['id']}.csv",
            "text/csv",
        )

        col2.download_button(
            "Download full table schema as JSON",
            json.dumps(table_schema),
            f"table-schema-{selected_base_id}-{table_schema['id']}.json",
            "application/json",
        )

    st.divider()
    st.markdown("### Base records")
    st.markdown(
        "The following record previews display the first 10 records from the [Airtable list records API](https://airtable.com/developers/web/api/list-records) in `string` cell format. This will return the same values you see in the Airtable UI and when you export as CSV."
    )

    table_record_tabs = st.tabs([f"{table['name']}" for table in base_schema["tables"]])

    # Show the full schema for each table in an expander
    for i, table_schema in enumerate(base_schema["tables"]):
        this_tab = table_record_tabs[i]

        this_tab.dataframe(
            airtable_conn.query(
                base_id=selected_base_id,
                table_id=table_schema["id"],
                max_records=10,
                cell_format="string",
                time_zone="America/Los_Angeles",
                user_locale="en-us",
            ),
            hide_index=True,
        )

    st.divider()
    st.markdown("### AI")
    st.markdown(
        "Uses [LangChain](https://www.langchain.com), [the LangChain Pandas Dataframe Agent](https://python.langchain.com/docs/integrations/toolkits/pandas), and [OpenAI's `chatgpt-3.5-turbo` model](https://platform.openai.com/docs/models/gpt-3-5) to answer questions about the table's records.")

    if not openai_api_key.startswith("sk-"):
        st.warning(
            "Please enter your OpenAI API in the configuration question to the left to test the AI functionality",
            icon="⚠",
        )
    else:
        # Select a table to query
        table_for_ai = st.selectbox(
            "Which table would you like ask questions about?",
            tables_df["name"],
        )

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                # {"role": "assistant", "content": "How can I help you?"}
            ]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(
            f"Enter a question about the '{table_for_ai}' table"
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Generating response ..."):
                # TODO refactor, but caching is built into the connector
                df_for_ai = airtable_conn.query(
                    base_id=selected_base_id,
                    table_id=table_for_ai,
                    cell_format="string",
                    time_zone="America/Los_Angeles",
                    user_locale="en-us",
                )

                # Initiate language model
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.2,
                    openai_api_key=openai_api_key,
                )
                # Create Pandas DataFrame Agent
                agent = create_pandas_dataframe_agent(
                    llm,
                    df_for_ai,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                )
                # Perform Query using the Agent
                response = {"content": agent.run(prompt), "role": "assistant"}

            st.session_state.messages.append(response)
            st.chat_message("assistant").write(response["content"])
