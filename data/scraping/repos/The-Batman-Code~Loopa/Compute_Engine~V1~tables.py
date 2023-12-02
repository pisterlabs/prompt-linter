import json
import pandas as pd
import requests
from vertexai.language_models import TextGenerationModel
from langchain.llms import VertexAI
from langchain.agents import create_pandas_dataframe_agent
from io import StringIO
import os
from google.oauth2 import service_account

# from langchain.llms import VertexAI
import vertexai

credentials = service_account.Credentials.from_service_account_file(
    "/home/ksgcpcloud/myapp/Ver_1/loopa_key.json"
)
vertexai.init(project="", location="us-central1", credentials=credentials)


def get_tabular_data(session_ID):
    with open(f"/home/ksgcpcloud/myapp/data/{session_ID}/data.txt") as f:
        directory_path = f"/home/ksgcpcloud/myapp/csv_data/{session_ID}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print("Directory created")
        else:
            print("Directory already exists")

        try:
            data = json.load(f)

            num_list = []
            mykeys = list(map(lambda x: x[0], data.items()))
            print(f"This is my keys : {mykeys,mykeys[0]}")
            if mykeys[0] == "Error Message":
                print("I am in df_error")
                df_error = pd.DataFrame(
                    [
                        "Error, please rephrase your prompt. If it's an issue from our side do this."
                    ]
                )
                print(df_error)
                return df_error.to_csv()
            print(f"This is the key : {mykeys}")
            try:
                for i in range(len(mykeys)):
                    if mykeys[i] in data and isinstance(data[mykeys[i]], list):
                        # print(f"The {mykeys[i]} key exists and is a list.")
                        num_list.append(i)
                    # else:
                    #     print(
                    #         # f"The {mykeys[i]} key either doesn't exist or is not a list."
                    #     )
                # print(num_list)

                df_list = []
                # df_list2 = []
                for i in range(num_list[0], len(mykeys)):
                    # column_families = []
                    column_names = []
                    key = mykeys[i]
                    for key2 in data[key][0].keys():
                        # column_families.append((key, key2))
                        column_names.append(key2)
                    df = pd.DataFrame(columns=[column_names])
                    for l in range(len(data[key])):
                        df.loc[len(df)] = data[key][l].values()
                    # df.columns = pd.MultiIndex.from_tuples(column_families)

                    df.to_csv(
                        f"/home/ksgcpcloud/myapp/csv_data/{session_ID}/data{i}.csv"
                    )
                    df_list.append(df.to_csv())
                    # df.to_csv("data.csv")
                    # print(df.dtypes())
                    # print(df)
                print("I am inside the nested try block", session_ID)
                # final_df = pd.concat(df_list)
                # final_df = final_df.reset_index(drop=True)
                return df_list
            except:
                df = pd.DataFrame([data])
                # if not os.path.exists(directory_path):
                #     os.makedirs(directory_path)
                #     print("Directory created")
                # else:
                #     print("Directory already exists")
                df.to_csv(f"/home/ksgcpcloud/myapp/csv_data/{session_ID}/data1.csv")
                print("I am the inside nested except block", session_ID)
                return df.to_csv()
        except:
            # if not os.path.exists(directory_path):
            #     os.makedirs(directory_path)
            #     print("Directory created")
            # else:
            #     print("Directory already exists")
            df = pd.read_csv(f"/home/ksgcpcloud/myapp/data/{session_ID}/data.txt")

            print("I am the outside except block", session_ID)
            df.to_csv(f"/home/ksgcpcloud/myapp/csv_data/{session_ID}/data1.csv")
            # print(df.to_csv())
            return df.to_csv()


# url_generation_functionality ----------------------------------------------------------------------------------------------------------
def get_url_generation_functionality(user_text):
    url_dict = {
        "url": "The constructed url here",
        "heading": "The few word title for the user's query",
    }

    url_dict = json.dumps(url_dict)

    # vertexai.init(project="pde-3-391703", location="us-central1")
    parameters = {
        "temperature": 0.1,
        "max_output_tokens": 256,
        "top_p": 0.6,
        "top_k": 10,
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(
        f"""
{url_dict}
The text is : "{user_text}"""
        "",
        **parameters,
    )

    # Cleaning the response
    json_string_without_backticks = response.text.replace("`", "")
    element = "{"

    split_parts = json_string_without_backticks.split(element)
    if len(split_parts) > 1:
        json_string_without_backticks = split_parts[1].strip()
        json_string_without_backticks = "{" + json_string_without_backticks
        clean_json_answer = json.loads(json_string_without_backticks)
        print(f"{clean_json_answer}\n\n")
    else:
        print("Element not found in the string.")

    return clean_json_answer


def write_to_file(data, session_ID):
    # writing to txt file
    directory_path = f"/home/ksgcpcloud/myapp/data/{session_ID}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory created")
    else:
        print("Directory already exists")
    with open(f"{directory_path}/data.txt", "w") as f:
        f.write(data)


def get_url_fetch_data(json_text):
    url = json_text["url"] + "&apikey="
    # print(url)
    r = requests.get(url)
    # print(r.headers.get("Content-Type"))
    return r.text


# def get_url_fetch_data(json_text):
#     url = json_text["url"] + "&apikey="
#     # print(url)
#     r = requests.get(url)
#     # print(r.headers.get("Content-Type"))
#     return r.text


def table_queries_2(user_query, session_ID):
    print("I am in queries 2")
    try:
        # data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)[1]))
        data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)))
    except:
        # data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)))
        data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)[1]))
    agent = create_pandas_dataframe_agent(
        VertexAI(temperature=0.2, model_name="text-bison", max_output_tokens=1024),
        data_frame.convert_dtypes(),
        verbose=True,
    )
    return agent.run(user_query)


def table_queries_1(user_query, session_ID):
    print("I am in queries 1")
    try:
        # data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)[1]))
        data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)))
    except:
        # data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)))
        data_frame = pd.read_csv(StringIO(get_tabular_data(session_ID)[0]))
    print(f"I am printing df here:\n{data_frame}")
    agent = create_pandas_dataframe_agent(
        VertexAI(temperature=0.2, model_name="text-bison", max_output_tokens=1024),
        data_frame.convert_dtypes(),
        verbose=True,
    )
    return agent.run(user_query)


def data_retrieval_main(user_table_text, session_ID):
    write_to_file(
        get_url_fetch_data(get_url_generation_functionality(user_table_text)),
        session_ID,
    )
    return get_tabular_data(session_ID)
    # try:
    #     return get_tabular_data()[0]
    # except:
    #     return get_tabular_data()


if __name__ == "__main__":
    while True:
        text = input("Enter text here:\n")
        data = get_url_generation_functionality(text)

        write_to_file(get_url_fetch_data(data))
        data_frame = get_tabular_data()[0]
        data_frame2 = pd.read_csv(StringIO(data_frame)).infer_objects()
        print(data_frame2.dtypes)
        # data_frame2 = data_frame.convert_dtypes()
        # print(data_frame2.dtypes)
        text2 = input("Enter data query here:\n")
        agent = create_pandas_dataframe_agent(
            VertexAI(temperature=0.2, model_name="text-bison", max_output_tokens=1024),
            data_frame2,
            verbose=True,
        )
        answer = agent.run(text2)
        print(answer)
