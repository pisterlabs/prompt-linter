from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
prompt = ChatPromptTemplate.from_template(
    "What could be the catagory for the query {input}, The Categories are [Banking and Finance, Civil, Constitutional, Consumer Protection,Corporate, Criminal, Environmental, Family,Human Rights, Immigration, Intellectual Property,Labor, Media and Entertainment, Medical,Real Estate, Tax] return 3 catagories in a array"
)

llm = OpenAI(openai_api_key="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")
model = llm

str_chain = prompt | model | StrOutputParser()

catagories = [i.replace("and ", "") for i in str_chain.invoke({"input": "Two brothers were tenant of a landlord in a commercial property. One brother had one son and a daughter (both minor) when he got divorced with his wife. The children went into the mother's custody at the time of divorce and after some years the husband (co tenant) also died. Now can the children of the deceased brother (co tenant) claim the right."}).strip().replace("[",'').split(', ')]

print(catagories)

import pandas as pd
df_new = pd.read_csv(r"C:\Users\a21ma\OneDrive\Desktop\Datahack\DataHack_2_Tensionflow\Vector Database\recproject\FINALFINALFINALdataset.csv")
df_new = df_new.loc[:, ~df_new.columns.str.contains('^Unnamed')]
filtered_df = df_new[df_new['Type_of_Lawyer'].str.contains(f'{catagories[0]}|{catagories[1]}|{catagories[2]}')]
filter2 = filtered_df.sort_values(by=['Years_of_Experience', 'Rating'], ascending=[False, False])

filtered_df = df_new[df_new['Type_of_Lawyer'].str.contains(f'{catagories[0]}|{catagories[1]}|{catagories[2]}')]
def matching_algorithm(lawyer):
    score = 0
    ty = [i for i in lawyer['Type_of_Lawyer'].replace("[", "").replace("]",'').replace("'",'').split(", ")]
    for x in catagories:
      if x in ty:
          score += 1
    return score

filtered_df['Match_Score'] = filtered_df.apply(matching_algorithm, axis=1)

top_lawyers = filtered_df.sort_values(by=['Match_Score','Rating','Years_of_Experience','Charges'], ascending=[False,False,False,True])

print(top_lawyers.head(10))
