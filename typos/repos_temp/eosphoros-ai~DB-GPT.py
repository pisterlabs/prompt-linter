"""
This is an example data，please learn to understand the structure and content of this data:
    {data_example}
Explain the meaning and function of each column, and give a simple and clear explanation of the technical terms.  
Provide some analysis options,please think step by step.

Please return your answer in JSON format, the return format is as follows:
    {response}
""""""
下面是一份示例数据，请学习理解该数据的结构和内容:
    {data_example}
分析各列数据的含义和作用，并对专业术语进行简单明了的解释。
提供一些分析方案思路，请一步一步思考。

请以JSON格式返回您的答案，返回格式如下：
    {response}
""""""
Goals: 
    {input}
    
Constraints:
0.Exclusively use the commands listed in double quotes e.g. "command name"
{constraints}
    
Commands:
{commands_infos}

Please response strictly according to the following json format:
{response}
Ensure the response is correct json and can be parsed by Python json.loads
""""""
Given an input question, create a syntactically correct {dialect} sql.

Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
Use as few tables as possible when querying.
Only use the following tables schema to generate sql:
{table_info}
Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Question: {input}

Respond in JSON format as following format:
{response}
Ensure the response is correct json and can be parsed by Python json.loads
""""""
你是一个 SQL 专家，给你一个用户的问题，你会生成一条对应的 {dialect} 语法的 SQL 语句。

如果用户没有在问题中指定 sql 返回多少条数据，那么你生成的 sql 最多返回 {top_k} 条数据。 
你应该尽可能少地使用表。

已知表结构信息如下：
{table_info}

注意：
1. 只能使用表结构信息中提供的表来生成 sql，如果无法根据提供的表结构中生成 sql ，请说：“提供的表结构信息不足以生成 sql 查询。” 禁止随意捏造信息。
2. 不要查询不存在的列，注意哪一列位于哪张表中。
3. 使用 json 格式回答，确保你的回答是必须是正确的 json 格式，并且能被 python 语言的 `json.loads` 库解析, 格式如下：
{response}
""""""请根据提供的上下文信息的进行总结:
{context}
回答的时候最好按照1.2.3.点进行总结
""""""
Write a summary of the following context: 
{context}
When answering, it is best to summarize according to points 1.2.3.
""""""A chat between a curious user and an artificial intelligence assistant, who very familiar with database related knowledge. 
    The assistant gives helpful, detailed, professional and polite answers to the user's questions. """""" 基于以下已知的信息, 专业、简要的回答用户的问题,
            如果无法从提供的内容中获取答案, 请说: "知识库中提供的内容不足以回答此问题" 禁止胡乱编造。 
            已知内容: 
            {context}
            问题:
            {question}
"""""" Based on the known information below, provide users with professional and concise answers to their questions. If the answer cannot be obtained from the provided content, please say: "The information provided in the knowledge base is not sufficient to answer this question." It is forbidden to make up information randomly. 
            known information: 
            {context}
            question:
            {question}
"""""" 基于以下已知的信息, 专业、简要的回答用户的问题,
            如果无法从提供的内容中获取答案, 请说: "知识库中提供的内容不足以回答此问题" 禁止胡乱编造。 
            已知内容: 
            {context}
            问题:
            {question}
            
""""""
Based on the following known database information?, answer which tables are involved in the user input.
Known database information:{db_profile_summary}
Input:{db_input}
You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads
The response format must be JSON, and the key of JSON must be "table".

"""