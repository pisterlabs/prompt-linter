"""Break down or rephrase the follow up input into fewer than heterogeneous one-hop queries to be the input of a retrieval tool, if the follow up inout is multi-hop, multi-step, complex or comparative queries and relevant to Chat History and Document Names. Otherwise keep the follow up input as it is.


The output format should strictly follow the following, and each query can only conatain 1 document name:
```
1. One-hop standalone query
...
3. One-hop standalone query
...
```


Document Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Begin:

Follow Up Input: {question}

One-hop standalone queries(s):
""""""Below are some verified sources and a human input. If you think any of them are relevant or contain any keywords related to the human input, then list all possible context numbers.

```
{snippets}
```

The output format must be like the following, nothing else. If not, you will output []:
[0, ..., n]

Human Input: {query}
""""""You are a helpful assistant designed by IncarnaMind.
If you think the below below information are relevant to the human input, please respond to the human based on the relevant retrieved sources; otherwise, respond in your own words only about the human input.""""""
File Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Verified Sources:
```
{context}
```


User: {question}
""""""
File Names in the database:
```
{database}
```


Chat History:
```
{chat_history}
```


Verified Sources:
```
{context}
```
"""