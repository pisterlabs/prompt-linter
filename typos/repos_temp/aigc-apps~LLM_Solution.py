"""
            #  <center> \N{fire} Chatbot Langchain with LLM on PAI ! 

            ### <center> \N{rocket} Build your own personalized knowledge base question-answering chatbot. 
                        
            <center> 
            
            \N{fire} Platform: [PAI](https://help.aliyun.com/zh/pai)  /  [PAI-EAS](https://www.aliyun.com/product/bigdata/learn/eas)  / [PAI-DSW](https://pai.console.aliyun.com/notebook)
            
            \N{rocket} Supported VectorStores:  [Hologres](https://www.aliyun.com/product/bigdata/hologram)  /  [ElasticSearch](https://www.aliyun.com/product/bigdata/elasticsearch)  /  [AnalyticDB](https://www.aliyun.com/product/apsaradb/gpdb)  /  [FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
                
            """"""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""""""请根据聊天记录和新问题，将新问题改写为一个独立问题。
不需要回答问题，一定要返回一个疑问句。
聊天记录：
{chat_history}
新问题：{question}
独立问题："""