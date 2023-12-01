## code for loading and training the language model
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import langchain
import pinecone
import time
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd
import os
from generate_graph import attr_to_num


def node_to_ans(nodename:str)->str:
    return nodename.split(':')[1].replace('_',' ')

def dataset_to_csv(input_file:str='dataset.txt',output_file:str='dataset.csv')->None:
    with open(input_file,'r') as f:
        lines = f.readlines()
        qlines = lines[::3]
        answers = lines[1::3]
    qlines = [q.strip() for q in qlines if q.strip()!='']
    answers = [a.split(':')[1].strip() for a in answers if a.strip()!='']
    df = pd.DataFrame({'queries':qlines,'answers':answers})
    df.to_csv(output_file)
    return 

### somehow scrape dataset.txt for finetuning the lm ###
def get_lm_train_data(dataset_file:str='dataset.csv',add_desc:bool=False,save_to_csv:bool=True)->pd.DataFrame:
    openaiapi_key = os.environ.get('OPENAI_KEY',None)
    if openaiapi_key is None:
        raise Exception("Set env OPENAI_KEY to your OpenAI Key")

    client = openai.OpenAI(
      api_key=openaiapi_key,  # this is also the default, it can be omitted
    )
    
    df = pd.read_csv(dataset_file)
    qlines = df['queries'].values.to_list()
    
    if add_desc:
        #desc_prompt = 
        responses = []
        descriptions = []
        
        # df['sparqls']= responses
        # df['descs']= descriptions
        raise NotImplementedError
    else: ## no desc
        prmpt = lambda query:f'''Given queries enclosed in arrows, convert them into the SPARQL language in order to query over a knowledge graph containing nodes for 'actor','director','movie', 'genre', 'year'. Each node name is prefixed by its type, and contains underscores instead of spaces. For example actor Michael Scott's node reads 'actor:Michael_Scott'. Each relation is one out of {' '.join(attr_to_num.keys())}, with the edge pointing in the appropriate direction.
        You may think over your answer, but your final answer for each query must be enclosed in triple slashes '/// ///'.
        
        The queries are :
        {query}'''
        responses = []
        
        # give k queries at a time
        k = 10
        qb = ['\n'.join([f'<<<{q}>>>'for q in qlines[i:i+k]]) for i in range(0, len(qlines), k)]
        for query in qb:
            rp = client.completions.create(
                    model="davinci-002",
                    prompt=prmpt(query)
                )
            ans = rp.choices[0].text.split('///')[1::2]
            responses += ans
            
        df['sparqls']=responses
    
    if save_to_csv:
        df.to_csv(dataset_file)
    return df
        


### an LM, when given a query, extracts entities and relationships from the query ###
class ParserLM:
    def __init__(self,lm_name:str='flan-t5-base',tokenizer_name:str='flan-t5-base',finetune:bool=False,finetuned_path:str|None=None,desc:bool=False)->None:
        self.lm = transformers.AutoModelForSeq2SeqLM.from_pretrained(lm_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if finetune:
            self.finetune('dataset.csv')
        self.lm.eval()
        self.lm.to(device)
        self.tokenizer.to(device)
        self.desc = desc
        return
    
    def finetune(self,data_path:str='dataset.csv')->None:
        df = pd.read_csv(data_path)
        if self.desc:
            if 'descs' not in df.columns:
                df = get_lm_train_data(data_path,add_desc=True,save_to_csv=True)
            queries,responses,descs = df['queries'].values.tolist(),df['sparqls'].values.tolist(),df['descs'].values.tolist() 
            raise NotImplementedError
            return 
            
            
        # no desc    
        if 'sparqls' not in df.columns:
            df = get_lm_train_data(data_path,add_desc=False,save_to_csv=True)
            queries,responses = df['queries'].values.tolist(),df['sparqls'].values.tolist()
            ## tokenize queries and responses and convert to dataloader###
            queries = self.tokenizer(queries,return_tensors='pt',padding=True,truncation=True)
            responses = self.tokenizer(responses,return_tensors='pt',padding=True,truncation=True)
            
            data = torch.utils.data.TensorDataset(queries['input_ids'],queries['attention_mask'],responses['input_ids'],responses['attention_mask'])
            train_data, eval_data = torch.utils.data.random_split(data, [int(len(data)*0.9), len(data) - int(len(data)*0.9)])
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=8)
            eval_dataloader = DataLoader(eval_data, sampler=None, batch_size=8)
            ## train the model ##
            trainer = transformers.Trainer(
                model=self.lm,
                args=transformers.TrainingArguments(
                    output_dir="./flan-t5-base-imdb-nodesc",
                    evaluation_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=1,
                    weight_decay=0.01,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    logging_dir="./logs",
                ),
                train_dataset=train_dataloader,
                eval_dataset=eval_dataloader,
                tokenizer=self.tokenizer,
            )
                
            return
            
        
    def parse(self,query:str)->str:
        encoded_text = self.tokenizer(query, return_tensors="pt")
        response = self.lm.generate(**encoded_text)
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
    


### if the rag model is to be extended with the description this will be needed ###
class Embedder: ## generate contextual embeddings and ids for given input texts, then return closest matches on queries
    def __init__(self,index_name:str,emb_dim=1536)->bool:
        self.index_name = index_name
        new = False
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=emb_dim,
                metric='cosine'
            )
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
            new = True
        self.index = pinecone.Index(index_name)
        print(f"Index stats: {self.index.describe()}")

        self.embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")## this is the model we are using for now, change it later
        return new
    
    def push(self,text:str)->str:
        raise NotImplementedError
        pass
    
    def compile(self)->None:
        self.vectorstore = pinecone.Pinecone(
            self.index, self.embed_model.embed_query, text_field
        )
        return
    def query(self,q:str,top_k:int)->list[str]:
        return self.vectorstore.similarity_search(q,k=top_k)