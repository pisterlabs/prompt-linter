from scipy.spatial.distance import cosine
import argparse
import json
import os
import openai
import pdb

def read_text(input_file):
    arr = open(input_file).read().split("\n")
    return arr[:-1]


class OpenAIQnAModel:
    def __init__(self):
        self.debug = False
        self.q_model_name = None
        self.d_model_name = None
        self.skip_key = True
        print("In OpenAI API constructor")


    def init_model(self,model_name = None):
        #print("OpenAI: Init model",model_name)
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if (openai.api_key == None):
            openai.api_key = ""
            print("API key not set")

        if (len(openai.api_key) == 0 and not self.skip_key):
                print("Open API key not set")
        
        if (model_name is None):
            self.d_model_name = "text-search-ada-doc-001"
        else:
            self.d_model_name = model_name
        self.q_model_name = self.construct_query_model_name(self.d_model_name)
        print(f"OpenAI: Init model complete :query model {self.q_model_name} doc:{self.d_model_name}")

    def construct_query_model_name(self,d_model_name):
        return d_model_name.replace('-doc-','-query-')


    def compute_embeddings(self,input_file_name,input_data,is_file):
        if (len(openai.api_key) == 0 and not self.skip_key):
                print("Open API key not set")
                return [],[]
        #print("In compute embeddings after key check")
        in_file = input_file_name.split('/')[-1]
        in_file = self.d_model_name + '_' +  '.'.join(in_file.split('.')[:-1]) + "_search.json"
        cached = False
        try:
            fp = open(in_file)
            cached = True
            embeddings = json.load(fp)
            q_embeddings = [embeddings[0]]
            d_embeddings = embeddings[1:]
            print("Using cached embeddings")
        except:
            pass
            
        texts = read_text(input_data) if is_file == True else input_data
        queries = [texts[0]]
        docs = texts[1:]

        if (not cached):
            print(f"Computing embeddings for {input_file_name} and query model {self.q_model_name}")
            query_embeds = openai.Embedding.create(
                input=queries,
                model=self.q_model_name
            )
            print(f"Computing embeddings for {input_file_name} and doc model {self.q_model_name}")
            doc_embeds = openai.Embedding.create(
                input=docs,
                model=self.d_model_name
            )
            q_embeddings = []
            d_embeddings = []
            for i in range(len(query_embeds['data'])):
                q_embeddings.append(query_embeds['data'][i]['embedding'])
            for i in range(len(doc_embeds['data'])):
                d_embeddings.append(doc_embeds['data'][i]['embedding'])
        if (not cached):
            embeddings = q_embeddings + d_embeddings
            with open(in_file,"w") as fp:
                json.dump(embeddings,fp)
        return texts,(q_embeddings,d_embeddings)

    def output_results(self,output_file,texts,embeddings,main_index = 0):
        # Calculate cosine similarities
        # Cosine similarities are in [-1, 1]. Higher means more similar
        query_embeddings = embeddings[0]
        doc_embeddings = embeddings[1]
        cosine_dict = {}
        queries = [texts[0]]
        docs = texts[1:]
        if (self.debug):
            print("Total sentences",len(texts))
        for i in range(len(docs)):
            cosine_dict[docs[i]] = 1 - cosine(query_embeddings[0], doc_embeddings[i])

        if (self.debug):
            print("Input sentence:",texts[main_index])
        sorted_dict = dict(sorted(cosine_dict.items(), key=lambda item: item[1],reverse = True))
        if (self.debug):
            for key in sorted_dict:
                print("Cosine similarity with  \"%s\" is: %.3f" % (key, sorted_dict[key]))
        if (output_file is not None):
            with open(output_file,"w") as fp:
                fp.write(json.dumps(sorted_dict,indent=0))
        return sorted_dict



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='OpenAI model for document search embeddings ',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-input', action="store", dest="input",required=True,help="Input file with sentences")
        parser.add_argument('-output', action="store", dest="output",default="output.txt",help="Output file with results")
        parser.add_argument('-model', action="store", dest="model",default="text-search-ada-doc-001",help="model name")

        results = parser.parse_args()
        obj = OpenAIQnAModel()
        obj.init_model(results.model)
        texts, embeddings = obj.compute_embeddings(results.input,results.input,is_file = True)
        results = obj.output_results(results.output,texts,embeddings)
