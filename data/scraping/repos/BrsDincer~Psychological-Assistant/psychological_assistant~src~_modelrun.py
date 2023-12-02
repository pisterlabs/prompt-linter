from src._classinits import *
from src._errorsinit import *
from src._readapitokens import *
from src._wikidatasearch import *
from llama_index import LLMPredictor,GPTSimpleVectorIndex,PromptHelper,download_loader
from llama_index import GPTSimpleKeywordTableIndex,GPTListIndex
from llama_index.indices.composability import ComposableGraph
from langchain.chat_models import ChatOpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from warnings import filterwarnings
import gradio as gr
import os

def IGNOREWARNINGOUTPUT()->RESPONSES:
    filterwarnings("ignore",category=DeprecationWarning)
    filterwarnings("ignore",category=UserWarning)


class MODELRUN(object):
    def __init__(self)->CLASSINIT:
        self.__api = READAPITOKEN()
        self.__api._TOKEN()
        self.__base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "data")
        self.__dir = self.__base+"/maindatapsy.csv"
        self.__vec = self.__base+"/"
        os.environ["OPENAI_API_KEY"] = self.__api.token
        self.__rcs = download_loader("SimpleCSVReader")
        self.__ndd = SimpleNodeParser()
    def __str__(self)->str:
        return "MODEL RUN PARAMETERS - SUBPROCESS"
    def __call__(self)->None:
        return None
    def __getstate__(self)->CLASSINIT:
        raise TypeError("[DENIED - PERMISSION]")
    def __repr__(self)->str:
        return MODELRUN.__doc__
    def _NODEDOCUMENT(self)->RESPONSES:
        loader = self.__rcs()
        doc = loader.load_data(file=self.__dir)
        nod = self.__ndd.get_nodes_from_documents(doc)
        return nod
    def _GETPREDICTOR(self,
                      tem:int=0.1,
                      mdn:str="gpt-3.5-turbo",
                      mtk:int=4096)->RESPONSES:
        return LLMPredictor(llm=ChatOpenAI(temperature=tem,
                                           model_name=mdn,
                                           max_tokens=mtk))
    def _GETPROMPT(self,
                   mx:int=4096,
                   ou:int=4096,
                   ck:int=600,
                   mc:int=20)->RESPONSES:
        return PromptHelper(max_input_size=mx,
                            chunk_size_limit=ck,
                            num_output=ou,
                            max_chunk_overlap=mc)
    def _GETVECTOR(self,
                   dct:str or list or tuple or classmethod,
                   ctx:classmethod,
                   fln:str="respsyvec.json")->RESPONSES:
        try:
            smp = GPTSimpleVectorIndex.from_documents(dct,
                                                      service_context=ctx)
            smp.save_to_disk(self.__vec+fln)
            return smp
        except Exception as err:
            print(str(err))
    def _GETWIKI(self,
                 ctx:classmethod,
                 fln:str="reswkkvec.json")->RESPONSES:
        try:
            __wkk = WIKIDATASEARCH()
            __wkk._SEARCH()
            smp = GPTSimpleVectorIndex.from_documents(__wkk.targetcontent,
                                                      service_context=ctx)
            smp.save_to_disk(self.__vec+fln)
            return smp
        except Exception as err:
            print(str(err))
    def _SERVICE(self,pred:classmethod,prom:classmethod)->RESPONSES:
        return ServiceContext.from_defaults(llm_predictor=pred,
                                            prompt_helper=prom)
    def _PREMODELPROCESS(self):
        # main structure for parameters
        md_ = self._GETPREDICTOR()
        pr_ = self._GETPROMPT()
        nd_ = self._NODEDOCUMENT()
        sr_ = self._SERVICE(md_,pr_)
        vc_ = self._GETVECTOR(nd_,sr_)
        wc_ = self._GETWIKI(sr_)
    def _LOAD(self,
              fln:str="respsyvec.json",
              wln:str="reswkkvec.json")->RESPONSES:
        try:
            if os.path.exists(self.__vec+fln) and os.path.exists(self.__vec+wln):
                ix = GPTSimpleVectorIndex.load_from_disk(self.__vec+fln)
                iw = GPTSimpleVectorIndex.load_from_disk(self.__vec+wln)
                return ix,iw
            else:
                FILEERROR().print()
        except Exception as err:
            print(str(err))
    def _LAUNCH(self,
                fln:str="respsyvec.json",
                wln:str="reswkkvec.json")->RESULTS:
        #control
        if not os.path.exists(self.__vec+fln) and not os.path.exists(self.__vec+wln):
            self._PREMODELPROCESS()
        else:
            pass
        #loading modules
        ix,iw = self._LOAD()
        return ix,iw



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    