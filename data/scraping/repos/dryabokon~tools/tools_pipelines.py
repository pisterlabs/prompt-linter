import uuid
import io
import pandas as pd
import requests
import yaml
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from halo import Halo
# ---------------------------------------------------------------------------------------------------------------------
import tools_DF
import tools_time_profiler
import tools_Logger
from LLM import tools_Langchain_API
# ---------------------------------------------------------------------------------------------------------------------
class Pipeliner:
    def __init__(self,filename_config_chat_model,filename_config_api,filename_api_spec,folder_out):
        self.L = tools_Logger.Logger(folder_out + 'log.txt')
        self.TP = tools_time_profiler.Time_Profiler()
        self.load_config(filename_config_api)
        self.A  = tools_Langchain_API.Assistant_API(filename_config_chat_model,filename_api_spec,access_token=self.get_access_token())
        self.folder_out = folder_out
        self.df = None
        return
# ---------------------------------------------------------------------------------------------------------------------
    def load_config(self,filename_config_api):
        with open(filename_config_api, 'r') as config_file:
            config = yaml.safe_load(config_file)
            if 'credentials' in config:
                self.host = config['credentials']['host']
                self.client_id = config['credentials']['client_id']
                self.tenant_id = config['credentials']['tenant_id']
                self.scope = config['credentials']['scope']
                self.client_secret = config['credentials']['client_secret']
        return
# ----------------------------------------------------------------------------------------------------------------------
    def yaml_to_json(self, text_yaml):
        io_buf = io.StringIO()
        io_buf.write(text_yaml)
        io_buf.seek(0)
        res_json = yaml.load(io_buf, Loader=yaml.Loader)
        return res_json
# ----------------------------------------------------------------------------------------------------------------------
    def get_access_token(self):
        url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret,"scope": f'{self.scope}'}
        response = requests.post(url, headers=headers, data=data)
        self.access_token = self.yaml_to_json(response.text)['access_token']
        #print(self.access_token)
        return self.access_token
# ----------------------------------------------------------------------------------------------------------------------
    def run_request(self,url,params=None):
        headers = {"Authorization": f"Bearer {self.access_token}","User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Content-Type": "application/json"}
        response = requests.get(url, headers=headers,params=params)
        return response
# ----------------------------------------------------------------------------------------------------------------------
    def parse_response(self,txt_data):
        df = pd.DataFrame([])
        for dct in self.A.yaml_to_json(txt_data):
            df = pd.concat([df, pd.DataFrame.from_dict([dct])])

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def run_and_parse(self,url):
        if url is None:
            url = f'https://{self.host}'
            params = {
                'limit':100}
        else:
            params = None

        response = self.run_request(url, params=params)
        print(response.status_code, response.reason)
        if response.status_code==200:
            print(self.parse_response(response.text))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def query_to_df(self,query,verbose=False):
        res = self.A.chain.api_request_chain.predict(question=query, api_docs=self.A.api_spec)
        url = res.split()[0]
        print(query)
        print(url)
        response = self.run_request(url)
        print(response.status_code, response.reason)
        df = pd.DataFrame([])
        if response.status_code == 200:
            df = self.parse_response(response.text)
            df.to_csv(self.folder_out + f'{uuid.uuid4().hex}.csv', index=False)

        if verbose:
            print(tools_DF.prettify(df, showindex=False))

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def query_over_df(self,query, df,post_proc = '',verbose=True,as_df=True):
        if (self.df is None) or (self.df.equals(df)):
            self.df = df
            self.pandas_agent = create_pandas_dataframe_agent(self.A.LLM, df, verbose=verbose,return_intermediate_steps=True)

        if not verbose:
            self.TP.tic('query_over_df',verbose=False)
            spinner = Halo(color='white',spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        try:
            agent_res = self.pandas_agent(query + post_proc)

            if as_df:
                dfs = []
                for step in agent_res["intermediate_steps"]:
                    for sub_step in step:
                        if isinstance(sub_step,pd.DataFrame):
                            dfs.append(sub_step)

                if len(dfs)>0:
                    res = dfs[-1]
                else:
                    res = agent_res['output']
            else:
                res= agent_res['output']

        except:
            res = 'Error'

        if not verbose:
            spinner.succeed(text=self.TP.print_duration('query_over_df', verbose=False))
            spinner.stop()

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def query_agent(self,query,verbose=True):

        if not verbose:
            self.TP.tic('query_over_df', verbose=False)
            spinner = Halo(color='white',spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        res = self.A.agent.run(query)

        if not verbose:
            spinner.succeed(text=self.TP.print_duration('query_over_df', verbose=False))
            spinner.stop()

        return res
# ----------------------------------------------------------------------------------------------------------------------