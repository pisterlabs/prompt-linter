import json
import requests
import pandas as pd
import openai
import time

def call_openai(prompt: str) -> str:
    """
    Call the OpenAI API and return the generated text.

    Args:
        prompt (str): The prompt to generate text from.

    Returns:
        str: The generated text.
    """
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0]['text'].strip()
    except openai.error.RateLimitError:
        # Catch the rate limit error and wait for one minute before trying again
        print("Rate limit reached. Waiting for 60 seconds before trying again...")
        time.sleep(60)
    except Exception as e:
        # Catch any exceptions thrown by the OpenAI API and print the error message
        print(f"Error calling OpenAI API: \nError Type: {type(e)}\nError Message: {e}")
        return "Error calling OpenAI API"

class TheCompanyAPI:
    """
    Goal: Use the company API to enrich our damn ta

    Two steps:

    Step 1:
        We 
    """
    def __init__(self,
                 thecompanyapi_token: str,
                 thecompanyapi_cached_file_location: str,
                 company_domain_file_location: str,
                 columns_to_keep: list[str]
                ) -> None:
        # API Token
        self.thecompanyapi_token = thecompanyapi_token

        # File location of the cached database (JSON)
        self.thecompanyapi_cached_file_location = thecompanyapi_cached_file_location
        self.company_domain_file_location = company_domain_file_location

        # Company API cached is where the database will be loaded/updated
        # This variable is saved at the end
        self.thecompanyapi_cached = self.load_data(self.thecompanyapi_cached_file_location)
        self.company_domain = self.load_data(self.company_domain_file_location)
        self.columns_to_keep = columns_to_keep

        self.total_domain_count = 0
        self.total_company_name_count = 0
        self.count = 0
        self.total_already_cached = 0
        self.chatgpt_call = 0

        print("""
    Enriching data about companies...
    Powered by:
  _____ _                                                          
 |_   _| |__   ___  ___ ___  _ __ ___  _ __   __ _ _ __  _   _     
   | | | '_ \ / _ \/ __/ _ \| '_ ` _ \| '_ \ / _` | '_ \| | | |    
   | | | | | |  __/ (_| (_) | | | | | | |_) | (_| | | | | |_| |    
   |_| |_| |_|\___|\___\___/|_| |_| |_| .__/ \__,_|_| |_|\__, |    
                                      |_|                |___/     
                            ___    ____ ___                        
                           / / \ ||  _ \_ _|                       
  _____ _____ _____ _____ / / _ \|| |_) | |_____ _____ _____ _____ 
 |_____|_____|_____|_____/ / ___ \|  __/| |_____|_____|_____|_____|
                        /_/_/   \_\_| ||___|                       """)
    
    @staticmethod
    def load_data(thecompanyapi_cached_file_location: str):
        with open(thecompanyapi_cached_file_location, "r") as file:
            return json.load(file)
    
    def save_data(self, which_file: str):
        if which_file == "tca":
            with open(self.thecompanyapi_cached_file_location,"w") as file:
                json.dump(self.thecompanyapi_cached,file)
        elif which_file == "company_domain":
            with open(self.company_domain_file_location,"w") as file:
                json.dump(self.company_domain,file)

    def search_company_domain(self, s: str)-> str:
        if "." in s:
            return s
        if s in self.company_domain:
            return self.company_domain[s]
        else:
            if self.chatgpt_call == 1000:
                self.save_data(which_file="company_domain")
                self.chatgpt_call = 0
            
            print(f"\n{s} - not found in company_domain_cache")
            prompt = f"""
            What is the main domain of this company {s}
            Output the main domain only"""

            domain = call_openai(prompt)
            if "Answer:" in domain:
                domain = domain.replace("Answer: ","")

            print(f"ChatGPT suggests domain is --> {domain}")

            self.company_domain[s] = domain

            return domain.strip()
    
    def fill_domain(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Sometimes, domain is missing data or has corrupt domains
        This can be addressed by replacing the bad domain row with 
        data from the company_name_cleaned column

        The company API will also takes straight up company names

        In:  pd.DataFrame[['website_cleaned','company_name_cleaned']]
        Out: pd.DataFrame[['website_cleaned','company_name_cleaned','website_filled']]
        """
        # fill na
        # Let row below be a lesson on why you shouldn't ignore "SettingWithCopyWarming"
        # df.insert(0,"website_filled",df.loc[:,'website_cleaned'].fillna(df.loc[:,'company_name_cleaned']))
        df.loc[df['website_cleaned'].isna(), 'website_filled'] = df['company_name_cleaned'].apply(lambda x: str(x).lower())

        # fill short
        # this is the official best practice for not getting SettingWithCopy warning
        df.loc[df['website_filled'].str.len() < 4, 'website_filled'] = df['company_name_cleaned'].apply(lambda x: str(x).lower())
        df.loc[df['website_filled'].isna(),'website_filled'] = df['website_cleaned']

        # search company domain based on company name
        df['website_filled'] = df['website_filled'].apply(lambda x: self.search_company_domain(x))

        return df
    
    def call_the_companyapi(self, company_domain: str):
        """
        Intakes company domain (such as www.google.com)

        and extracts data from ther request and returns the 
        """

        if self.count == 1000:
            print("\n---\n1000 new company names have been processed...Saving\n---\n")
            self.save_data(which_file="tca")
            self.count = 0

        self.count += 1

        res = requests.get(
            f"https://api.thecompaniesapi.com/v1/companies/{company_domain}?token={self.thecompanyapi_token}"
        ).json()

        # If domain is not found, then the response is an empty dict.
        if res == {}:
            self.thecompanyapi_cached[company_domain] = {}
            return

        try:
            data_we_want = {}

            # extract only columns we want
            for col in self.columns_to_keep:
                data_we_want[col] = res[col]

            # extract social networks data
            for social_url in data_we_want['socialNetworks']:
                data_we_want[social_url] = data_we_want["socialNetworks"][social_url]
            
            # This column is dropped because we don't need it anymore
            del data_we_want["socialNetworks"]
            
            # save data onto the cache
            self.thecompanyapi_cached[company_domain] = data_we_want
        except KeyError as e:
            print(f"KeyError: \n {e} \n for company name: {company_domain}")
            self.thecompanyapi_cached[company_domain] = {}
            data_we_want = {}

        return data_we_want
    
    def call_the_companyapi_company_name(self, company_domain: str):
        """
        Intake company names

        When domains are missing or corrupted --> it is replaced with company names with no space

        A different API is needed for using with company names
        """
        if self.count == 1000:
            print("\n---\n1000 new company names have been processed...Saving\n---\n")
            self.save_data()
            self.count = 0
        
        self.count += 1

        res = requests.get(
            f"https://api.thecompaniesapi.com/v1/companies/by-name/{company_domain}?token={self.thecompanyapi_token}"
        ).json()

        try:
            if res == {} or res['companies'][0] == {}:
                self.thecompanyapi_cached[company_domain] = {}
                return

            res = res['companies'][0]

            try:
                data_we_want = {}

                # extract only columns we want
                for col in self.columns_to_keep:
                    data_we_want[col] = res[col]

                # extract social networks data
                for social_url in data_we_want['socialNetworks']:
                    data_we_want[social_url] = data_we_want["socialNetworks"][social_url]
                
                # This column is dropped because we don't need it anymore
                del data_we_want["socialNetworks"]
                
                # save data onto the cache
                self.thecompanyapi_cached[company_domain] = data_we_want
            except KeyError as e:
                print(f"KeyError: \n {e} \n for company name: {company_domain}")
                self.thecompanyapi_cached[company_domain] = {}
        except IndexError as e:
            print(f"IndexError:\n{e}\nWhile processing: {company_domain}")
            self.thecompanyapi_cached[company_domain] = {}
            data_we_want = {}
        except KeyError as e:
            print(f"KeyError: \n {e} \n for company name: {company_domain}")
            self.thecompanyapi_cached[company_domain] = {}
            data_we_want = {}

        return data_we_want
    
    def check_company(self, company_domain: str):
        """
        Check if company domain is cached in our database

        If not, use call_the_company function to call api
        The function automatically caches the api call data
        into the self.thecompanyapi_cached dictionary
        """
        if "." in company_domain:
            self.total_domain_count += 1
            if len(company_domain) <= 3:
                return
            elif company_domain in self.thecompanyapi_cached:
                self.total_already_cached += 1
                return
            else:
                print(f"New domain detected --> {company_domain}")
                self.call_the_companyapi(company_domain)
                return
        else:
            self.total_company_name_count += 1
            if len(company_domain) <= 3:
                return
            elif company_domain in self.thecompanyapi_cached:
                self.total_already_cached += 1
                return
            else:
                company_domain = company_domain.replace(" ","")
                print(f"New domain detected --> {company_domain}")
                self.call_the_companyapi_company_name(company_domain)
                return