import pandas as pd
import requests
import msal
from typing import Optional, Type

from langchain.agents import create_pandas_dataframe_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from prompt_templates.custom_templates import PANDAS_AGENT_PREFIX

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from dotenv import dotenv_values

# ENV = dotenv_values("../.env")
ENV = dotenv_values(".env")



class ListDataExtract():

   ENDPOINT = "https://graph.microsoft.com/v1.0"

   def __init__(
       self,
       client_id: str,
       client_secret: str,
       authority: str,
       scopes: list,
       site_id: str,
       list_id: str,
       usecols: dict
   ):
      self.client_id = client_id
      self.client_secret = client_secret
      self.authority = authority
      self.scopes = scopes
      self.site_id = site_id
      self.list_id = list_id
      self.usecols = usecols

   def get_access_token(self):
      """Obtiene token de acceso de MSAL bajo el flujo de credenciales de cliente (o aplicacion).

        Raises:
            Exception: Error al obtener el token de acceso.

        Returns:
            str: Token de acceso (access token) de MSAL.
      """
      app = msal.ConfidentialClientApplication(
          self.client_id,
          authority=self.authority,
          client_credential=self.client_secret
      )

      result = app.acquire_token_for_client(scopes=self.scopes)
      if 'access_token' in result:
         print(result['access_token'])
         return result['access_token']
      else:
         raise Exception(f"Error al obtener el token de acceso: {result}")

   def get_all_sharepoint_items(self) -> list:
      """Obtiene los items de la lista del sharepoint indicada.

      Raises:
         Exception: Error en la peticion para la obtencion de intems de la lista de sharepoint.

      Returns:
         list: Items de la lista de sharepoint en cuestion.
      """

      url = f"{self.ENDPOINT}/sites/{self.site_id}/lists/{self.list_id}/items?expand=fields"
      headers = {"Authorization": f"Bearer {self.get_access_token()}"}
      items = []

      while url:
         response = requests.get(url, headers=headers)
         if response.status_code == 200:
            json_response = response.json()
            items.extend(json_response["value"])

            url = json_response.get("@odata.nextLink", None)
         else:
            raise Exception(
                f"Error al obtener los elementos de SharePoint: {response.text}")

      return items

   def get_dataframe(self) -> pd.DataFrame:
      items_list = []

      all_sp_items = self.get_all_sharepoint_items()

      for item in all_sp_items:
         items_list.append(item['fields'])

      usecols = list(self.usecols.keys())
      items_df = pd.DataFrame(items_list)[usecols]

      return items_df


class CustomPandasTool(BaseTool):
   name = "custom_pandas_tool"
   description = """
                  Usar cuando el usuario desee conocer lo siguiente de su requerimiento:
                  - El estado.
                  - Negociador.
                  - El rol Asesor.
                  - Fecha de radicación.
                  - La fecha de asignación o delegación.
                  - Familia o categoría.
                  - Cómo se va a gestionar el requerimieto (Estado).
                  - Proveedores conocidos
                  - Detalle del producto o servicio.
                  - El CW a renovar (id del contrato).
                  - Nombre de la iniciativa.
                 """
   pandas_llm = OpenAI(model_name="text-davinci-003", temperature=0,
                       verbose=True, openai_api_key=ENV['OPENAI_API_KEY'])

   def _run(self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
      """Use the tool."""
      # df = pd.read_csv("data/data_test.csv")
      df = pd.read_excel("data/data_test.xlsx")

      pandas_agent = create_pandas_dataframe_agent(
          llm=self.pandas_llm,
          df=df,
          verbose=True,
          agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
          prefix=PANDAS_AGENT_PREFIX
      )

      return pandas_agent.run(prompt)
