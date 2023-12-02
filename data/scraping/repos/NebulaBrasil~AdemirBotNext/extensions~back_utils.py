import interactions
from interactions import Client, Extension, Guild, Modal, ModalContext, ParagraphText, ShortText
import openai
import config
import pymongo

openai.api_key = config.OPENAPI_TOKEN

# Comando para obter o status do banco de dados
class DataBaseStatus(Extension):
    def __init__(self, client: Client) -> None:
        self.client: Client = client

    def get_database_status(self):
        try:
            client = pymongo.MongoClient(config.MONGO_URI, tls=True, tlsCertificateKeyFile=config.CERTIFICATE_FILE)
            client.admin.command('ping')
            return 'Online'
        except Exception as e:
            return 'Offline'
        
    
    @interactions.slash_command(
            name="dbstatus", 
            description="Retorna o status do banco de dados",
            default_member_permissions=interactions.Permissions.ADMINISTRATOR
    )
    async def dbstatus(self,ctx: interactions.SlashContext):
        status = self.get_database_status()
        await ctx.send(f'Database: **{status}**')


def setup(client):
    DataBaseStatus(client)
