from services.claude.claude_api import AnthropicAPI
from services.salesforce.salesforce_api_client import SalesforceAPIClient
from services.salesforce.salesforce_config import AUTH_URL, SECURITY_TOKEN


class Main():
    def __init__(self):
        self.anthropic_api = AnthropicAPI()
        self.salesforce_api = SalesforceAPIClient(access_token=SECURITY_TOKEN, instance_url=AUTH_URL)

    def run(self):
        self.token = self.salesforce_api.get_access_token()
        print(self.token)
        return self.token

if __name__ == "__main__":
    main = Main()
    main.run()
