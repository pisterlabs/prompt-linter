from langchain import LLMChain, OpenAI, PromptTemplate
from agent.utils import image_generator


class AestheticModel:
    def __init__(self, client, config):
        self.client = client
        self.config = config

    # This is an LLM which translates an aesthetic into brightness and color values.
    def aesthetic_to_image(self, aesthetic):
        hsv = self.aesthetic_to_json(aesthetic)
        image = image_generator.ImageGenerator().generate_image(hsv)
        print("uploading image..")
        self.client.upload_image(
            self.config.ZULIP_STREAM, self.config.ZULIP_TOPIC, image
        )
        return ""
        # loop = asyncio.get_event_loop()
        # loop.run_until_complete(cosmos.CosmosSDKClient().execute_wasm_msg(output))

    def aesthetic_to_json(self, aesthetic):
        template = """
        Take an aesthetic and translate it into an HSV value. Format the response as JSON where there is a different key for each variable:
        hue, saturation, and value. Each value must be an integer. 
        What is a good brightness and color that represents {aesthetic}?
        """
        prompt = PromptTemplate(
            input_variables=["aesthetic"],
            template=template,
        )

        llm = OpenAI(temperature=0.9)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(aesthetic)


def format_aesthetic_msg(msg):
    msg = msg.replace("\n", "")
    return {
        "propose": {
            "msg": {
                "propose": {
                    "title": "propose microworld configuration",
                    "description": f"this is a mock message, will be a real one later: {msg}",
                    "msgs": [],
                }
            }
        }
    }
