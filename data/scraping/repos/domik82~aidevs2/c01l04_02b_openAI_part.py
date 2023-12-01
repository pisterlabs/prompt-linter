from dotenv import load_dotenv, find_dotenv
from common.logger_setup import configure_logger


from langchain.chains import (
    OpenAIModerationChain,
)


# please return 0/1 for each sentence to classify if this is content that requires moderation

input = [
    "majonez Winiary jest lepszy od Kieleckiego",
    "ten gość musi zginąć. Nie pozwole sobię na obrażanie mnie.",
]


load_dotenv(find_dotenv())
log = configure_logger("moderation_openAI")

moderation_chain = OpenAIModerationChain()

moderation_chain_error = OpenAIModerationChain(error=True)
validated_list = []
for element in list(input):
    try:
        moderation_chain_error.run(element)
        validated_list.append(0)
    except ValueError:
        validated_list.append(1)
    except Exception as e:
        log.error(f"Exception: {e}")

log.info(f"validated_list: {validated_list}")
