import sys
from pprint import pprint
pprint(sys.path)

from pathlib import Path
_pwd = Path(__file__).absolute()
_module_path = _pwd.parent.parent
print(_module_path)
sys.path.append(str(_module_path))
pprint(sys.path)

from dotenv import load_dotenv
load_dotenv()

# from module.tool import top_google_results, search_google, search_serp
from module.tools._tools import search_google_serp
from langchain.tools import Tool
import pprint as pp

# tool = Tool(
#     name = "Intermediate Answer",
#     description="Search Google for recent results.",
#     func=top_google_results
# )

# tool = Tool(
#     name = "I'm Feeling Lucky",
#     description="Search Google and return the first result.",
#     func=search_google.run
# )

# tool = Tool(
#         name="Intermediate Answer",
#         func=search_serp.run,
#         description="useful for when you need to ask with search"
# )

tool = Tool(
        name="Intermediate Answer",
        func=search_google_serp.run,
        description="useful for when you need to ask with search"
)

_re = tool.run("Who is the U.S. Navy admiral that David Chanoff collaborated with?")

print(type(_re))
pp.pprint(_re)
