from inspect import signature
import json
from typing import Any, Callable, Dict, Iterable, Literal, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field, PrivateAttr
import requests

try:
    from duckduckgo_search import DDGS
    import tiktoken
except ImportError:
    ...

QUERY = Field(
    description="Be sure to use keywords that would actually be present in the documents you are searching for",
    examples=[
        "cats dogs",
        '"cats and dogs"',
        "cats -dogs",
        "cats +dogs",
        "cats filetype:pdf",
        "dogs site:example.com",
        "cats -site:example.com",
        "intitle:dogs",
        "inurl:cats",
    ],
)


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def limit_tokens(
    seq: Iterable,
    limit: int,
    stringifier: Optional[Callable[[Any], str]] = None,
    padding: int = 0,
    offset: int = 0,
) -> list:
    out = []
    total_len = 0
    for chunk in seq:
        chunk_len = count_tokens(stringifier(chunk)) if stringifier else count_tokens(chunk)
        if total_len + chunk_len + len(out) * padding + offset > limit:
            break
        else:
            out.append(chunk)
            total_len += chunk_len
    return out


def prune_json(doc):
    if isinstance(doc, list):
        return list(map(prune_json, doc))
    elif isinstance(doc, dict):
        return {k: prune_json(v) for k, v in doc.items() if v and k not in ("Result", "Height", "Width")}
    else:
        return doc


def drop_keys(doc, keys):
    return {k: v for k, v in doc.items() if k not in keys}


def construct_with(before_attribute: Optional[str] = None, after_attribute: Optional[str] = None) -> Callable:
    """
    ```
    >>> @construct_with(after_attribute = "__post_init__")
    ... class SampleModel(BaseModel):
    ...     foo: str
    ...     bar: str
    ... # define method to be called after the constructor
    ... def __post_init__(self, arguments: List[object] = None, knownarguments: Dict[str,object] = None):
    ...     # whatever you want ...
    ```
    """

    def decorate_redefined_constructor(cls):
        # hook the constructor from `cls` which is being decorated.
        constructor = cls.__init__
        before_constr = getattr(cls, before_attribute) if before_attribute else None
        after_constr = getattr(cls, after_attribute) if after_attribute else None

        # redefine `__init__` which is being wrapped with attributes
        def __init__(self, *arg, **kwargs):
            # if there is a specifed attribute name to call before the constructor
            if before_attribute:
                before_constr(self, *arg, **kwargs)
            # inject the hooked constructor
            constructor(self, *arg, **kwargs)
            # if there is a specifed attribute name to call after the constructor
            if after_attribute:
                after_constr(self, *arg, **kwargs)

        # replace `cls.__init__` with redefined `__init__(self, ...)`
        cls.__init__ = __init__
        return cls

    return decorate_redefined_constructor


class CustomTool(StructuredTool):
    toolkit: BaseToolkit

    def __init__(self, *args, **kwargs):
        self.update_forward_refs()
        StructuredTool.__init__(
            self,
            *args,
            name=type(self).__name__,
            description=self.__doc__ or "",
            args_schema=self.Input,
            func=lambda: NotImplemented,
            **kwargs,
        )

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        new_argument_supported = signature(self.func).parameters.get("callbacks")
        if new_argument_supported:
            return self.run(self.Input(*args, **kwargs), callbacks=run_manager.get_child() if run_manager else None)
        else:
            return self.run(self.Input(*args, **kwargs))

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        new_argument_supported = signature(self.func).parameters.get("callbacks")
        if new_argument_supported:
            raise NotImplementedError
        return self.run(self.Input(*args, **kwargs))


@construct_with(after_attribute="model_post_init")
class DuckDuckGoToolkit(BaseToolkit):
    region: str = Field("us-en", description='wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".')
    safesearch: Literal["on", "moderate", "off"] = Field("moderate")
    result_limit: int = 1000
    selected_tools: Optional[list[str]]

    _client: "DDGS" = PrivateAttr()
    __dependencies__ = ["duckduckgo_search"]

    def model_post_init(self):
        self._client = DDGS()

    @tool("Lookup")
    def ddg_lookup(self, query: str) -> Dict:
        """
        Looks up entities using DuckDuckGo's Quick Answers box, which aggregates structured metadata from around the web.
        Often returns nothing at all.
        """
        params = {"q": query, "format": "json", "region": self.region}
        result = prune_json(
            drop_keys(
                requests.get("https://api.duckduckgo.com", params=params).json(),
                ["ImageHeight", "ImageWidth", "meta", "Type"],
            )
        )
        if "Infobox" in result:
            result["Infobox"] = {
                **{c["label"]: c["value"] for c in result["Infobox"]["content"]},
                **{c["label"]: c["value"] for c in result["Infobox"]["meta"]},
                **drop_keys(result["Infobox"], ["content", "meta"]),
            }
        return result

    class Videos(CustomTool):
        """Search for videos. Returns a list of JSON objects that you should summarize into Markdown with citations."""

        class Input(BaseModel):
            keywords: str = QUERY
            timelimit: Optional[Literal["d", "w", "m"]] = Field(description="Get results from the last day, week, or month")
            resolution: Optional[Literal["high", "standard"]]
            duration: Optional[Literal["short", "medium", "long"]]
            license_videos: Optional[Literal["creativeCommon", "youtube"]]

        def run(self, input: Input) -> list:
            return limit_tokens(
                (
                    prune_json(drop_keys(vid, ["images", "embed_html", "embed_url", "image_token", "provider"]))
                    for vid in self.toolkit._client.videos(
                        **input.dict(), region=self.toolkit.region, safesearch=self.toolkit.safesearch
                    )
                ),
                limit=self.toolkit.result_limit,
                stringifier=json.dumps,
                padding=2,
                offset=2,
            )

    class News(CustomTool):
        """Search for news. Returns a list of JSON objects that you should summarize into Markdown with citations."""

        class Input(BaseModel):
            keywords: str
            timelimit: Optional[Literal["d", "w", "m"]] = Field(description="Get results from the last day, week, or month")

        def run(self, input: Input) -> list:
            return limit_tokens(
                (
                    drop_keys(article, ["image"])
                    for article in self.toolkit._client.news(
                        **input.dict(), region=self.toolkit.region, safesearch=self.toolkit.safesearch
                    )
                ),
                limit=self.toolkit.result_limit,
                stringifier=json.dumps,
                padding=2,
                offset=2,
            )

    class Maps(CustomTool):
        """Geographic search for POIs and addresses."""

        class Input(BaseModel):
            keywords: str
            place: Optional[str] = Field(description="if set, the other parameters are not used.")
            street: Optional[str] = Field(description="house number/street.")
            city: Optional[str]
            county: Optional[str]
            state: Optional[str]
            country: Optional[str]
            postalcode: Optional[str]
            latitude: Optional[int] = Field(description="geographic coordinate (north-south position)")
            longitude: Optional[int] = Field(
                description="""
                geographic coordinate (east-west position);
                if latitude and longitude are set, the other parameters are not used.
                """
            )
            radius: Optional[int] = Field(0, description="expand the search square by the distance in kilometers. Defaults to 0.")

        def run(self, input: Input) -> list:
            return limit_tokens(
                self.toolkit._client.maps(**input.dict(), region=self.toolkit.region, safesearch=self.toolkit.safesearch),
                limit=self.toolkit.result_limit,
                stringifier=json.dumps,
                padding=2,
                offset=2,
            )

    class Translate(CustomTool):
        toolkit: "DuckDuckGoToolkit"

        class Input(BaseModel):
            keywords: str = Field(description="string or a list of strings to translate")
            from_: Optional[str] = Field(description="translate from (defaults to autodetect)")
            to: Optional[str] = Field(default="en", examples=["de"], description='what language to translate. Defaults to "en"')

        def run(self, input: Input) -> Dict:
            return self.toolkit._client.translate(**input.dict())

    @tool("Suggestions")
    def ddg_suggestions(self, keywords: str) -> list:
        """Generate list of search completions; useful for refining short keyword search queries"""
        return self._client.suggestions(keywords, region=self.region)

    def get_tools(self) -> list[BaseTool]:
        allowed_tools = [
            self.ddg_lookup,
            self.Videos(toolkit=self),
            self.News(toolkit=self),
            self.Translate(toolkit=self),
            self.ddg_suggestions,
        ]
        tools: list[BaseTool] = []
        if self.selected_tools is None:
            return allowed_tools
        for t in allowed_tools:
            if t.name in self.selected_tools:
                tools.append(t)
        return tools
