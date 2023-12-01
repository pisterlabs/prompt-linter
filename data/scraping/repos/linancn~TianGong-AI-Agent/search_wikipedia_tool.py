from typing import Optional, Type

import streamlit as st
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS
from pydantic import BaseModel


class SearchWikipediaTool(BaseTool):
    name = "search_wikipedia_tool"
    description = "Search Wikipedia for results."

    llm_model = st.secrets["llm_model"]
    langchain_verbose = st.secrets["langchain_verbose"]

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def wiki_query_func_calling_chain(self):
        func_calling_json_schema = {
            "title": "identify_query_language_to_search_Wikipedia",
            "description": "Accurately identifying the language of the query to search Wikipedia.",
            "type": "object",
            "properties": {
                "language": {
                    "title": "Language",
                    "description": "The accurate language of the query",
                    "type": "string",
                    "enum": [
                        "en",
                        "es",
                        "fr",
                        "de",
                        "ru",
                        "zh",
                        "pt",
                        "ar",
                        "it",
                        "ja",
                        "tr",
                        "id",
                        "simple",
                        "nl",
                        "pl",
                        "fa",
                        "he",
                        "vi",
                        "sv",
                        "ko",
                        "hi",
                        "uk",
                        "cs",
                        "ro",
                        "no",
                        "fi",
                        "hu",
                        "da",
                        "ca",
                        "th",
                        "bn",
                        "el",
                        "sr",
                        "bg",
                        "ms",
                        "hr",
                        "az",
                        "zh-yue",
                        "sk",
                        "sl",
                        "ta",
                        "arz",
                        "eo",
                        "sh",
                        "et",
                        "lt",
                        "ml",
                        "la",
                        "ur",
                        "af",
                        "mr",
                        "bs",
                        "sq",
                        "ka",
                        "eu",
                        "gl",
                        "hy",
                        "tl",
                        "be",
                        "kk",
                        "nn",
                        "ang",
                        "te",
                        "lv",
                        "ast",
                        "my",
                        "mk",
                        "ceb",
                        "sco",
                        "uz",
                        "als",
                        "zh-classical",
                        "is",
                        "mn",
                        "wuu",
                        "cy",
                        "kn",
                        "be-tarask",
                        "br",
                        "gu",
                        "an",
                        "bar",
                        "si",
                        "ne",
                        "sw",
                        "lb",
                        "zh-min-nan",
                        "jv",
                        "ckb",
                        "ga",
                        "war",
                        "ku",
                        "oc",
                        "nds",
                        "yi",
                        "ia",
                        "tt",
                        "fy",
                        "pa",
                        "azb",
                        "am",
                        "scn",
                        "lmo",
                        "gan",
                        "km",
                        "tg",
                        "ba",
                        "as",
                        "sa",
                        "ky",
                        "io",
                        "so",
                        "pnb",
                        "ce",
                        "vec",
                        "vo",
                        "mzn",
                        "or",
                        "cv",
                        "bh",
                        "pdc",
                        "hif",
                        "hak",
                        "mg",
                        "ht",
                        "ps",
                        "su",
                        "nap",
                        "qu",
                        "fo",
                        "bo",
                        "li",
                        "rue",
                        "se",
                        "nds-nl",
                        "gd",
                        "tk",
                        "yo",
                        "diq",
                        "pms",
                        "new",
                        "ace",
                        "vls",
                        "bat-smg",
                        "eml",
                        "cu",
                        "bpy",
                        "dv",
                        "hsb",
                        "sah",
                        "os",
                        "chr",
                        "sc",
                        "wa",
                        "szl",
                        "ha",
                        "ksh",
                        "bcl",
                        "nah",
                        "mt",
                        "co",
                        "ug",
                        "lad",
                        "cdo",
                        "pam",
                        "arc",
                        "crh",
                        "rm",
                        "zu",
                        "gv",
                        "frr",
                        "ab",
                        "got",
                        "iu",
                        "ie",
                        "xmf",
                        "cr",
                        "dsb",
                        "mi",
                        "gn",
                        "min",
                        "lo",
                        "sd",
                        "rmy",
                        "pcd",
                        "ilo",
                        "ext",
                        "sn",
                        "ig",
                        "nv",
                        "haw",
                        "csb",
                        "ay",
                        "jbo",
                        "frp",
                        "map-bms",
                        "lij",
                        "ch",
                        "vep",
                        "glk",
                        "tw",
                        "kw",
                        "bxr",
                        "wo",
                        "udm",
                        "av",
                        "pap",
                        "ee",
                        "cbk-zam",
                        "kv",
                        "fur",
                        "mhr",
                        "fiu-vro",
                        "bjn",
                        "roa-rup",
                        "gag",
                        "tpi",
                        "mai",
                        "stq",
                        "kab",
                        "bug",
                        "kl",
                        "nrm",
                        "mwl",
                        "bi",
                        "zea",
                        "ln",
                        "xh",
                        "myv",
                        "rw",
                        "nov",
                        "pfl",
                        "kaa",
                        "chy",
                        "roa-tara",
                        "pih",
                        "lfn",
                        "kg",
                        "bm",
                        "mrj",
                        "lez",
                        "za",
                        "om",
                        "ks",
                        "ny",
                        "krc",
                        "sm",
                        "st",
                        "pnt",
                        "dz",
                        "to",
                        "ary",
                        "tn",
                        "xal",
                        "gom",
                        "kbd",
                        "ts",
                        "rn",
                        "tet",
                        "mdf",
                        "ti",
                        "hyw",
                        "fj",
                        "tyv",
                        "ff",
                        "ki",
                        "ik",
                        "koi",
                        "lbe",
                        "jam",
                        "ss",
                        "lg",
                        "pag",
                        "tum",
                        "ve",
                        "ban",
                        "srn",
                        "ty",
                        "ltg",
                        "pi",
                        "sat",
                        "ady",
                        "olo",
                        "nso",
                        "sg",
                        "dty",
                        "din",
                        "tcy",
                        "gor",
                        "kbp",
                        "avk",
                        "lld",
                        "atj",
                        "inh",
                        "shn",
                        "nqo",
                        "mni",
                        "smn",
                        "mnw",
                        "dag",
                        "szy",
                        "gcr",
                        "awa",
                        "alt",
                        "shi",
                        "mad",
                        "skr",
                        "ami",
                        "trv",
                        "nia",
                        "tay",
                        "pwn",
                        "guw",
                        "pcm",
                        "kcg",
                        "blk",
                        "guc",
                        "anp",
                        "gur",
                        "fat",
                        "gpe",
                    ],
                }
            },
            "required": ["language"],
        }

        prompt_func_calling_msgs = [
            SystemMessage(
                content="""You are a world class algorithm for accurately identifying the language of the query to search Wikipedia, strictly follow the language mapping: {"English": "en", "Spanish": "es", "French": "fr", "German": "de", "Russian": "ru", "Chinese": "zh", "Portuguese": "pt", "Arabic": "ar", "Italian": "it", "Japanese": "ja", "Turkish": "tr", "Indonesian": "id", "Simple English": "simple", "Dutch": "nl", "Polish": "pl", "Persian": "fa", "Hebrew": "he", "Vietnamese": "vi", "Swedish": "sv", "Korean": "ko", "Hindi": "hi", "Ukrainian": "uk", "Czech": "cs", "Romanian": "ro", "Norwegian": "no", "Finnish": "fi", "Hungarian": "hu", "Danish": "da", "Catalan": "ca", "Thai": "th", "Bangla": "bn", "Greek": "el", "Serbian": "sr", "Bulgarian": "bg", "Malay": "ms", "Croatian": "hr", "Azerbaijani": "az", "Cantonese": "zh-yue", "Slovak": "sk", "Slovenian": "sl", "Tamil": "ta", "Egyptian Arabic": "arz", "Esperanto": "eo", "Serbo-Croatian": "sh", "Estonian": "et", "Lithuanian": "lt", "Malayalam": "ml", "Latin": "la", "Urdu": "ur", "Afrikaans": "af", "Marathi": "mr", "Bosnian": "bs", "Albanian": "sq", "Georgian": "ka", "Basque": "eu", "Galician": "gl", "Armenian": "hy", "Tagalog": "tl", "Belarusian": "be", "Kazakh": "kk", "Norwegian Nynorsk": "nn", "Old English": "ang", "Telugu": "te", "Latvian": "lv", "Asturian": "ast", "Burmese": "my", "Macedonian": "mk", "Cebuano": "ceb", "Scots": "sco", "Uzbek": "uz", "Swiss German": "als", "Literary Chinese": "zh-classical", "Icelandic": "is", "Mongolian": "mn", "Wu Chinese": "wuu", "Welsh": "cy", "Kannada": "kn", "Belarusian (Taraškievica orthography)": "be-tarask", "Breton": "br", "Gujarati": "gu", "Aragonese": "an", "Bavarian": "bar", "Sinhala": "si", "Nepali": "ne", "Swahili": "sw", "Luxembourgish": "lb", "Min Nan Chinese": "zh-min-nan", "Javanese": "jv", "Central Kurdish": "ckb", "Irish": "ga", "Waray": "war", "Kurdish": "ku", "Occitan": "oc", "Low German": "nds", "Yiddish": "yi", "Interlingua": "ia", "Tatar": "tt", "Western Frisian": "fy", "Punjabi": "pa", "South Azerbaijani": "azb", "Amharic": "am", "Sicilian": "scn", "Lombard": "lmo", "Gan Chinese": "gan", "Khmer": "km", "Tajik": "tg", "Bashkir": "ba", "Assamese": "as", "Sanskrit": "sa", "Kyrgyz": "ky", "Ido": "io", "Somali": "so", "Western Punjabi": "pnb", "Chechen": "ce", "Venetian": "vec", "Volapük": "vo", "Mazanderani": "mzn", "Odia": "or", "Chuvash": "cv", "Bhojpuri": "bh", "Pennsylvania German": "pdc", "Fiji Hindi": "hif", "Hakka Chinese": "hak", "Malagasy": "mg", "Haitian Creole": "ht", "Pashto": "ps", "Sundanese": "su", "Neapolitan": "nap", "Quechua": "qu", "Faroese": "fo", "Tibetan": "bo", "Limburgish": "li", "Rusyn": "rue", "Northern Sami": "se", "Low Saxon": "nds-nl", "Scottish Gaelic": "gd", "Turkmen": "tk", "Yoruba": "yo", "Zazaki": "diq", "Piedmontese": "pms", "Newari": "new", "Achinese": "ace", "West Flemish": "vls", "Samogitian": "bat-smg", "Emiliano-Romagnolo": "eml", "Church Slavic": "cu", "Bishnupriya": "bpy", "Divehi": "dv", "Upper Sorbian": "hsb", "Yakut": "sah", "Ossetic": "os", "Cherokee": "chr", "Sardinian": "sc", "Walloon": "wa", "Silesian": "szl", "Hausa": "ha", "Colognian": "ksh", "Central Bikol": "bcl", "Nāhuatl": "nah", "Maltese": "mt", "Corsican": "co", "Uyghur": "ug", "Ladino": "lad", "Min Dong Chinese": "cdo", "Pampanga": "pam", "Aramaic": "arc", "Crimean Tatar": "crh", "Romansh": "rm", "Zulu": "zu", "Manx": "gv", "Northern Frisian": "frr", "Abkhazian": "ab", "Gothic": "got", "Inuktitut": "iu", "Interlingue": "ie", "Mingrelian": "xmf", "Cree": "cr", "Lower Sorbian": "dsb", "Māori": "mi", "Guarani": "gn", "Minangkabau": "min", "Lao": "lo", "Sindhi": "sd", "Vlax Romani": "rmy", "Picard": "pcd", "Iloko": "ilo", "Extremaduran": "ext", "Shona": "sn", "Igbo": "ig", "Navajo": "nv", "Hawaiian": "haw", "Kashubian": "csb", "Aymara": "ay", "Lojban": "jbo", "Arpitan": "frp", "Basa Banyumasan": "map-bms", "Ligurian": "lij", "Chamorro": "ch", "Veps": "vep", "Gilaki": "glk", "Twi": "tw", "Cornish": "kw", "Russia Buriat": "bxr", "Wolof": "wo", "Udmurt": "udm", "Avaric": "av", "Papiamento": "pap", "Ewe": "ee", "Chavacano": "cbk-zam", "Komi": "kv", "Friulian": "fur", "Eastern Mari": "mhr", "Võro": "fiu-vro", "Banjar": "bjn", "Aromanian": "roa-rup", "Gagauz": "gag", "Tok Pisin": "tpi", "Maithili": "mai", "Saterland Frisian": "stq", "Kabyle": "kab", "Buginese": "bug", "Kalaallisut": "kl", "Norman": "nrm", "Mirandese": "mwl", "Bislama": "bi", "Zeelandic": "zea", "Lingala": "ln", "Xhosa": "xh", "Erzya": "myv", "Kinyarwanda": "rw", "Novial": "nov", "Palatine German": "pfl", "Kara-Kalpak": "kaa", "Cheyenne": "chy", "Tarantino": "roa-tara", "Norfuk / Pitkern": "pih", "Lingua Franca Nova": "lfn", "Kongo": "kg", "Bambara": "bm", "Western Mari": "mrj", "Lezghian": "lez", "Zhuang": "za", "Oromo": "om", "Kashmiri": "ks", "Nyanja": "ny", "Karachay-Balkar": "krc", "Samoan": "sm", "Southern Sotho": "st", "Pontic": "pnt", "Dzongkha": "dz", "Tongan": "to", "Moroccan Arabic": "ary", "Tswana": "tn", "Kalmyk": "xal", "Goan Konkani": "gom", "Kabardian": "kbd", "Tsonga": "ts", "Rundi": "rn", "Tetum": "tet", "Moksha": "mdf", "Tigrinya": "ti", "Western Armenian": "hyw", "Fijian": "fj", "Tuvinian": "tyv", "Fula": "ff", "Kikuyu": "ki", "Inupiaq": "ik", "Komi-Permyak": "koi", "Lak": "lbe", "Jamaican Creole English": "jam", "Swati": "ss", "Ganda": "lg", "Pangasinan": "pag", "Tumbuka": "tum", "Venda": "ve", "Balinese": "ban", "Sranan Tongo": "srn", "Tahitian": "ty", "Latgalian": "ltg", "Pali": "pi", "Santali": "sat", "Adyghe": "ady", "Livvi-Karelian": "olo", "Northern Sotho": "nso", "Sango": "sg", "Doteli": "dty", "Dinka": "din", "Tulu": "tcy", "Gorontalo": "gor", "Kabiye": "kbp", "Kotava": "avk", "Ladin": "lld", "Atikamekw": "atj", "Ingush": "inh", "Shan": "shn", "N’Ko": "nqo", "Manipuri": "mni", "Inari Sami": "smn", "Mon": "mnw", "Dagbani": "dag", "Sakizaya": "szy", "Guianan Creole": "gcr", "Awadhi": "awa", "Southern Altai": "alt", "Tachelhit": "shi", "Madurese": "mad", "Saraiki": "skr", "Amis": "ami", "Taroko": "trv", "Nias": "nia", "Tayal": "tay", "Paiwan": "pwn", "Gun": "guw", "Nigerian Pidgin": "pcm", "Tyap": "kcg", "Pa"O": "blk", "Wayuu": "guc", "Angika": "anp", "Frafra": "gur", "Fanti": "fat", "Ghanaian Pidgin": "gpe"}"""
            ),
            HumanMessage(content="The query:"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]

        prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

        llm_func_calling = ChatOpenAI(
            model_name=self.llm_model, temperature=0, streaming=False
        )

        query_func_calling_chain = create_structured_output_chain(
            output_schema=func_calling_json_schema,
            llm=llm_func_calling,
            prompt=prompt_func_calling,
            verbose=self.langchain_verbose,
        )

        return query_func_calling_chain

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        language = self.wiki_query_func_calling_chain().run(query)["language"]
        docs = WikipediaLoader(
            query=query, lang=language, load_max_docs=3, load_all_available_meta=True
        ).load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=10
        )
        chunks = []

        for doc in docs:
            chunk = text_splitter.create_documents(
                [doc.page_content],
                metadatas=[
                    {
                        "source": "[{}]({})".format(
                            doc.metadata["title"], doc.metadata["source"]
                        )
                    }
                ],
            )
            chunks.extend(chunk)

        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)

        result_docs = faiss_db.similarity_search(query, k=16)

        docs_list = []

        for doc in result_docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        language = self.wiki_query_func_calling_chain().run(query)["language"]
        docs = WikipediaLoader(
            query=query, lang=language, load_max_docs=3, load_all_available_meta=True
        ).load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=10
        )
        chunks = []

        for doc in docs:
            chunk = text_splitter.create_documents(
                [doc.page_content],
                metadatas=[
                    {
                        "source": "[{}]({})".format(
                            doc.metadata["title"], doc.metadata["source"]
                        )
                    }
                ],
            )
            chunks.extend(chunk)

        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)

        result_docs = faiss_db.similarity_search(query, k=16)

        docs_list = []

        for doc in result_docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list
