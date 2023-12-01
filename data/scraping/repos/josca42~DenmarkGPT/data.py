import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import numpy as np
import faiss
import networkx as nx
import cohere
import openai
import pickle

load_dotenv()

AVATARS = {"user": "🧑‍💻", "assistant": "🤖"}

DATA_DIR = Path(os.environ["DATA_DIR"])
EVAL_DIR = DATA_DIR / "evaluations"

config = os.environ

LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])
openai.api_key = os.environ.get("OPENAI_API_KEY")

gdf = gpd.read_file(DATA_DIR / "maps/kommune_og_region.shp")
REGIONER_ID = [x for x in gdf["geo_id"] if x[0] == "0"]
KOMMUNER_ID = [x for x in gdf["geo_id"] if x[0] != "0"]
ALL_GEO_IDS = set(REGIONER_ID + KOMMUNER_ID)

LLM_cohere = cohere.Client(os.environ["COHERE_API_KEY"])

TABLE_INFO_EN_DIR = DATA_DIR / "tables_info_en"
TABLE_INFO_DA_DIR = DATA_DIR / "tables_info_da"


G = nx.read_gml(DATA_DIR / "table_network" / "subjects_graph.gml")
table2node = pickle.load(open(DATA_DIR / "table_network" / "table2node.pkl", "rb"))


DST_SUBJECTS_INDEX_0_1 = {
    "Borgere": [
        "Befolkning",
        "Husstande, familier og børn",
        "Flytninger",
        "Boligforhold",
        "Sundhed",
        "Demokrati",
        "Folkekirken",
        "Navne",
    ],
    "Arbejde og indkomst": [
        "Befolkningens arbejdsmarkedsstatus",
        "Beskæftigede",
        "Arbejdsløse",
        "Fravær og arbejdskonflikter",
        "Indkomst og løn",
        "Formue",
    ],
    "Økonomi": [
        "Nationalregnskab",
        "Offentlig økonomi",
        "Betalingsbalance og udenrigshandel",
        "Prisindeks",
        "Forbrug",
        "Ejendomme",
        "Valutakurser, renter og værdipapirer",
        "Digitale betalinger",
    ],
    "Sociale forhold": [
        "Offentligt forsørgede",
        "Social støtte",
        "Kriminalitet",
        "Levevilkår",
    ],
    "Uddannelse og forskning": [
        "Befolkningens uddannelsesstatus",
        "Fuldtidsuddannelser",
        "Veje gennem uddannelsessystemet",
        "Voksen- og efteruddannelse",
        "Forskning, udvikling og innovation",
    ],
    "Erhvervsliv": [
        "Erhvervslivets struktur",
        "Erhvervslivets økonomi",
        "Internationale virksomheder",
        "Landbrug, gartneri og skovbrug",
        "Fiskeri og akvakultur",
        "Industri",
        "Byggeri og anlæg",
        "Handel",
        "Overnatninger og rejser",
        "Finansiel sektor",
        "Serviceerhverv",
        "Konjunkturbarometre for erhvervene",
    ],
    "Transport": [
        "Transportmidler",
        "Trafik og infrastruktur",
        "Persontransport",
        "Godstransport",
        "Trafikulykker",
    ],
    "Kultur og fritid": [
        "Museer og zoologiske haver",
        "Kulturarv",
        "Biblioteker",
        "Nyhedsmedier og litteratur",
        "Film og teater",
        "Musik",
        "Idræt",
        "Digital adfærd og kulturvaner",
        "Kulturområdets uddannelse, økonomi og beskæftigelse",
    ],
    "Miljø og energi": [
        "Areal",
        "Energiforbrug og energipriser",
        "Grønt Nationalregnskab",
        "Økologi",
    ],
    "Tværgående": [
        "Statistisk Årbog",
        "Statistisk Tiårsoversigt",
        "Danmark i Tal",
        "Nordic Statistical Yearbook",
        "Publikationer",
        "De kommunale serviceindikatorer",
    ],
}
