import streamlit as st
from streamlit import session_state as ss
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
import  streamlit_toggle as tog
from streamlit_elements import elements, mui, html, dashboard, editor, lazy, sync
import pandas as pd
import json
import requests
import openai
import stock_api
from PIL import Image
import os

primary_color = "#91b36b"
background_color = "#FFFFFF"
secondary_background_color = "#F0F2F6"

companies = [
    {
        "name": "Pepsico",
        "symbol": "PEP",
    },
    {
        "name": "Fisker",
        "symbol": "FSR",
    },
    {
        "name": "General Mills",
        "symbol": "GIS",
    },
    {
        "name": "Ford",
        "symbol": "F",
    },
]

# load all json data from output_data folder
def load_json_data():
    data = []
    for filename in os.listdir("output_data"):
        if filename.endswith(".json"):
            with open("output_data/" + filename) as f:
                temp_data = json.load(f)
                # change qa_pairs list to map of category to qa_pair
                qa_pairs = {}
                for qa_pair in temp_data["qa_pairs"]:
                    if qa_pair["category"] not in qa_pairs:
                        qa_pairs[qa_pair["category"]] = []
                    # for each qa_pair, order the qa_pair["answers"] by confidence
                    qa_pair["answers"] = sorted(qa_pair["answers"], key=lambda x: x["confidence"], reverse=True)
                    qa_pairs[qa_pair["category"]].append(qa_pair)

                temp_data["qa_pairs"] = qa_pairs
                data.append(temp_data)
    return data

data = load_json_data()

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

available_companies = (company["name"] for company in companies)

### Streamlit app starts here

c1 = st.container()
c2 = st.container()
c3 = st.container()
c4 = st.container()

def get_styled_title(title):
    return st.markdown(f'<p style="color:{primary_color};font-size:45px;border-radius:2%;font-weight:bold">{title}</p>', unsafe_allow_html=True)

def get_symbol_from_company(company):
    for companyInfo in companies:
        if companyInfo["name"] == company:
            return companyInfo["symbol"]

    return ""

def format_market_cap(market_cap):
    if market_cap < 1000:
        rounded = round(market_cap, 2)
        return "$" + str(rounded) + " M"
    elif market_cap < 1000000:
        rounded = round(market_cap / 1000, 2)
        return "$" + str(rounded) + " B"
    else:
        rounded = round(market_cap / 1000000, 2)
        return "$" + str(rounded) + " T"

def get_investment_profile(company):
    with st.expander(label="Investment Profile"):
        company_info = stock_api.get_company_info(symbol=get_symbol_from_company(company))
        # Write and format exchange, country, market capitalization, and industry
        st.write("Exchange: " + company_info["exchange"])
        st.write("Country: " + company_info["country"])
        st.write("Market Capitalization: " + format_market_cap(company_info["marketCapitalization"]))
        st.write("Industry: " + company_info["finnhubIndustry"])

def get_peers(company):
    peers = stock_api.get_peers(symbol=get_symbol_from_company(company))

    ret = []
    for peer in peers:
        st.write(peer)

        company_info = stock_api.get_company_info(symbol=peer)
        ret.append(company_info["name"])
    return ret

def get_confidence_style(qa_pair, bg_color):
    if "confidence" in qa_pair:
        conf = qa_pair["confidence"]
    else:
        conf = 0.5
    color = "rgb({},{},{},{})".format(145, 179, 107, conf)
    return f'radial-gradient(farthest-side at 40% 50%, {color}, {bg_color})'

def get_no_confidence(bg_color):
    color = "rgb({},{},{},{})".format(255, 0, 0, 0.5)
    return f'radial-gradient(farthest-side at 40% 50%, {color}, {bg_color})'

# Share to social media
def compose_share_text():
    params = st.experimental_get_query_params()
    if "companies" in params:
        # Format a returned statement like "Here's a sustainability comparison of Apple, Boeing, and Bayer"
        companies = params["companies"]
        if len(companies) == 1:
            return "Here's a sustainability evaluation of " + companies[0] + ":"
        elif len(companies) == 2:
            return "Here's a sustainability comparison of " + companies[0] + " and " + companies[1] + ":"
        else:
            return "Here's a sustainability comparison of " + ", ".join(companies[:-1]) + ", and " + companies[-1]
    else:
        return "Check out this website to see how sustainable your favourite companies are!"

def compose_curr_url():
    domain = "akhilgupta1093-openai-hackathon-scope3-ungreenwash-app-v8ncns.streamlit.app/"
    queryParams = []
    if "companies" in ss:
        for c in ss.companies:
            cFormatted = "+".join(c.split(" "))
            queryParams.append(f'companies={cFormatted}')
    
    queryStr = ""
    if len(queryParams) > 0:
        queryStr = "?" + "&".join(queryParams)

    # using domain and query params (map of query params), compose the current url
    return "https://" + domain + queryStr

def get_share_text():
    return """
                <div style="display:flex;margin:0px">
                    <div style="margin-top:12px;margin-right:10px">
                        <a class="twitter-share-button"
                            href="https://twitter.com/intent/tweet?text={text}"
                            data-url="{url}">
                        Tweet</a>
                        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
                    </div>
                    <div style="margin-top:12px;margin-right:10px">
                        <a data-size="large" data-url="{url}"/>
                        <script src="https://platform.linkedin.com/in.js" type="text/javascript"> lang: en_US</script>
                        <script type="IN/Share" data-url="{url}"></script>
                    </div>
                </div>
            """.format(text=compose_share_text(), url=compose_curr_url())

def get_share_elements():
    components.html(get_share_text())

# Mock function for now, will be an api call.
def get_company_info(company):
    for companyInfo in data:
        if companyInfo["name"] == company:
            return companyInfo

    return "{}"

def handle_company_select():
    st.experimental_set_query_params(companies=ss.companies)

    for company in ss.companies:
        if company not in ss:
            ss[company] = get_company_info(company)

# Get company info based on query params
params = st.experimental_get_query_params()
if "companies" in params:
    param_companies = params["companies"]
    ss.companies = param_companies
    for company in param_companies:
        if company not in ss:
            ss[company] = get_company_info(company)

with st.sidebar:
        page = option_menu(
            menu_title=None,
            options=["Company Lookup", "Trust & Data", "About Us"],
            icons=["house", "clipboard-data", "person-circle"],
        )

if page == "Company Lookup":
    with c1:
        get_styled_title("Company Lookup")
        get_share_elements()
    
    with c2:
        title_column_1, title_column_2 = st.columns([8, 1])
        with title_column_1:
            st.multiselect("", available_companies, key="companies", on_change=handle_company_select)
        with title_column_2:
            st.markdown('#')
            if "compare" in ss:
                default_val = ss.compare
            else:
                default_val = False
            tog.st_toggle_switch(label="Compare", 
                        key="compare", 
                        default_value=default_val, 
                        label_after = False, 
                        inactive_color = '#D3D3D3', 
                        active_color=primary_color, 
                        track_color=primary_color,
                        )
        st.markdown('#')

    with c3:
        params = st.experimental_get_query_params()
        param_companies = params["companies"] if "companies" in params else []
        if len(param_companies) > 0:
            # comparison mode
            if ss.compare:
                with elements("dashboard"):
                    if len(param_companies) > 0:
                        if "layout" not in ss:
                            ss.layout = []
                        
                        for i, company in enumerate(param_companies):
                            # check whether company is already in layout
                            exists = False
                            for l in ss.layout:
                                if l["i"] == company:
                                    exists = True
                                    break
                            
                            # if not, add it
                            if not exists:
                                # if it's an odd index, add it to the right
                                if i % 2 != 0:
                                    x = 6
                                else:
                                    x = 0
                                ss.layout.append(dashboard.Item(company, x, 0, 5, 4, allowOverlap=True))
                            
                        with dashboard.Grid(ss.layout):
                            for company in param_companies:  
                                company_info = ss[company]                      
                                with mui.Card(key=company, sx={"display": "flex", "flexDirection": "column"}, raised=True):
                                    mui.CardHeader(title=company, subheader=f'Disclosure Score: {company_info["score"]}', sx={"color": "white", "background-color": primary_color, "padding": "5px 15px 5px 15px", "borderBottom": 2, "borderColor": "divider"})
                                    with mui.CardContent(sx={"flex": 1, "minHeight": 0, "background-color": secondary_background_color}):
                                        # with mui.List():
                                        #     for qa_pair in company_info["qa_pairs"]:
                                        #         with mui.ListItem(sx={"background-image": get_confidence_style(qa_pair, secondary_background_color)}):
                                        #             mui.ListItemText(primary= f'Q: {qa_pair["question"]}', secondary= f'A: {qa_pair["answer"]}', sx={"padding": "0px 0px 0px 0px"})
                                        
                                        for category, qa_pairs in company_info["qa_pairs"].items():
                                            expanded = category == "General"
                                            with mui.Accordion(defaultExpanded=expanded):
                                                with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                                                    mui.Typography(category)
                                                with mui.AccordionDetails():
                                                    with mui.List():
                                                        for qa_pair in qa_pairs:
                                                            with mui.ListItem(alignItems="flex-start", sx={"padding": "0px 0px 0px 0px"}):
                                                                mui.ListItemText(primary= f'Q: {qa_pair["question"]}', sx={"padding": "0px 0px 0px 0px"})
                                                            if len(qa_pair["answers"]) == 0:
                                                                mui.ListItemText(secondary= f'A: No answer found', sx={"padding": "0px 0px 0px 0px", "background-image": get_no_confidence("white")})
                                                            for answer in qa_pair["answers"]:
                                                                mui.ListItemText(secondary= f'A: {answer["answer"]}', sx={"padding": "0px 0px 0px 0px", "background-image": get_confidence_style(answer, "white")})
                                    # with mui.CardActions(sx={"color": "white", "padding": "5px 15px 5px 15px", "background-color": "#ff4b4b", "borderTop": 2, "borderColor": "divider"}):
                                    #     mui.Button("Learn More", size="small", sx={"color": "white"})

            # tabular mode
            else:
                if "prev_company" in ss and ss.prev_company in param_companies:
                    df = ss.prev_company
                else:
                    df = param_companies[0]

                tabs = st.tabs(param_companies)
                for i, tab in enumerate(tabs):
                    with tab:
                        curr_company = param_companies[i]
                        company_info = ss[curr_company]

                        col1, col2, _col, _col, col3 = st.columns([1, 2, 1, 1, 1])

                        with col1:
                            get_styled_title(company_info["name"])
                        #col1.subheader(company_info["name"])
                        with col2:
                            get_investment_profile(curr_company)
                        col3.metric(label="Disclosure Score", value=company_info["score"])
                        for category, qa_pairs in company_info["qa_pairs"].items():
                            expanded = category == "General"
                            with st.expander(category, expanded=expanded):
                                for qa_pair in qa_pairs:
                                    st.write(f'**Q:** {qa_pair["question"]}')
                                    if len(qa_pair["answers"]) == 0:
                                        answer_html = """
                                        <div style="background-image: {}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                            <div>{}</div>
                                        </div>
                                        """.format(get_no_confidence("white"), "No answer found")
                                        st.markdown(answer_html, unsafe_allow_html=True)
                                    for answer in qa_pair["answers"]:
                                        answer_html = """
                                        <div style="background-image: {}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                            <div>{}</div>
                                        </div>
                                        """.format(get_confidence_style(answer, "white"), answer["answer"])
                                        st.markdown(answer_html, unsafe_allow_html=True)

                                    # qa_html = """
                                    # <div style="margin:10px;background-image:{}">
                                    #     <div style="font-weight: bold">Q: {}</div>
                                    #     <div>A: {}</div>
                                    # </div>
                                    # """.format(get_confidence_style(qa_pair, background_color), qa_pair["question"], qa_pair["answer"])
                                    # st.markdown(qa_html, unsafe_allow_html=True)
                        
                        

                # ss.curr_company = stx.tab_bar(
                #     data=(stx.TabBarItemData(id=company, title=company, description="") for company in param_companies),
                #     default=df,
                # )

                # if ss.curr_company in ss:
                #     col1, _col, col3 = st.columns([1, 3, 1])
                #     company_info = ss[ss.curr_company]

                #     col1.subheader(company_info["name"])
                #     col3.metric(label="Disclosure Score", value=company_info["score"])
                #     for qa_pair in company_info["qa_pairs"]:
                #         qa_html = """
                #         <div style="margin:10px;background-image:{}">
                #             <div style="font-weight: bold">Q: {}</div>
                #             <div>A: {}</div>
                #         </div>
                #         """.format(get_confidence_style(qa_pair, background_color), qa_pair["question"], qa_pair["answer"])
                #         st.markdown(qa_html, unsafe_allow_html=True)
                #     st.markdown('#')
                    
                #     ss.prev_company = ss.curr_company

                #     get_investment_profile(ss.curr_company)
                # else:
                #     st.write("N/A")

    with c4:
        for i in range(6):
            st.markdown("#")
        
default_trust = """

We want to ensure that you view this tool as a trustworthy source of reliable information on firm-level, climate-relevant information. In particular, it will guide you through the maze of corporate climate data to aid your decision-making. We base the information that is presented to you on credible data disclosed by the firm itself.

#### Method
At a high-level, the information is compiled as follows:
1. The tool searches through one of three firm disclosures to identify passages that match your search query.
2. Subsequently, you will be presented with either the passages that best match your search query or a high-quality summary of that information.

#### Sources
We ensure credibility by relying on the following three data sources:
1. A firm's 10-K filing, filed with the SEC EDGAR database.
2. The most recent earnings conference call transcript which features managements' discussion of quarterly results, alongside a Q&A session with sell-side analysts.
3. A firm's voluntarily disclosed sustainability report.

#### Why is this important?
Importantly, trawling through the various data sources individually is a tremendous challenge for investors. The different sources are not only heterogeneous in layout and structure, but also 100+ pages long, transcripts of hour long conversations, and hard to navigate and parse as a regular investor seeking aid in investment decision-making. We aim to alleviate this issue by providing answers to a curated set of climate-relevant questions.
"""

# Map of company name to references section
company_references = {
    "Ford": [
        "Ford Motor Company 10-K filing (0000037996-22-000013): https://www.sec.gov/ix?doc=/Archives/edgar/data/37996/000003799622000013/f-20211231.htm",
        "Ford Motor Company Earnings Conference Call (2022-02-04): https://seekingalpha.com/article/4484425-ford-motor-company-2021-q4-results-earnings-call-presentation",
        "Ford Motor Company Integrated Sustainabiltiy Report (2022): https://corporate.ford.com/content/dam/corporate/us/en-us/documents/reports/integrated-sustainability-and-financial-report-2022.pdf",
    ],
    "General Mills": [
        "General Mills, Inc. 10-K filing (0001193125-21-204830): https://www.sec.gov/ix?doc=/Archives/edgar/data/40704/000119312521204830/d184854d10k.htm",
        "General Mills, Inc. Earnings Conference Call (2022-03-23): https://seekingalpha.com/article/4497316-general-mills-inc-gis-ceo-jeff-harmening-on-q3-2022-results-earnings-call-transcript",
        "General Mills, Inc. Sustainabiltiy Report (2022): https://globalresponsibility.generalmills.com/images/General_Mills-Global_Responsibility_2022.pdf",
    ],
    "Fisker": [
        "Fisker Inc. 10-K filing (0001720990-22-000010): https://www.sec.gov/ix?doc=/Archives/edgar/data/1720990/000172099022000010/fsr-20211231.htm",
        "Fisker Inc. Earnings Conference Call (2022-02-16): https://seekingalpha.com/article/4487648-fisker-inc-fsr-ceo-henrik-fisker-on-q4-2021-results-earnings-call-transcript",
        "Fisker Inc. Company ESG Impact Report (2021): https://assets.ctfassets.net/cghen8gr8e1n/2sBPf0jjfZa20R8Ycwar4Q/ff96bb41c1348978af542610f3f7a88e/2021_Fisker_ESG_Report.pdf",
    ],
    "Pepsico": [
        "PepsiCo, Inc. 10-K filing (0000077476-22-000010): https://www.sec.gov/ix?doc=/Archives/edgar/data/77476/000007747622000010/pep-20211225.htm",
        "PepsiCo, Inc. Earnings Conference Call (2022-02-10): https://seekingalpha.com/article/4485846-pepsico-inc-pep-ceo-ramon-laguarta-on-q4-2021-results-earnings-call-transcript",
        "PepsiCo, Inc. SASB Index (2021): https://www.pepsico.com/docs/default-source/sustainability-and-esg-topics/2021-sasb-index.pdf",
    ],
}


if page == "Trust & Data":
    get_styled_title("Notes on Data Use and Trust")
    st.markdown(default_trust)
    params = st.experimental_get_query_params()
    param_companies = params["companies"] if "companies" in params else []
    if len(param_companies) > 0:
        # Create a separating line
        st.markdown('---')
        st.markdown("### References")
        st.markdown("The presented information can be looked up in the following documents:")
        for company in param_companies:
            if company in company_references:
                st.markdown("#### " + company)
                for ref in company_references[company]:
                    st.markdown(ref)


if page == "About Us":
    get_styled_title("About Us")
    # display all images in the "pictures" folder
    # display them in 3 columns
    col1, col2, col3 = st.columns(3)
    pics = os.listdir("pictures")
    # reverse the list so Akhil isn't first, lol
    pics.reverse()
    for i, img in enumerate(pics):
        # get image name without .jpeg
        person = img.split(".")[0].upper()
        if i % 3 == 0:
            col1.image("pictures/" + img, caption=person)
        elif i % 3 == 1:
            col2.image("pictures/" + img, caption=person)
        else:
            col3.image("pictures/" + img, caption=person)


