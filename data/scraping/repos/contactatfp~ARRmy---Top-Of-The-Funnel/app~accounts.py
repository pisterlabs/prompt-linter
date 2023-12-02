import logging
import os
from datetime import datetime, timedelta

import faker
import random
from flask import Blueprint, request, jsonify, abort, render_template
from flask_login import login_required
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from app.models import Account, Contact, Interaction, db
from app.opps import get_closed_won_opps, get_open_opps
from app.tokens import tokens
from app.utils import cache, create_api_request

openai_api_key = os.environ.get('OPENAI_API_KEY')
google_cse_id = os.environ.get('GOOGLE_CSE_ID')
google_api_key = os.environ.get('GOOGLE_API_KEY')


accounts_blueprint = Blueprint('accounts', __name__)

DOMAIN = "https://fakepicasso-dev-ed.develop.my.salesforce.com"
SALESFORCE_API_ENDPOINT = "/services/data/v58.0/sobjects/"
SALESFORCE_API_OPPS = "/services/data/v58.0/graphql"


@accounts_blueprint.route('/get_account_name', methods=['GET'])
def get_account_name():
    account_id = request.args.get('account_id')
    if not account_id:
        return jsonify({"error": "Missing account_id parameter"}), 400

    account = Account.query.filter_by(Id=account_id).first()
    if account:
        return jsonify({"AccountName": account.Name})
    else:
        return jsonify({"error": "Account not found"}), 404


@accounts_blueprint.route('/tier/<account_id>', methods=['GET'])
def get_tier(account_id):
    account = Account.query.get(account_id)

    # get all opportunities for account that closed won in last 12 months
    closed_won_opps = get_closed_won_opps()

    # Filter closed_won_opps
    closed_won_opps = [opp for opp in closed_won_opps if
                       opp and opp.get("node") and opp["node"].get("CloseDate") and datetime.strptime(
                           opp["node"]["CloseDate"].get("value", "1900-01-01"),
                           '%Y-%m-%d') > datetime.now() - timedelta(days=365)]
    closed_won_opps = [opp for opp in closed_won_opps if
                       opp.get("node") and opp["node"].get("Account") and opp["node"]["Account"].get(
                           "Id") == account_id]

    # get all opportunities for account that are open
    open_opps = get_open_opps()

    # Filter open_opps
    open_opps = [opp for opp in open_opps if
                 opp and opp.get("node") and opp["node"].get("Account") and opp["node"]["Account"].get(
                     "Id") == account_id]
    open_opps = [opp for opp in open_opps if opp.get("node") and opp["node"].get("CloseDate") and datetime.strptime(
        opp["node"]["CloseDate"].get("value", "1900-01-01"), '%Y-%m-%d') < datetime.now() + timedelta(days=180)]

    # Format the data
    formatted_data = f"<strong>Rank:</strong> {account.Rank if account else 'N/A'}<br>"
    formatted_data += f"<strong>Closed Won Opportunities:</strong> {len(closed_won_opps)}<br>"
    formatted_data += f"<strong>Open Opportunities:</strong> {len(open_opps)}<br>"
    total_won = 0
    for opp in closed_won_opps:
        close_date = opp.get("node", {}).get("CloseDate", {}).get("value", "")
        formatted_data += f"<span class='closed-won-opp'><strong>Closed Won Opp:</strong> Closed on {close_date}</span><br>"
        total_won += opp.get("node", {}).get("Amount", {}).get("value", 0)

    if total_won > 0:
        total_won = int(total_won)
        formatted_data += f"<span class='closed-won-opp'><br><strong>Total Closed in Last 12: $</strong>{total_won}</span><br>"

    for opp in open_opps:
        close_date = opp.get("node", {}).get("CloseDate", {}).get("value", "")
        formatted_data += f"<span class='open-opp'><strong>Open Opp:</strong> Should close on {close_date}</span><br>"

    return formatted_data


@accounts_blueprint.route('/update-tier/<account_id>', methods=['POST'])
def update_tier(account_id):
    try:
        account = Account.query.get(account_id)
        new_tier = request.form.get('tier')
        account.Rank = new_tier
        db.session.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e))


@accounts_blueprint.route('/account/<string:account_id>', methods=['GET'])
def account_details(account_id):
    account = Account.query.get(account_id)
    if account is None:
        abort(404, description="Account not found")
    cache_key = f"prospecting_data_{account_id}"
    prospecting_data = cache.get(cache_key)

    if prospecting_data:
        # Pass the cached data to the template
        return render_template('account_tooltip.html', account=account, prospecting_data=prospecting_data)
    else:
        return render_template('account_tooltip.html', account=account)


@accounts_blueprint.route('/account/<id>')
@login_required
def account(id):
    account = Account.query.filter_by(id=id).first()
    contacts = Contact.query.filter_by(account_id=id).all()
    interactions = Interaction.query.filter_by(account_id=id).all()

    return render_template('account.html', account=account, contacts=contacts, interactions=interactions)


@accounts_blueprint.route('/add_account', methods=['POST'])
@login_required
def add_account():
    data = request.json
    url = f"{DOMAIN}{SALESFORCE_API_ENDPOINT}account"

    token = tokens()
    if not token:
        return {"error": "Failed to get Salesforce token"}

    # Map input data to Salesforce fields
    sf_data = {
        "Name": data.get('name'),
        "BillingStreet": data.get('street'),
        "BillingCity": data.get('city'),
        "BillingState": data.get('state'),
        "BillingPostalCode": data.get('zip')
    }

    response = create_api_request("POST", url, token['access_token'], sf_data)
    if response.status_code == 201:
        # get_data()
        return jsonify({"success": True})
    else:
        logging.error(f"Error adding account: {response.text}")
        return jsonify({"success": False, "error": response.text})


@accounts_blueprint.route('/salesforce/account-status/<account_id>', methods=['GET'])
def get_account_status(account_id):
    url = f"{DOMAIN}{SALESFORCE_API_OPPS}"
    token = tokens()
    if not token:
        return {"error": "Failed to get Salesforce token"}

    # Check for open opportunities
    open_query = f"""
    query openOpportunitiesForAccount {{
      uiapi {{
        query {{
          Opportunity(
            where: {{
              and: [
                {{ Account: {{ Id: {{ eq: "{account_id}" }} }} }},
                {{ StageName: {{ eq: "Open" }} }}
              ]
            }}
          ) {{
            edges {{
              node {{
                Id
              }}
            }}
          }}
        }}
      }}
    }}
    """
    open_payload = {"query": open_query, "variables": {}}
    response_open = create_api_request("POST", url, token['access_token'], open_payload)
    open_count = len(response_open.json()["data"]["uiapi"]["query"]["Opportunity"]["edges"])

    # Check for closed opportunities within the last 12 months
    twelve_months_ago = (datetime.now() - timedelta(days=365)).isoformat()
    closed_query = f"""
    query recentClosedOpportunitiesForAccount {{
      uiapi {{
        query {{
          Opportunity(
            where: {{
              and: [
                {{ Account: {{ Id: {{ eq: "{account_id}" }} }} }},
                {{ CloseDate: {{ gte: "{twelve_months_ago}" }} }},
                {{ or: [
                  {{ StageName: {{ eq: "Closed Won" }} }},
                  {{ StageName: {{ eq: "Closed Lost" }} }}
                ]}}
              ]
            }}
          ) {{
            edges {{
              node {{
                Id
              }}
            }}
          }}
        }}
      }}
    }}
    """
    closed_payload = {"query": closed_query, "variables": {}}
    response_closed = create_api_request("POST", url, token['access_token'], closed_payload)
    test = response_closed.json()
    # closed_count = len(response_closed.json()["data"]["uiapi"]["query"]["Opportunity"]["edges"])

    # closed_count = len(response_closed.json()["data"]["uiapi"]["query"]["Opportunity"]["edges"])

    # Determine color based on counts
    if open_count > 0:
        status_color = "Green"
    # elif closed_count > 0:
    #     status_color = "Yellow"
    else:
        status_color = "Red"

    return status_color


@accounts_blueprint.route('/address', methods=['GET', 'POST'])
def add_account_billing_address():
    fake = faker.Faker('en_US')
    top_75_cities = [
        'New York City', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
        'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte',
        'San Francisco', 'Indianapolis', 'Seattle', 'Denver', 'Washington',
        'Boston', 'El Paso', 'Nashville', 'Detroit', 'Oklahoma City',
        'Portland', 'Las Vegas', 'Memphis', 'Louisville', 'Baltimore',
        'Milwaukee', 'Albuquerque', 'Tucson', 'Fresno', 'Mesa',
        'Sacramento', 'Atlanta', 'Kansas City', 'Colorado Springs', 'Omaha',
        'Raleigh', 'Miami', 'Long Beach', 'Virginia Beach', 'Oakland',
        'Minneapolis', 'Tulsa', 'Arlington', 'Tampa', 'New Orleans',
        'Wichita', 'Cleveland', 'Bakersfield', 'Aurora', 'Anaheim',
        'Honolulu', 'Santa Ana', 'Riverside', 'Corpus Christi', 'Lexington',
        'Stockton', 'Henderson', 'Saint Paul', 'St. Louis', 'Cincinnati',
        'Pittsburgh', 'Greensboro', 'Anchorage', 'Plano', 'Lincoln',
        'Orlando', 'Irvine', 'Newark', 'Toledo', 'Durham'
    ]

    # with app.app_context():
    # Fetch all contacts
    accounts = Account.query.all()
    for account in accounts:
        account.BillingStreet = fake.street_address()
        account.BillingCity = top_75_cities[random.randint(0, len(top_75_cities) - 1)]
        account.BillingState = fake.state()
        account.BillingPostalCode = fake.zipcode()
        account.BillingCountry = "USA"
    # Commit the changes to the database
    db.session.commit()

    return jsonify({"success": True})


def add_account_notes():
    fake = faker.Faker()
    # with app.app_context():
    # Fetch all contacts
    accounts = Account.query.all()
    for account in accounts:
        account.Notes = fake.paragraph(nb_sentences=5)
        # Commit the changes to the database
    db.session.commit()

    return jsonify({"success": True})


def add_account_industry():
    fake = faker.Faker()
    # with app.app_context():
    # Fetch all contacts
    accounts = Account.query.all()
    industry_list = [
        'Software Development',
        'Information Technology Services',
        'Semiconductor Manufacturing',
        'E-commerce',
        'Cybersecurity',
        'Cloud Computing',
        'Artificial Intelligence and Machine Learning',
        'Telecommunications',
        'Data Analytics',
        'Internet of Things (IoT)',
        'Healthcare',
        'Finance',
        'Manufacturing',
        'Construction',
        'Energy (Oil & Gas)',
        'Renewable Energy',
        'Automotive',
        'Retail',
        'Education',
        'Agriculture',
        'Real Estate',
        'Transportation',
        'Food and Beverage',
        'Tourism and Hospitality',
        'Media and Entertainment',
        'Pharmaceuticals',
        'Logistics',
        'Consulting',
        'Legal Services',
        'Environmental Services'
    ]

    for account in accounts:
        account.Industry = industry_list[random.randint(0, len(industry_list) - 1)]
        account.NumberOfEmployees = random.randint(1, 10000)
    # Commit the changes to the database
    db.session.commit()

    return jsonify({"success": True})


def notes_summary(account_id):
    db = SQLDatabase.from_uri("sqlite:///instance/sfdc.db")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", verbose=True, openai_api_key=openai_api_key)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    # account = query account with ID 001Dp00000KBTVRIA5
    account = Account.query.get(account_id)
    summary = db_chain.run(f'''
    Create an account summary that includes Interactions for the account: {account.Name}
    If there are not Interactions for the account then return "No Interactions for this account"
    ''')
    return summary


@accounts_blueprint.route('/save_notes', methods=['POST'])
def save_notes():
    account_id = request.form.get('account_id')
    note_text = request.form.get('note_text')

    # Fetch the account using the account_id
    account = Account.query.get(account_id)
    if not account:
        return jsonify(status="error", message="Account not found"), 404

    # Update the Notes field
    account.Notes = note_text
    db.session.commit()

    return jsonify(status="success", message="Notes updated successfully")


@accounts_blueprint.route('/news', methods=['GET', 'POST'])
@login_required
def news():
    # get company name from request
    company_name = request.args.get('company', default='', type=str)

    import os

    os.environ["GOOGLE_CSE_ID"] = google_cse_id
    os.environ["GOOGLE_API_KEY"] = google_api_key

    search = GoogleSearchAPIWrapper()

    def top5_results(query):
        return search.results(query, 5)

    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=top5_results,
    )
    answer = tool.run("Company News for: " + company_name)

    formatted_response = ""
    for item in answer:
        formatted_response += f'<p><a href="{item["link"]}">{item["title"]}</a><br>{item["snippet"]}</p>'

    return formatted_response



