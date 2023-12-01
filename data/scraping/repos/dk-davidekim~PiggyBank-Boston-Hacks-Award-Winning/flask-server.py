import os
import json
import threading

from flask import Flask, request, jsonify
import sqlalchemy
from google.cloud import pubsub_v1
from google.cloud.sql.connector import Connector
from dotenv import load_dotenv

from openai import OpenAI

from flask_cors import CORS

from queue import Queue
  
load_dotenv()

app = Flask(__name__)
CORS(app)


DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

PROJECT_ID = os.getenv('PROJECT_ID')
TOPIC_NAME = os.getenv('TOPIC_NAME')
SUBSCRIPTION_NAME = os.getenv('SUBSCRIPTION_NAME')


class PubSubManager:
    def __init__(self, project_id, topic_name, subscription_name):
        self.project_id = project_id
        self.topic_name = topic_name
        self.subscription_name = subscription_name
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

    def publish(self, message):
        topic_path = self.publisher.topic_path(self.project_id, self.topic_name)
        message_json = json.dumps({'message': message})
        message_bytes = message_json.encode('utf-8')
        publish_future = self.publisher.publish(topic_path, data=message_bytes)
        publish_future.result()

    def subscribe(self, callback):
        subscription_path = self.subscriber.subscription_path(self.project_id, self.subscription_name)
        streaming_pull_future = self.subscriber.subscribe(subscription_path, callback=callback)
        streaming_pull_future.result()

class DatabaseManager:
    def __init__(self, connection_string, db_user, db_password, db_name):
        self.connection_string = connection_string
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.connector = Connector()

    def get_engine(self):
        def getconn():
            return self.connector.connect(
                self.connection_string,
                "pymysql",
                user=self.db_user,
                password=self.db_password,
                db=self.db_name
            )
        return sqlalchemy.create_engine("mysql+pymysql://", creator=getconn)

messages = Queue()

def callback(message):
    messages.put(message.data.decode('utf-8'))
    message.ack()

pubsub_manager = PubSubManager(PROJECT_ID, TOPIC_NAME, SUBSCRIPTION_NAME)
db_manager = DatabaseManager(DB_CONNECTION_STRING, DB_USER, DB_PASSWORD, DB_NAME)
pool = db_manager.get_engine()

@app.route('/api/complete-chore', methods=['POST'])
def complete_chore():
    data = request.json
    chore_id = data.get('choreId')
    if not chore_id:
        return jsonify({'error': 'Missing choreId'}), 400
    
    with pool.connect() as conn:
        conn.execute(
            sqlalchemy.text('UPDATE Chore SET status = TRUE WHERE id = :chore_id'), 
            {'chore_id': chore_id}
        )
        conn.commit()  

        chore_name = conn.execute(
            sqlalchemy.text('SELECT name FROM Chore WHERE id = :chore_id'), 
            {'chore_id': chore_id}
        ).scalar()
        pubsub_manager.publish(chore_name)  

    return jsonify({'message': 'Chore marked as completed'})

@app.route('/api/start-parent-session', methods=['GET']) 
def start_parent_session():
    def run_subscription():
        pubsub_manager.subscribe(callback)

    thread = threading.Thread(target=run_subscription)
    thread.daemon = True
    thread.start()
    return jsonify({'message': 'Parent session started and listening for messages'})

@app.route('/api/get-messages', methods=['GET'])
def get_messages():
    messages_list = []
    while not messages.empty():
        message = messages.get()
        messages_list.append(message)
    return jsonify(messages_list)

def chat_with_gpt(item, description=None, model="gpt-3.5-turbo"):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant, knowledgeable about shopping, pricing, and providing specific recommendations."},
            {"role": "user", "content": f"I am a child planning to buy a {item}."}
        ]

        if description:
            messages.append({"role": "user", "content": f"It should be like this: {description}"})
            messages.append({"role": "system", "content": f"Based on the description, provide a specific price recommendation for the {item}."})

        response = client.chat.completions.create(model=model, messages=messages)

        if response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            else:
                return "Sorry, I couldn't process that response."
        else:
            return "No response received from the API."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Sorry, there was an error."

@app.route('/api/gpt-chat', methods=['POST'])
def gpt_chat():
    data = request.json
    item = data.get('item')
    description = data.get('description', None)
    response = chat_with_gpt(item, description)
    return jsonify({'response': response})


@app.route('/api/insert-item', methods=['POST'])
def insert_item():
    data = request.json
    try:
        item = data.get('item')
        price = data.get('price')
        if item is not None and price is not None:
            with pool.connect() as conn:
                conn.execute(
                    sqlalchemy.text('INSERT INTO SavingsGoal (item, price) VALUES (:item, :price)'), 
                    {'item': item, 'price': price}
                )
                conn.commit()
            return jsonify({'message': 'Item added successfully'})
        else:
            return jsonify({'error': 'Missing item or price data'}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/api/insert-chore', methods=['POST'])
def insert_chore():
    data = request.json
    try:
        name = data.get('name')
        compensation = data.get('compensation')
        if name is not None and compensation is not None:
            with pool.connect() as conn:
                conn.execute(
                    sqlalchemy.text('INSERT INTO Chore (name, compensation, status) VALUES (:name, :compensation, 0)'), 
                    {'name': name, 'compensation': compensation}
                )
                conn.commit()
            return jsonify({'message': 'Chore added successfully'})
        else:
            return jsonify({'error': 'Missing name or compensation data'}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/api/insert-allowance', methods=['POST'])
def insert_allowance():
    data = request.json
    try:
        amount = data.get('amount')
        if amount is not None:
            with pool.connect() as conn:
                conn.execute(
                    sqlalchemy.text('INSERT INTO Allowance (amount) VALUES (:amount)'), 
                    {'amount': amount}
                )
                conn.commit()
            return jsonify({'message': 'Allowance added successfully'})
        else:
            return jsonify({'error': 'Missing amount data'}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/api/mark-chore-complete', methods=['POST'])
def mark_chore_complete():
    data = request.json
    chore_id = data.get('choreId')
    try:
        with pool.connect() as conn:
            conn.execute(
                sqlalchemy.text('UPDATE Chore SET status = TRUE WHERE id = :chore_id'), 
                {'chore_id': chore_id}
            )
            conn.commit() 

            chore_name = conn.execute(
                sqlalchemy.text('SELECT name FROM Chore WHERE id = :chore_id'), 
                {'chore_id': chore_id}
            ).scalar()
            pubsub_manager.publish(chore_name)

        return jsonify({'message': 'Chore marked as completed'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/get-item', methods=['GET'])
def get_item():
    try:
        with pool.connect() as conn:
            item = conn.execute(
                sqlalchemy.text('SELECT item FROM SavingsGoal')
            ).scalar()
            if item:
                return jsonify({'item': item})
            else:
                return jsonify({'item': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-price', methods=['GET'])
def get_price():
    try:
        with pool.connect() as conn:
            price = conn.execute(
                sqlalchemy.text('SELECT price FROM SavingsGoal')
            ).scalar()
            if price:
                return jsonify({'price': price})
            else:
                return jsonify({'price': None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-allowance', methods=['GET'])
def get_allowance():
    try:
        with pool.connect() as conn:
            allowance = conn.execute(
                sqlalchemy.text('SELECT amount FROM Allowance ORDER BY id DESC LIMIT 1')
            ).scalar() or 0
            return jsonify({'allowance': allowance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-chores', methods=['GET'])
def get_chores():
    try:
        with pool.connect() as conn:
            ids = conn.execute(sqlalchemy.text('SELECT id FROM Chore')).fetchall()
            names = conn.execute(sqlalchemy.text('SELECT name FROM Chore')).fetchall()
            compensations = conn.execute(sqlalchemy.text('SELECT compensation FROM Chore')).fetchall()
            statuses = conn.execute(sqlalchemy.text('SELECT status FROM Chore')).fetchall()

            chores = []
            for i in range(len(ids)):
                chore = {
                    'id': ids[i][0],
                    'name': names[i][0],
                    'compensation': float(compensations[i][0]),
                    'isComplete': bool(statuses[i][0])
                }
                chores.append(chore)

            return jsonify(chores)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/get-total-balance', methods=['GET'])
def get_total_balance():
    try:
        with pool.connect() as conn:
            total_compensation = conn.execute(
                sqlalchemy.text('SELECT SUM(compensation) FROM Chore WHERE status = TRUE')
            ).scalar() or 0
            allowance = conn.execute(
                sqlalchemy.text('SELECT amount FROM Allowance ORDER BY id DESC LIMIT 1')
            ).scalar() or 0
            total_balance = total_compensation + allowance
            return jsonify({'total_balance': total_balance})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)