import random
from faker import Faker
from langchain import SQLDatabase

fake = Faker()

# Define the number of records to generate
num_records = 100

# Define the range of user_id and product_id
user_id_range = (1, 100)
product_id_range = (1, 3)

# Define the range of transaction_amount
transaction_amount_range = (1, 1000)

# Define the list of payment methods
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Cash']

db_user = "root"
db_password = "password"
db_host = "20.119.36.225:8036"
db_name = "payment"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Generate the data
for _ in range(num_records):
    user_id = random.randint(*user_id_range)
    product_id = random.randint(*product_id_range)
    transaction_date = fake.date_between(start_date='-1m', end_date='today')
    transaction_amount = round(random.uniform(*transaction_amount_range), 2)
    payment_method = random.choice(payment_methods)


    command = f"INSERT INTO transactions (user_id, product_id, transaction_date, transaction_amount, payment_method) VALUES ({user_id}, {product_id}, '{transaction_date}', {transaction_amount}, '{payment_method}');"
    db.run(command)