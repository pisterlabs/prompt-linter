import xmlrpc.client
import openai
import ssl
import configparser



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
    
# Load config file
config = configparser.ConfigParser()
config.read('config.ini')

# Set OpenAI API key from config file
openai.api_key = config.get('OpenAI', 'api_key')

# Set Odoo details from config file
url = config.get('Odoo', 'url')
db = config.get('Odoo', 'db')
username = config.get('Odoo', 'username')
password = config.get('Odoo', 'password')

# Login to Odoo
common = xmlrpc.client.ServerProxy('{}/xmlrpc/2/common'.format(url))
uid = common.authenticate(db, username, password, {})
models = xmlrpc.client.ServerProxy('{}/xmlrpc/2/object'.format(url))

# Get the first 100 products
product_ids = models.execute_kw(db, uid, password,
    'product.template', 'search',
    [[]],
    {'limit': 100})

products = models.execute_kw(db, uid, password,
    'product.template', 'read', [product_ids])

# Update product descriptions
for product in products:
    print(f"Updating product {product['name']}...")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Describe the product: {product['name']}",
        temperature=0.5,
        max_tokens=100
    )
    description = response.choices[0].text.strip()
    
    # Update product description
    models.execute_kw(db, uid, password, 'product.template', 'write', [product['id'], {'description': description}])
    print(f"Updated product {product['name']} with description: {description}")