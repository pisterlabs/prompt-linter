import openai


class GPTQuery:
    def __init__(self,key):
        openai.api_key = key
        self.key = key
    ###
    def giveMePromt(self,query_user: str):
        x = f"""### Postgres SQL tables, with their properties:
        #
        #marketdata(transfer_type, days_for_shipping_real,days_for_shipment_scheduled, benefit_per_order, sales_per_customer, delivery_status, late_delivery_risk,
        category_Id, category_Name, customer_City, customer_Country,
        customer_Email, customer_Fname, customer_Id, customer_Lname,
        customer_Password, customer_Segment, customer_State,
        customer_Street, customer_Zipcode, department_Id,
        department_Name, latitude, longitude, market, order_City,
        order_Country, order_Customer_Id, order_date, order_Id,
        order_Item_Cardprod_Id, order_Item_Discount,
        order_Item_Discount_Rate, order_Item_Id, order_Item_Product_Price,
        order_Item_Profit_Ratio, order_Item_Quantity, sales,
        order_Item_Total, order_Profit_Per_order, order_Region,
        order_State, order_Status, product_Card_Id, product_Category_Id,
        product_Image, product_Name, product_price, product_status,
        shipping_date, shipping_mode)
    #
    ###A query to list of {str(query_user)} SELECT""",
        return x
    ###
    def make_sql_statement(self,query_user:str):
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=self.giveMePromt(query_user),
            temperature=0,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["#", ";"]
        )
        text = response['choices'][0]['text']
        statement = "SELECT " + str(text) + ';'
        return statement


class DataAndDataTypes:
    def give_structure_data(self):
        return {
    "Transfer Type" :"varchar",
	"Days For Shipping Real" :"integer",
    "Days For Shipment Scheduled" :"integer",
    "Benefit Per Order" :"numeric",
    "Sales Per Customer" :"numeric",
    "Delivery Status" :"varchar",
    "Late Delivery Risk" :"integer",
    "Category Id" :"integer",
    "Category Name" :"varchar",
    "Customer City" :"varchar",
    "Customer Country" :"varchar",
    "Customer Email" :"varchar",
    "Customer Fname" :"varchar",
    "Customer Id" :"integer",
    "Customer Lname" :"varchar",
    "Customer Password" :"varchar",
    "Customer Segment" :"varchar",
    "Customer State" :"varchar",
    "Customer Street" :"varchar",
    "Customer Zipcode" :"integer",
    "Department Id" :"integer",
    "Department Name" :"varchar",
    "Latitude" :"numeric",
    "Longitude" :"numeric",
    "Market" :"varchar",
    "Order City" :"varchar",
    "Order Country" :"varchar",
    "Order Customer Id" :"varchar",
    "Order Date" :"timestamp",
    "Order Id" :"integer",
    "Order Item Cardprod Id" :"integer",
    "Order Item Discount" :"numeric",
    "Order Item Discount Rate" :"numeric",
    "Order Item Id" :"integer",
    "Order Item Product Price" :"numeric",
    "Order Item Profit Ratio" :"numeric",  
    "Order Item Quantity" :"integer",
    "Sales" :"numeric",
    "Order Item Total" :"numeric",
    "Order Profit Per order" :"numeric",
    "Order Region" :"varchar",
    "Order State" :"varchar",
    "Order Status" :"varchar",
    "Product Card Id" :"integer",
    "Product Category Id" :"integer",
    "Product Image" :"varchar",
    "Product Name" :"varchar",
    "Product Price" :"numeric",
    "Product Status" :"integer",
    "Shipping Date" :"timestamp",
    "Shipping Mode" :"varchar"
}