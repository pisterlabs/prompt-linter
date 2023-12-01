import boto3
from collections import defaultdict
import pandas as pd
import cohere
from cohere.classify import Example

def check_item(food_item):
    co = cohere.Client('rkFkHXfiEHRXu5Ni9uILfHckQMaloiCHH0ERVmN8') # This is your trial API key
    response = co.generate(
    model='command-xlarge-20221108',
    prompt=f'Can you check whether this is a food item or not: {food_item}',
    max_tokens=300,
    temperature=0.9,
    k=0,
    p=0.75,
    frequency_penalty=0,
    presence_penalty=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    return (response.generations[0].text)

def classify_food(fd):
    co = cohere.Client('rkFkHXfiEHRXu5Ni9uILfHckQMaloiCHH0ERVmN8') # This is your trial API key
    response = co.classify(
    model='large',
    inputs=["plane", "dhfvhevbs", "jajaja", "cake", "127y3", "cucumber"],
    examples=[Example("strawberry", "food"), Example("zucchini", "food"), Example("bread", "food"), Example("sunflower oil", "food"), Example("banana", "food"), Example("burger", "food"), Example("table", "not food"), Example("road", "not food"), Example("car", "not food"), Example("cake", "food"), Example("284", "not food")])

    return ([x.prediction for x in response.classifications])

def cl_food_items(food_items):
    

    co = cohere.Client('rkFkHXfiEHRXu5Ni9uILfHckQMaloiCHH0ERVmN8') # This is your trial API key
    response = co.classify(
    model='large',
    inputs=[food_items],
    examples=[Example("banana", "fruit/vegetable"), Example("zucchini", "fruit/vegetable"), Example("bread", "Bread/Grains"), Example("roti", "Bread/Grains"), Example("flatbread", "Bread/Grains"), Example("Baguette", "Bread/Grains"), Example("cake", "Bread/Grains"), Example("pasta", "Bread/Grains"), Example("brown rice", "Bread/Grains"), Example("rice cake", "Bread/Grains"), Example("raddish", "fruit/vegetable"), Example("spinach", "fruit/vegetable"), Example("tomato", "fruit/vegetable"), Example("potato", "fruit/vegetable"), Example("berries", "fruit/vegetable"), Example("car", "not food"), Example("mat", "not food"), Example("burger", "Bread/Grains"), Example("donut", "Bread/Grains"), Example("chocolate", "fruit/vegetable"), Example("red chilli", "fruit/vegetable"), Example("beef", "Meat"), Example("lamb", "Meat"), Example("red meat", "Meat"), Example("chicken curry", "Meat"), Example("alcohol", "Drinks"), Example("juice", "Drinks"), Example("milk", "Drinks"), Example("door", "not food"), Example("charcoal", "not food"), Example("door", "not food"), Example("vhhdbvh", "not food"), Example("jsfbanc", "not food"), Example("jja", "not food"), Example("l", "not food")])
    return ([x.prediction for x in response.classifications])


"""
 The following was used as a source for code:
 https://docs.aws.amazon.com/textract/latest/dg/examples-export-table-csv.html
 https://docs.aws.amazon.com/textract/latest/dg/examples-extract-kvp.html
 
 This is short demo on how to extract the TABLES and KEY-VALUE PAIRS using textract analyze_expense. 
 
 NOTE:
 * This code uses textract's analyze_document.
 * Textract analyze_document and textract analyze_expense are DIFFERENT.
 * Textract analyze_expsense calls the textract model trained on INVOICES ONLY and will provide different results than textract analyze_document. 
 * Textract analyze_document calls the textract model trained on ANY TYPE OF DOCUMENT.


"""
'''
def load_pdf_page_as_byte(file_name, page_number):
    # load pdf as images
    pages = pdf2image.convert_from_path(file_name, first_page  = page_number, last_page  = page_number, dpi = 300, poppler_path="./poppler-21.03.0/Library/bin")
    
    # convert to byte array
    img_byte_arr = io.BytesIO()
    pages[0].save(img_byte_arr, format='PNG')
    bytes_test = img_byte_arr.getvalue()    
    print(f'PDF {file_name}, page {page_number} loaded as an image')
    return bytes_test
'''
def load_image_as_byte(file_name):
    # load images
    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
        print('Image loaded', file_name)
    return bytes_test

def get_response(file_name):
    
    # If your file_name is an image, then use load_image_as_byte. If it is a pdf, then you can use use load_pdf_page_as_bytes, specifying a page_number.    
    bytes_test =  load_image_as_byte(file_name)
    # bytes_test =  load_pdf_page_as_byte(file_name, page_number = 1)
    # process using image bytes
    client = boto3.client('textract', aws_access_key_id="AKIARTWPZA256ABBPBUR", aws_secret_access_key="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", region_name="ca-central-1")
    response = client.analyze_expense(Document={'Bytes': bytes_test})
    return response
    
def get_kvs(response):
    summary_fields = response.get('ExpenseDocuments')[0]['SummaryFields']
    kvs = defaultdict(list)
    # keyvalue pairs
    for field in summary_fields:
        key = get_key(field)
        val = get_value(field)
        kvs[key].append(val)
    return kvs
    
def get_key(pair):
    if pair.get("LabelDetection"):
        return pair.get("LabelDetection").get("Text")
    else:
        return pair.get("Type").get("Text")
    
def get_value(pair):
    return pair.get("ValueDetection").get("Text")

def get_line_item_text(line_item_data):    
    return line_item_data.get("ValueDetection").get('Text')

def get_line_item_column_name(line_item_data):    
    if  line_item_data.get("LabelDetection"):
        return line_item_data.get("LabelDetection").get("Text")
    else:
        return line_item_data.get("Type").get("Text")
    
def get_expense_table(response):
    df = pd.DataFrame()
    for line_item in response.get("ExpenseDocuments")[0]['LineItemGroups'][0].get("LineItems"):
        d_line_item = {}
        for line_item_data in line_item.get("LineItemExpenseFields"):
            text = get_line_item_text(line_item_data)
            column_name = get_line_item_column_name(line_item_data)
            d_line_item[column_name] = text
        df = df.append(d_line_item, ignore_index = True)
    return df

def print_kvs(kvs):
    for key, value in kvs.items():
        print(key, ":", value)

def main():
    response = get_response("./ml_models/image.png")

    # Get Key Value relationship
    # kvs = get_kvs(response)
    # print("\n\n== FOUND KEY : VALUE pairs ===\n")
    # print_kvs(kvs)
    
    df = get_expense_table(response)
    #print(df)
    
    js_obj = []
    for i in range(df.shape[0]):
        food_items = df.loc[i]['ITEM'].lower()
        food_category = cl_food_items(food_items)[0]

        if food_category != 'not food':
            js_obj.append({'item': food_items, 'category': food_category, 'amount' : df.loc[i]['QUANTITY']})
    
    # print(js_obj)
    return js_obj


    
    
    