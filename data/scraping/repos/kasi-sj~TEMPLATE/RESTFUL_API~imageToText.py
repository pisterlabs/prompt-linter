import requests
import openai
import io

# ocr_api_key = "sMyTAXJrkUA1oMb7V7Eo/Q==1JZOORPP5NNEzzd3"
ocr_api_key = "M41KHFeJZeJvsu/oIi7Hgw==9908OP45hHzgWMr7"
openai.api_key = "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"  

def perform_ocr(binary_data):
    api_url = 'https://api.api-ninjas.com/v1/imagetotext'

    files = {'image': binary_data}

    headers = {
        "X-Api-Key": ocr_api_key,  # replace with your API key
    }
    r = requests.post(api_url,headers=headers, files=files)
    print(r.json())

    ocr_result = r.json()
    return ocr_result





def generate_gpt_response(input_text):
    try:
        res=[]
        for item in input_text:
            l=[]
            for k in item:
                if k == 'text':
                    l.append(item[k])
                else :
                    l.append(item['bounding_box']['x1'])
                    l.append(item['bounding_box']['y1'])
                    l.append(item['bounding_box']['x2'])
                    l.append(item['bounding_box']['y2'])
            res.append(l)
        data = f"""
                extracted data from the image
                [text , x1 , y1 , x2 , y2] title where x1,y1,x2,y2 are the cordinates of the bounding box of the text
                {res}
                => format the data in json formet given below
                => if this field is empty produce 'null' 
                "hospital_name : value",
                "hospital_contact :[phone:value , address:value , mail:value]" ,
                "patent_name : value",
                "patent_contact : [phone:value , address:value , mail:value]",
                "invoice no : value",
                "product details : values[product_no : value , product_name1 : value , product_quantity1 : value ,  product_price1 : value product_tax1 : value]" , 
                "total_amount : value" ,
                "total_tax : value"
            """
        response = openai.Completion.create(
            engine="text-davinci-003",  
            prompt=data,
            max_tokens=1000,  
            temperature=0.1
        )
        print('doing here')
    except Exception as e :
        print(e)
    return response["choices"][0]["text"]


def convert(data):
    ocr_text = perform_ocr(data)
    print()
    if ocr_text:
        print("OCR Result:")
        print("here")
        print(ocr_text)
        print("hai")
        gpt_response = generate_gpt_response(ocr_text)

        if gpt_response:
            print("\nGPT Response:")
            print(gpt_response)
            return gpt_response
    return ''

def reduce(input_text):
    res=""
    for item in input_text:
        l=[]
        for k in item:
            if k == 'text':
                l.append(item[k])
        res+=str(l)
    return res