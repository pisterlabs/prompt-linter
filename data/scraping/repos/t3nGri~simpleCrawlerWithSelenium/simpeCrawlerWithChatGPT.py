# Import necessary libraries from Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Import required libraries for OpenAI API and other functionalities
import openai
import traceback
import base64
import time
import csv
import json
from PIL import Image

# Setting OpenAI API key
openai.api_key ="OpenAI key"

# Set up ChromeDriver and Chrome options
webdriver_service = Service('C:/Users/Administrator/Desktop/python/chromedriver.exe')
chrome_options = Options()

# Some Chrome settings
chrome_options.add_argument("--headless")
chrome_options.add_argument("--start-maximized")  
chrome_options.add_argument("--window-size=1920,1080") 
chrome_options.add_argument("--window-position=0,0")  
chrome_options.add_argument("--force-device-scale-factor=0.8") 

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Function to extract headings and text from a URL
def get_headings_and_text(url):

    try:
        driver.get(url)
    except TimeoutException as e:
        print("Error:", e)
        return None

    brand = "Gucci"

    # Extract Product Name
    product_name_element = driver.find_element(By.CLASS_NAME, 'showalbumheader__gallerytitle')
    product_name = product_name_element.text

    # Extract Product Description
    product_description_element = driver.find_element(By.CLASS_NAME, 'showalbumheader__gallerysubtitle')
    product_description = product_description_element.text

    # Communicate with GPT-3 to generate JSON containing product information
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a bot that only response json."},
        {"role": "user", "content": f"""Information about product:brand:{brand} product title:{product_name} description:{product_description}
         Find the following variations in this informations:
         size, product_code bag_type, main_material, and review. product_code must be numeral and plain. Exclude any trailing hyphen and numbers from the product_code, such as -3 or '*'. product_code must be in product title information. If there is no product_code as numeral, return as '1'.

         If there is no bag_type, return bag_type as 'Handbag'.
         If there is no main_material, return main_material as 'Lambskin'.
         If there is more than one size, separate them with commas. If there is no size information return 'Medium' value as size.

         Choose only one bag type from this list: Handbag, Backpack, Mini Handbag, Travel Bag, 
         Toiletry Bag, Crossbody Bag, Shoulder Bag, Keepall, Sling Bag, Bum Bag, Clutch Bag, Jewelry Box,
         Cosmetic Bag, Tote Bag, Underarm Bag, Bucket Bag, Triangular Bag, Chest Bag, Peekaboo Bag, 
         Baguette Handbag, Messenger Bag, Shopping Bag. 
         If you mention multiple bag types in your response, kindly select the closest one.

         Translate every text to English. 
         Rewrite the review to evoke feelings of premium quality, timeless elegance, and an irresistible urge to shop. 
         Please use the brand name as {brand} replica bag, {brand} replica bags and {brand} replica handbags in the review. 
         Avoid replying with any comments and refrain from using Chinese alphabet. 

         Create and respond only with JSON format for whole variations."""},
    ],
    temperature=0.1,
    max_tokens=600
    )

    
    # Get and categorize the informations from ChatGPT
    data = response.choices[0].message.content
    dataFromGPT = json.loads(data)

    size = dataFromGPT["size"]
    review = dataFromGPT["review"]
    bag_type = dataFromGPT["bag_type"]
    main_material = dataFromGPT["main_material"]
    product_code = dataFromGPT["product_code"]

    # Set some dynamic informations
    brandId = 9
    factory_code = "3259"
    brandLink = "1263271038"
    lining_type = "Anti Bacterial Interior Lining"

    # Extract image URLs and store them in pictures_links list
    img_tags = driver.find_elements(By.CLASS_NAME, 'image__clickhandle')
    
    window_handles = []
    pictures_links = []
    photoid = None

    print(product_description)
    for img_tag in img_tags:
        photoid = img_tag.get_attribute('data-photoid')
        createdURL = f"https://{brandLink}.x.yupoo.com/{photoid}?uid=1"

        print(createdURL)
        driver.execute_script(f"window.open('{createdURL}');")
        driver.switch_to.window(driver.window_handles[-1])
        window_handles.append(driver.current_window_handle)

        try:
            product_gallery_url = driver.find_element(By.CLASS_NAME, 'viewer__img')
            image_src = product_gallery_url.get_attribute('src')
            save_image_from_base64(image_src, f"{product_code}_{photoid}{brandId}")

            pictures_links.insert(0, f"sampleWordPressUrl/2023/08/{product_code}_{photoid}{brandId}.jpg")

            driver.close()
        except NoSuchElementException:
            print("viewer__img element not found. Skipping to the next link.")
            driver.close()
        finally:
            driver.switch_to.window(driver.window_handles[0])
            
    # Construct the pictures_links_str
    pictures_links_str = ",".join(pictures_links)

    rareId = f"{photoid}{brandId}"

    return (brand, 
            factory_code, 
            rareId, 
            product_code, 
            size, 
            bag_type, 
            main_material, 
            review, 
            lining_type, 
            pictures_links_str, 
            pictures_links)

# Function to save an image from a base64 string
def save_image_from_base64(image_url, filename):
    if image_url.startswith('//'):
        image_url = '"https:' + image_url

    base64_image = get_base64_image(image_url)

    image_path = f'samplePath/picturesDG/{filename}.jpg'

    with open(image_path, 'wb') as file:
        file.write(base64.b64decode(base64_image))

    image = Image.open(image_path)
    image.save(image_path, optimize=True, quality=70)

# Function to get base64 image data from an image URL
def get_base64_image(image_url):
    driver.get(image_url)
    time.sleep(0.5) 
    
    base64_image = driver.execute_script("""
        var img = document.querySelector('img');
        var canvas = document.createElement('canvas');
        var context = canvas.getContext('2d');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        context.drawImage(img, 0, 0);
        return canvas.toDataURL('image/jpeg').substring(22);
    """)
    return base64_image

# List of website URLs to scrape
website_urls = [
"https://1263271038.x.yupoo.com/albums/138779032?uid=1&isSubCate=false&referrercate=3275081",
"https://1263271038.x.yupoo.com/albums/138778876?uid=1&isSubCate=false&referrercate=3275081",
]

# Open a CSV file for writing
with open('Gucci_WooCommerce.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Write header row to the CSV file, for WooCommerce / WordPress Import
    writer.writerow(["ID"
                     , "Parent"
                     , "Type"
                     , "SKU"
                     , "Name"
                     , "Description"
                     , "Categories"
                     , 'Images'
                     , "Regular price"

                     , "Attribute 1 name"
                     , "Attribute 1 value(s)"

                     , "Attribute 2 name"
                     , "Attribute 2 value(s)"

                     , "Attribute 3 name"
                     , "Attribute 3 value(s)"

                     , "Attribute 4 name"
                     , "Attribute 4 value(s)"

                     , "Attribute 5 name"
                     , "Attribute 5 value(s)"

                     , "Attribute 6 name"
                     , "Attribute 6 value(s)"

                     , "Attribute 7 name"
                     , "Attribute 7 value(s)"
                     , "Attribute 7 visible"
                     
])

    index = 1

    for url in website_urls:
        try:
            # Call the get_headings_and_text function to extract information
            brand, factory_code, rareId, product_no, size, bag_type, main_material, review, lining_type, pictures_links_str, pictures_links = get_headings_and_text(url)

            name = f"{brand} Replica Bag {product_no}/{factory_code}"
            description = f'<div class="soldaki-yazi">{review}</div><div class="sagdaki-resim"><img src={pictures_links[0]} width="500"></div>'
            categories = f"{brand} > {bag_type}"

            writer.writerow([f"{index}",None,"variable",f"{rareId}",name, description, categories, pictures_links_str,None,"BRAND",brand,"MODEL NO",product_no,"BAG TYPE",bag_type,"SIZE",size,"MATERIAL",main_material,"LINING TYPE",lining_type,"CHOOSE QUALITY","High Quality, Highest Quality",0])
            writer.writerow([None,f"id:{index}","variation",None,None, None, None, None,"750","CHOOSE QUALITY","High Quality",None,None,None,None,None,None,None,None,None,None,None,None,0])
            writer.writerow([None,f"id:{index}","variation",None,None, None, None, None,"1350","CHOOSE QUALITY","Highest Quality",None,None,None,None,None,None,None,None,None,None,None,None,0])

            print("---------------",index,"---------------")
            index += 1

        except Exception as e:
            print("Hata oluştu:", e)
            traceback.print_exc()
            pass

# Close the WebDriver
driver.quit()