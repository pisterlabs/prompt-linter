from django.shortcuts import render, redirect
from .models import Product, ReviewRating, ProductGallery, Variation
from .forms import ReviewForm
from category.models import Category
from django.shortcuts import get_object_or_404
from carts.models import CartItem
from carts.views import _cart_id
from django.core.paginator import EmptyPage, PageNotAnInteger, Paginator
from django.db.models import Q
from django.contrib import messages
from orders.models import OrderProduct
import os
from utils import bedrock, print_ww
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain import PromptTemplate
import warnings
from PIL import Image
import base64
import io
import json
import random
from decouple import config
import boto3
import string
import numpy as np
import requests
import psycopg2
from pgvector.psycopg2 import register_vector

# Initialize Bedrock client 
boto3_bedrock = bedrock.get_bedrock_client(assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None), region=config("AWS_DEFAULT_REGION"))

# Initialize S3 client
s3 = boto3.client('s3', region_name=config("AWS_DEFAULT_REGION"))

# Initialize secrets manager
secrets = boto3.client('secretsmanager', region_name=config("AWS_DEFAULT_REGION"))

# Create your views here.

####################### START SECTION - OTHER WEB APPLICATION FEATURES ##########################

## This section contains functions that will be used for other functionalities of our retail web application.
## For example, submitting user review or displaying products in catalog.

## This section can be safely ignored
## Please don't modify anything in this section

def store(request, category_slug=None):
    categories = None
    products = None

    if category_slug != None:
       categories = get_object_or_404(Category, slug=category_slug)
       products = Product.objects.filter(category=categories, is_available=True).order_by('category')
       paginator = Paginator(products, 6)
       page = request.GET.get('page')
       paged_products = paginator.get_page(page)
       product_count = products.count()
    else:
        products = Product.objects.all().filter(is_available=True).order_by('category')
        paginator = Paginator(products, 6)
        page = request.GET.get('page')
        paged_products = paginator.get_page(page)
        product_count = products.count()

    context = {
        'products': paged_products,
        'product_count': product_count,
    }
    return render(request, 'store/store.html', context)

def product_detail(request, category_slug, product_slug):
    request.session['product_description_flag'] = False
    request.session['product_details'] = None
    request.session['draft_flag'] = False
    request.session['summary_flag'] = False
    request.session['image_flag'] = False
    request.session['change_prompt'] = None
    request.session['negative_prompt'] = None
    request.session.modified = True

    try:
        single_product = Product.objects.get(category__slug=category_slug, slug=product_slug)
        in_cart = CartItem.objects.filter(cart__cart_id=_cart_id(request), product=single_product).exists()
    except Exception as e:
        raise e

    if request.user.is_authenticated:
        try:
            orderproduct = OrderProduct.objects.filter(user=request.user, product_id=single_product.id).exists()
        except OrderProduct.DoesNotExist:
            orderproduct = None
    else:
        orderproduct = None

    # Get reviews
    reviews = ReviewRating.objects.filter(product_id=single_product.id, status=True)

    # Get product gallery
    product_gallery = ProductGallery.objects.filter(product_id=single_product.id)

    context = {
        'single_product': single_product,
        'in_cart'       : in_cart,
        'orderproduct': orderproduct,
        'reviews': reviews,
        'product_gallery': product_gallery,
    }
    #print("user ->" +reviews[0].user.full_name())
    return render(request, 'store/product_detail.html', context)

def search(request):
    if 'keyword' in request.GET:
        keyword = request.GET['keyword']
        if keyword:
            products = Product.objects.order_by('-created_date').filter(Q(description__icontains=keyword) | Q(product_name__icontains=keyword))
            product_count = products.count()
    context = {
        'products': products,
        'product_count': product_count,
    }
    return render(request, 'store/store.html', context)

def submit_review(request, product_id):
    url = request.META.get('HTTP_REFERER')
    first_name=request.POST.get('first_name')
    last_name=request.POST.get('last_name')
    if request.method == 'POST':
        try:
            reviews = ReviewRating.objects.get(first_name=first_name, last_name=last_name, product__id=product_id)
            form = ReviewForm(request.POST, instance=reviews)
            form.save()
            messages.success(request, 'Thank you! Your review has been updated.')
            return redirect(url)
        except ReviewRating.DoesNotExist:
            form = ReviewForm(request.POST)
            if form.is_valid():
                data = ReviewRating()
                data.subject = form.cleaned_data['subject']
                data.rating = form.cleaned_data['rating']
                data.review = form.cleaned_data['review']
                data.ip = request.META.get('REMOTE_ADDR')
                data.product_id = product_id
                #data.user_id = request.user.id
                data.first_name = first_name
                data.last_name = last_name
                data.save()
                messages.success(request, 'Thank you! Your review has been submitted.')
                return redirect(url)

####################### END SECTION - OTHER WEB APPLICATION FEATURES ##########################

####################### START SECTION - HANDLER FUNCTIONS GENAI FEATURES ##########################

## Functions in this section will be used to: 

# 1. Render HTML pages needed for the GenAI features we are going to implement
# 2. Save LLM response to web application database
# 3. Any andler functions used for manipulating input fed to the LLM 

## This section can be safely ignored
## Please don't modify anything in this section

#### HANDLER FUNCTIONS FOR GENAI VIEWS (PLACEHOLDERS)
#### HANDLER FUNCTIONS FOR GENAI VIEWS (PLACEHOLDERS)
def create_review_response(request, product_id, review_id):
   url = request.META.get('HTTP_REFERER')
   messages.error(request, "This feature has not been implemented yet")
   return redirect(url)

def create_design_ideas(request, product_id):
   url = request.META.get('HTTP_REFERER')
   messages.error(request, "This feature has not been implemented yet")
   return redirect(url)

def generate_review_summary(request, product_id):
   url = request.META.get('HTTP_REFERER')
   messages.error(request, "This feature has not been implemented yet")
   return redirect(url)

def ask_question(request):
   url = request.META.get('HTTP_REFERER')
   messages.error(request, "This feature has not been implemented yet")
   return redirect(url)

def vector_search(request):
   url = request.META.get('HTTP_REFERER')
   messages.error(request, "This feature has not been implemented yet")
   return redirect(url)

#### HANDLER FUNCTIONS FOR GENERATING PRODUCT DESCRIPTION FEATURE ####

# This function is used to just render HTML page for generate product description functionality
def generate_description(request, product_id):
   try:
        # get product from product ID 
        single_product = Product.objects.get(id=product_id)

   except Exception as e:
        raise e
   
   # pass product object to context (to be used in generate_description.html)
   context = {
        'single_product': single_product,
    }
   
   # render HTML page generate_description.html
   return render(request, 'store/generate_description.html', context)

#This function is used for saving product description to database
def save_product_description(request, product_id):
    try:
        single_product = Product.objects.get(id=product_id)

        # If user input is to save description
        if 'save_description' in request.POST:
            single_product.description = request.POST.get('generated_description')
            single_product.save()
            success_message = "The product description for " + single_product.product_name + " has been updated successfully. "
            messages.success(request, success_message)
            return redirect('product_detail', single_product.category.slug, single_product.slug)
        # If user input is to regenerate
        elif 'regenerate' in request.POST:
            request.session['product_description_flag'] = False
            request.session.modified = True
            return redirect('generate_description', single_product.id)
        else:
            # do nothing
            pass
    except Exception as e:
        raise e

#### HANDLER FUNCTIONS FOR DRAFTING RESPONSE TO CUSTOMER REVIEW FEATURE ####

# This function is used to just render HTML page for create response to customer review functionality
def create_response(request, product_id, review_id):
    try:
        # get product from product ID
        single_product = Product.objects.get(id=product_id)
        # get single customer review using product ID, review ID
        review = ReviewRating.objects.get(product=single_product, id=review_id)

    except Exception as e:
            raise e
    
    # pass objects to context (to be used in create_response.html)
    context = {
            'single_product': single_product,
            'review': review,
        }
    # render HTML page create_response.html
    return render(request, 'store/create_response.html', context)

# This function is used for saving customer review response to database
def save_review_response(request, product_id, review_id):
    try:
        # get single product review using product ID and review ID 
        request.session['draft_flag'] = False
        single_product = Product.objects.get(id=product_id)
        review = ReviewRating.objects.get(product=single_product, id=review_id)

        # If user input is to save response
        if 'save_response' in request.POST:
            review.generated_response = request.POST.get('generated_response')
            review.prompt = request.session.get('draft_prompt')
            review.save()
            success_message = "The response for the review of " + single_product.product_name + " has been updated successfully. "
            messages.success(request, success_message)
            return redirect('product_detail', single_product.category.slug, single_product.slug)
        
        # If user input is to regenerate review response
        elif 'regenerate' in request.POST:
            request.session.modified = True
            return redirect('create_response', single_product.id, review.id)
        else:
            # do nothing
            pass
    except Exception as e:
        raise e

#### HANDLER FUNCTIONS FOR SUMMARIZING CUSTOMER REVIEWS FEATURE ####

# This function is used to just render HTML page for summarize customer reviews functionality
def generate_summary(request, product_id): 
    try:
        # get product from product ID
        single_product = Product.objects.get(id=product_id)
        # get all customer reviews for this product
        product_reviews = ReviewRating.objects.filter(product=single_product, status=True)

    except Exception as e:
            raise e
    
    # pass objects to context (to be used in generate_summary.html)
    context = {
            'single_product': single_product,
            'reviews': product_reviews,
        }
    
    # render HTML page generate_summary.html
    return render(request, 'store/generate_summary.html', context)

# This function is used for saving summarized customer reviews to database
def save_summary(request, product_id):
    try:
        # get single product review using product ID and review ID 
        single_product = Product.objects.get(id=product_id)
        request.session['summary_flag'] = False

        # If user input is to save review summary
        if 'save_summary' in request.POST:
            single_product.review_summary = request.session['generated_summary']
            single_product.save()
            success_message = "The summary for the review of " + single_product.product_name + " has been updated successfully. "
            messages.success(request, success_message)
            return redirect('product_detail', single_product.category.slug, single_product.slug)
        # If user input is to regenerate review summary
        elif 'regenerate' in request.POST:
            request.session.modified = True
            return redirect('generate_summary', single_product.id)
        else:
            # do nothing
            pass
    except Exception as e:
        raise e

#### HANDLER FUNCTIONS FOR CREATING NEW DESIGN IDEAS FEATURE ####

# This function is used to just render the HTML page studio.html
def design_studio(request, product_id):
    single_product = Product.objects.get(id=product_id)
    context = {
        'single_product': single_product,
    }
    return render(request, 'store/studio.html', context)

# Handy function to convert an image to base64 string
# Stabile Diffusion LLM expects the input image to be in base64 string format
def image_to_base64(img) -> str:
    """Convert a PIL Image or local image file path to a base64 string for Amazon Bedrock"""
    if isinstance(img, str):
        if os.path.isfile(img):
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

#### HANDLER FUNCTIONS FOR QUESTION ANSWERING FEATURE ####

# This function is used for extracting string within a tag. For example, get string embedded within <query></query>
def extract_strings_recursive(test_str, tag):
    # finding the index of the first occurrence of the opening tag
    start_idx = test_str.find("<" + tag + ">")
 
    # base case
    if start_idx == -1:
        return []
 
    # extracting the string between the opening and closing tags
    end_idx = test_str.find("</" + tag + ">", start_idx)
    res = [test_str[start_idx+len(tag)+2:end_idx]]
 
    # recursive call to extract strings after the current tag
    res += extract_strings_recursive(test_str[end_idx+len(tag)+3:], tag)
 
    return res

####################### END SECTION - HANDLER FUNCTIONS GENAI FEATURES ##########################


####################### START SECTION - IMPLEMENT GENAI FEATURES FOR WORKSHOP ##########################

#### This is the ONLY section in store/views.py where you will add functions needed for implementing GenAI features into your retail application
#### Please don't edit any other sections in THIS file. 

#### FEATURE 1 - GENERATE PRODUCT DESCRIPTION ####
# This function is used for generating product description using LLM from Bedrock
def generate_product_description(request, product_id):
    
    # STEP 1 - The product ID will be passed from the retail website. 
    # Filter out the Product object from this product ID and get the product name, brand, and colors available from the Product object. 
    # All these values will be passed to the LLM prompt template as input variables.
    single_product = Product.objects.get(id=product_id)
    product_colors = []
    product_vars = Variation.objects.filter(product=single_product, variation_category="color")
    for variation in product_vars:
        product_colors.append(variation.variation_value)

    product_brand = single_product.product_brand
    product_category = single_product.category
    product_name = single_product.product_name

    # STEP 2 - Get the product details and maximum number of words for the product description from the user input form in the web application. This will be passed as variables to the LLM prompt template. 
    product_details = request.GET.get('product_details')
    max_length = request.GET.get('wordrange')

    # STEP 3 - Get the inference parameters for Anthropic Claude model from the website. 
    inference_modifier = {}
    inference_modifier['max_tokens_to_sample'] = int(request.GET.get('max_tokens_to_sample') or 200)
    inference_modifier['temperature'] = float(request.GET.get('temperature') or 0.5)
    inference_modifier['top_k'] = int(request.GET.get('top_k') or 250)
    inference_modifier['top_p'] = float(request.GET.get('top_p') or 1)
    inference_modifier['stop_sequences'] = ["\n\nHuman"]

    try:

        # STEP 4 - Initialize Anthropic Claude LLM from Bedrock. 
        textgen_llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=boto3_bedrock,
            model_kwargs=inference_modifier,
        )
        
        # STEP 5 - Create a prompt template with all the input variables from STEP 2 and 3. 
        # This prompt template asks Bedrock to create a catchy product description based on the input variables we pass. 
        # These are the 6 input variables: 
        # product name
        # product brand
        # product color
        # product category (shirt, jeans etc.)
        # product details obtained from user input, and max length of the description requested from LLM. 

        prompt_template = PromptTemplate(
            input_variables=["brand", "colors", "category", "length", "name","details"], 
            template="""
                    Human: Create a catchy product description for a {category} from the brand {brand}. 
                    Product name is {name}. 
                    The number of words should be less than {length}. 
                    
                    Following are the product details:  
                    
                    <product_details>
                    {details}
                    </product_details>
                    
                    Briefly mention about all the available colors of the product.
                    
                    Example: Available colors are Blue, Purple and Orange. 
                    
                    If the <available_colors> is empty, don't mention anything about the color of the product.
                    
                    <available_colors>
                    {colors}
                    </available_colors>

                    Assistant:

                    """
                )

        # STEP 6 - Initialize prompt template with all the prompt input variables. 
        prompt = prompt_template.format(brand=product_brand, 
                                         colors=product_colors,
                                         category=product_category,
                                         length=max_length,
                                         name=product_name,
                                         details=product_details)
        
        # STEP 7 - Retrieve the generated product description from Bedrock.
        response = textgen_llm(prompt)
        # get the second paragraph i.e, only the product description 
        generated_description = response[response.index('\n')+1:]

    except Exception as e:
        raise e
    
    # STEP 8 - Set Django session variables which will be used in the HTML template to display the results. 
    request.session['product_details'] = product_details
    request.session['generated_description'] = generated_description
    request.session['prompt'] = prompt
    request.session['product_description_flag'] = True
    request.session.modified = True

    # STEP 9 - Re-direct to the same page. 
    # Now the page will display the generated product description from Bedrock. 
    # From there, user can either save description or regenerate it. 

    url = request.META.get('HTTP_REFERER')
    return redirect(url)


#### FEATURE 2 - DRAFTING RESPONSE TO CUSTOMER REVIEWS ####
# This function is used for drafting response to customer reviews using LLM from Bedrock
def create_review_response(request, product_id, review_id):


    # STEP 1 - The product ID and customer review ID will be passed from the retail website. 
    # Filter out the Product object using that product ID and Review object from the review ID. 
    # Get the product name, review text from the objects. 
    # All these values will be passed to the LLM prompt template as input variables.

    product = Product.objects.get(id=product_id)
    review = ReviewRating.objects.get(product=product, id=review_id)

    product_name = product.product_name
    review_text = review.review

    try:

        # STEP 2 - Get the maximum number of words to create response for the customer review. This will be passed as a variable to the LLM prompt template.
        max_length = request.GET.get('wordrange')

        # STEP 3 - Get the inference parameters for Anthropic Claude model from the retail website .
        inference_modifier = {}
        inference_modifier['max_tokens_to_sample'] = int(request.GET.get('max_tokens_to_sample') or 200)
        inference_modifier['temperature'] = float(request.GET.get('temperature') or 0.5)
        inference_modifier['top_k'] = int(request.GET.get('top_k') or 250)
        inference_modifier['top_p'] = float(request.GET.get('top_p') or 1)
        inference_modifier['stop_sequences'] = ["\n\nHuman"]

        # STEP 4 - Initialize Anthropic Claude LLM from Bedrock.
        textgen_llm = Bedrock(
            model_id="anthropic.claude-instant-v1",
            client=boto3_bedrock,
            model_kwargs=inference_modifier,
        )
        
        # STEP 5 - Create a prompt template with all the input variables from STEP 2 and 3. 
        # This prompt template asks Bedrock to create a response to customer review based on their sentiments. 
        # If the sentiment is negative, the manager will provide their phone number, requesting the customer to reach out directly.

        # This prompt template needs 7 input variables: product name, customer name, manager's name, manager's email, manager's phone number, response length, customer review text.
        prompt_template = PromptTemplate(
            input_variables=["product_name","customer_name","manager_name","email","phone","length","review"], 
            template="""
                    Human: 
                    
                    I'm the manager of re:Invent retails. 
                    
                    Draft a response for the review of the product {product_name} from our customer {customer_name}. 
                    The number of words should be less than {length}. 
                    
                    My contact information is email: {email}, phone: {phone}.
                    
                    <customer_review>
                        {review}
                    <customer_review>

                    <example_response_pattern>
                    
                        Dear <customer_name>,
                        <content_body>

                        <if negative review> 
                            Don't hesitate to reach out to me at {phone}.
                        <end if> 

                        Sincerely,
                        {manager_name}
                        <signature>
                        {email}
                    
                    </example_response_pattern>
                    
                    Assistant:
                    
                    """
                )

        # STEP 6 - Initialize prompt template with all the prompt input variables.
        prompt = prompt_template.format(product_name=product_name,
                                         customer_name=review.first_name,
                                         manager_name=request.user.full_name(),
                                         email=request.user.email,
                                         phone=request.user.phone_number,
                                         length=max_length,
                                         review=review_text)
        
        # STEP 7 - Call the LLM from Bedrock and retrieve the generated response to customer review.
        response = textgen_llm(prompt)

        # Get the second paragraph i.e, only the response to customer review
        generated_response = response[response.index('\n')+1:]

    except Exception as e:
        raise e

    # STEP 8 - Set Django session variables which will be used in the HTML template to display the results.
    request.session['generated_response'] = generated_response
    request.session['draft_prompt'] = prompt
    request.session['draft_flag'] = True
    request.session.modified = True

    # STEP 9 - Re-direct to the same page. Now the page will display the generated response to customer review from Bedrock.
    url = request.META.get('HTTP_REFERER')
    return redirect(url)


#### FEATURE 3 - CREATE NEW DESIGN IDEAS FROM PRODUCT ####

# This function is used for creating design ideas (images) for a product
def create_design_ideas(request, product_id):

    # STEP 1 - The product ID will be passed from the retail website. Filter out the Product object using that product ID. 
    # Get the S3 bucket name from the Django configuration file (we have already configured the S3 bucket name). This is the bucket where we store the generated images. 
    single_product = Product.objects.get(id=product_id)
    bucket_name = config('AWS_STORAGE_BUCKET_NAME')
    
    try:
        # If user chose to delete previously generated image from Stable Diffusion model 
        if 'delete_previous' in request.GET:
            # delete previously generated image  
            product_gallery_del = ProductGallery.objects.filter(product=single_product, image=request.session['image_file_path'])
            if product_gallery_del:
                # Delete previously generated image from S3
                s3.delete_object(Bucket=bucket_name, Key=request.session['image_file_path'])
                # Delete this image in product image gallery in Django
                product_gallery_del.delete()
                request.session['image_flag'] = False
            # redirect to design studio
            messages.info(request, "Deleted previously generated design idea. You can now create a new design idea.")
            return redirect('design_studio', single_product.id)
        
        # If user chose to delete all generated images from Stable Diffusion model
        if 'delete_all' in request.GET:
            # delete existing image gallery 
            request.session['image_flag'] = False
            product_gallery_del = ProductGallery.objects.filter(product=single_product)
            if product_gallery_del:
                 # Delete all generated images from S3
                for x in product_gallery_del:
                    s3.delete_object(Bucket=bucket_name, Key="media/"+str(x.image))
                # Delete product image gallery in Django
                product_gallery_del.delete()
            # redirect to product page
            messages.info(request, "Deleted all generated designs")
            return redirect('product_detail', single_product.category.slug, single_product.slug)
        
        # Generate new images using Bedrock
        else:
            # STEP 2 - Every product has an image in the catalog. Open the product's image. 
            # Resize product image to 512x512 and convert to base64 string format to comply with the requirements for Stable Diffusion LLM. 
            response = s3.get_object(Bucket=bucket_name, Key="media/"+str(single_product.images))
            input_file_stream = response['Body']
            image = Image.open(input_file_stream)
            resize = image.resize((512,512))
            init_image_b64 = image_to_base64(resize)

            # STEP 3 - Get the Stable Diffusion inference parameters passed from the retail website 
            # This prompt is used to generate new ideas from the existing image
            change_prompt = request.GET.get('change_prompt')

            # Negative prompts that will be given -1.0 weight while generating new image
            negprompts = request.GET.get('negative_prompt')
            negative_prompts = []
            if negprompts:
                for negprompt in negprompts.split('\n'):
                    negative_prompts.append(negprompt.replace('\r',''))
            
            # Other Stable Diffusion parameters
            start_schedule = 0.5 if request.GET.get('start_schedule') is None else float(request.GET.get('start_schedule'))
            steps = 30 if request.GET.get('steps') is None else int(request.GET.get('steps'))
            cfg_scale = 10 if request.GET.get('cfg_scale') is None else int(request.GET.get('cfg_scale'))
            image_strength = 0.5 if request.GET.get('image_strength') is None else float(request.GET.get('image_strength'))
            denoising_strength = 0.5 if request.GET.get('denoising_strength') is None else float(request.GET.get('denoising_strength'))
            seed = random.randint(1, 1000000) if request.GET.get('seed') is None else int(request.GET.get('seed'))
            style_preset = "photographic" if request.GET.get('style_preset') is None else request.GET.get('style_preset')


            # Construct request body for Stable Diffusion model
            sd_request = json.dumps({
                        "text_prompts": (
                            [{"text": change_prompt, "weight": 1.0}]
                            + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                        ),
                        "cfg_scale": cfg_scale,
                        "init_image": init_image_b64,
                        "seed": seed,
                        "start_schedule": start_schedule,
                        "steps": steps,
                        "style_preset": style_preset,
                        "image_strength":image_strength,
                        "denoising_strength": denoising_strength
                    })
            
            # STEP 4 - Invoke the Stable Diffusion model from Bedrock passing all the inference parameters 
            response = boto3_bedrock.invoke_model(body=sd_request, modelId="stability.stable-diffusion-xl")

            # STEP 5 - Extract image from the API response
            # Save the image to S3 bucket and the product's image gallery
            response_body = json.loads(response.get("body").read())
            genimage_b64_str = response_body["artifacts"][0].get("base64")
            genimage = Image.open(io.BytesIO(base64.decodebytes(bytes(genimage_b64_str, "utf-8"))))
            
            # Save the image to an in-memory file
            in_mem_file = io.BytesIO()
            genimage.save(in_mem_file, format="PNG")
            in_mem_file.seek(0)

            # Upload image to static s3 path
            image_file_path = single_product.slug + "_generated" + ''.join(random.choices(string.ascii_lowercase, k=5)) + ".png"
        
            s3.upload_fileobj(
                in_mem_file, # image
                bucket_name,
                'media/store/products/' + image_file_path
            )

            # Save generated image to database
            product_gallery = ProductGallery()
            product_gallery.product = single_product
            product_gallery.image = 'store/products/' + image_file_path
            product_gallery.save()

            # STEP 6 - Set Django session variables which will be used in the HTML template to display the generated image. 
            request.session['change_prompt'] = change_prompt
            request.session['negative_prompt'] = negprompts
            request.session['image_file_path'] = 'store/products/' + image_file_path
            request.session['image_flag'] = True
            request.session['image_url'] = product_gallery.image.url
            request.session.modified = True

            # Signal success message to user
            messages.success(request, "Design idea generated. Scroll down to check it out!")
            
    except Exception as e: 
        raise e
        
    # STEP 7 - Re-direct to the same page. Now the page will display the generated design ideas from Bedrock.
    url = request.META.get('HTTP_REFERER')
    return redirect(url)


#### FEATURE 4 - SUMMARIZE CUSTOMER REVIEWS FOR A PRODUCT ####
# This function is used for summarizing customer reviews using LLM from Bedrock
def generate_review_summary(request, product_id):
    #  STEP 1 - The product ID will be passed from the retail website. 
    # Filter out the Product object using that product ID and get all the customer reviews for this product
    single_product = Product.objects.get(id=product_id)
    product_reviews = ReviewRating.objects.filter(product=single_product)

    chunk_size = int(request.POST.get('chunk_size') or 1000)
    chunk_overlap = int(request.POST.get('chunk_overlap') or 100)

    # Enclose all the customer reviews in <review></review> tags
    # this will be used in the prompt template for summarizing customer reviews
    # doing it this way helps LLM understand our instruction better
    review_digest = ''

    for review in product_reviews:
        review_digest += "<review>" + '\n'
        review_digest += review.review + '\n'
        review_digest += "</review>" + '\n\n'

    # STEP 2 - Split the reviews into chunks using LangChain's TextSplitter transformer
    # Let's split our reviews into chunks using Langchain's RecursiveCharacterTextSplitter
    # chunk size and overlap are defined as input parameters
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    customer_reviews = text_splitter.create_documents([review_digest])

    try:
        # STEP 3 - Get the inference parameter for the LLM from the retail website 
        # If user chose Claude
        if 'Claude' in request.POST.get('llm'):
            #Inference parameters for Claude Anthropic
            inference_modifier = {}
            inference_modifier['max_tokens_to_sample'] = int(request.POST.get('claude_max_tokens_to_sample') or 1024)
            inference_modifier['temperature'] = float(request.POST.get('claude_temperature') or 0.5)
            inference_modifier['top_k'] = int(request.POST.get('claude_top_k') or 250)
            inference_modifier['top_p'] = float(request.POST.get('claude_top_p') or 1)
            inference_modifier['stop_sequences'] = ["\n\nHuman"]

            # Initialize Claude LLM
            textsumm_llm = Bedrock(
                model_id="anthropic.claude-instant-v1",
                client=boto3_bedrock,
                model_kwargs=inference_modifier,
            )
        
         # If user chose Titan
        elif 'Titan' in request.POST.get('llm'):
            #Inference parameters for Titan
            inference_modifier = {}
            inference_modifier['maxTokenCount'] = int(request.POST.get('titan_max_tokens_to_sample') or 1024)
            inference_modifier['temperature'] = float(request.POST.get('titan_temperature') or 0.5)
            inference_modifier['topP'] = int(request.POST.get('titan_top_p') or 250)

            # Initialize Titan LLM
            textsumm_llm = Bedrock(
                model_id="amazon.titan-text-express-v1",
                client=boto3_bedrock,
                model_kwargs=inference_modifier,
                )
            
        else:
            pass

        # STEP 4 - Create prompt for summarizing customer reviews. 
        # Passing product name and all the customer reviews as parameters to the prompt template. 
        summary_prompt='''

            Human: 

            Your task is to summarize the customer reviews for the product {product_name}. 
            Following are the customer reviews enclosed in <customer_reviews> tag. 
            
            <customer_reviews>
                `{text}`
            </customer_reviews>
            
            <example_review_summary_format>

            Here's a customer review summary of {product_name}
            Pros:
                
                - pro 1
                - pro 2 
                
            Cons:
            
                - con 1 
                - con 2
            
            Overall summary of the customer reviews. 

            </example_review_summary_format>

            Do not suggest the customer to make a purchasing decision. 
            Overall summary should be objective and should only echo the customer reviews.
            
            
            Assistant:
            
        '''
        
        # STEP 5 - Create prompt template with two input variables: 
        # product name and text i.e, customer reviews for this product
        summary_prompt_template = PromptTemplate(
            template=summary_prompt, 
            input_variables=['product_name','text']
        )

        summary_prompt_string = summary_prompt_template.format(product_name=single_product.product_name, text=customer_reviews)

        # STEP 6 - Invoke Langchain's load_summarize_chain to summarize the product reviews
        # Chain type "stuff" takes the list of customer reviews, inserts them all into a prompt and passes that prompt to an LLM.
        from langchain.chains.summarize import load_summarize_chain

        summary_chain = load_summarize_chain (
            llm=textsumm_llm,
            chain_type='stuff',
            prompt=summary_prompt_template,
            verbose=False
        )

        # STEP 7 - Pass in the input variables to the prompt template and invoke the summary chain using Bedrock LLM 
        summary=summary_chain.run({
                "product_name": single_product.product_name,
                "input_documents": customer_reviews
                })

        # STEP 8 - Set session parameters to use in HTML template
        request.session['generated_summary'] = summary
        request.session['summary_prompt'] = summary_prompt_string
        request.session['summary_flag'] = True
        request.session.modified = True

    except Exception as e: 
        raise e

    # STEP 9 - Re-direct to the same page. Now the page will display the summarized product review from Bedrock.    
    url = request.META.get('HTTP_REFERER')
    return redirect(url)

#### FEATURE 5 - QUESTION ANSWERING WITH SQL GENERATION ####

# This function is used for answering user questions in natural language using SQL generation and result interpretation by LLM
def ask_question(request):
    # initialize variables
    context={}
    is_query_generated = False
    describe_query_result = ''

    if 'question' in request.GET:
        # STEP 1 - Get the user question asked in natural language from the retail web application 
        question = request.GET.get('question')

        # STEP 2 - Get the schema file from S3 that has information about the tables storing the retail web application data 
        s3 = boto3.client('s3')
        resp = s3.get_object(Bucket=config('AWS_STORAGE_BUCKET_NAME'), Key="data/schema-postgres.sql")
        schema = resp['Body'].read().decode("utf-8")

        # STEP 3 - Create a prompt template to generate an SQL query based on the question and the database schema. 
        # We are passing PostgresQL documentation to help with the SQL generation. 
        # Generated query will be embedded in <query></query> tags
        prompt_template = """
            Human: Create an Postgres SQL query for a retail website to answer the question keeping the following rules in mind: 
            
            1. Database is implemented in Postgres SQL.
            2. Follow the Postgres syntax carefully while generating the query.
            3. Enclose the query in <query></query>. 
            4. Use "like" and upper() for string comparison on both left hand side and right hand side of the expression. For example, if the query contains "jackets", use "where upper(product_name) like upper('%jacket%')". 
	        5. Do not use upper() on integer or non-string or non-varchar table columns!
            6. If the question is generic, like "where is mount everest" or "who went to the moon first", then do not generate any query in <query></query> and do not answer the question in any form. Instead, mention that the answer is not found in context.
            7. If the question is not related to the schema, then do not generate any query in <query></query> and do not answer the question in any form. Instead, mention that the answer is not found in context.
            8. If the question is asked in a language other than English, convert the question into English before constructing the SQL query. The string and varchar table columns in the database are always in English.  

            <schema>
                {schema}
            </schema>

            Question: {question}

            Assistant:
            
            """

        # Prompt template variables
        prompt_vars = PromptTemplate(template=prompt_template, input_variables=["question","schema"])
            
        # Initialize LLM
        llm = Bedrock(model_id="anthropic.claude-instant-v1", client=boto3_bedrock)
        
        # Initialize prompt template with the question and database schema
        prompt = prompt_vars.format(question=question, schema=schema)

        try: 
            # STEP 4 - Call the LLM from Bedrock to generate the SQL query based on the question and the database schema. 
            llm_response = llm(prompt)

            # Check if query is generated under <query></query> tags as instructed in our prompt
            if "<query>".upper() not in llm_response.upper():
                print("no query generated")
                is_query_generated  = False
                describe_query_result = llm_response
                resultset=''
                query=''
            
            else: 
                # STEP 5 - Extract the query from the LLM response and execute the query on the PostgreSQL database. 
                is_query_generated = True
                # Extract the query from the response
                query = extract_strings_recursive(llm_response, "query")[0]
                print("Query generated by LLM: " +query)

                # Get database connection details from Secrets Manager
                response = secrets.get_secret_value(SecretId=config('AWS_DATABASE_SECRET_ID'))
                database_secrets = json.loads(response['SecretString'])

                # Connect to PostgreSQL database
                dbhost = database_secrets['host']
                dbport = database_secrets['port']
                dbuser = database_secrets['username']
                dbpass = database_secrets['password']
                dbname = database_secrets['name']

                dbconn = psycopg2.connect(host=dbhost, user=dbuser, password=dbpass, port=dbport, database=dbname, connect_timeout=10)
                dbconn.set_session(autocommit=True)
                cursor = dbconn.cursor()

                # Execute the extracted query
                cursor.execute(query)
                query_result = cursor.fetchall()

                # Close database connection
                cursor.close()
                dbconn.close()
                
                # get query result
                resultset = ''
                if len(query_result) > 0:
                    for x in query_result:
                        resultset = resultset + ''.join(str(x)) + "\n"

                print("Query result: \n" +resultset)

                # STEP 6 - Invoke the Bedrock LLM once again to interpret the query results 
                # This prompt template defines rules while describing query result. 
                # This is the final result that will be seen by the user as an answer to their question. 
                # Idea is to derive natural language answer for a natural language question. 
                prompt_template = """

                Human: This is a Q&A application. We need to answer questions asked by the customer at an e-commerce store. 
                The question asked by the customer is {question}
                We ran an SQL query in our database to get the following result. 

                <resultset>
                {resultset}
                </resultset>

                Summarize the above result and answer the question asked by the customer keeping the following rules in mind: 
                1. Don't make up answers if <resultset></resultset> is empty or none. Instead, answer that the item is not available based on the question.
                2. Mask the PIIs phone, email and address if found the answer with "<PII masked>"
                3. Don't say "based on the output" or "based on the query" or "based on the question" or something similar.  
                4. Keep the answer concise. 
                5. Don't give an impression to the customer that a query was run. Instead, answer naturally. 

                Assistant:

                """

                # Pass user question and query result to prompt template
                prompt_vars = PromptTemplate(template=prompt_template, input_variables=["question","resultset"])

                prompt = prompt_vars.format(question=question, resultset=resultset)

                # Invoke LLM and get response
                describe_query_result = llm(prompt)
                print("describe_query_result " + describe_query_result)

                # If length of response is 0, then set response to "Sorry, I could not answer that question."
                if len(describe_query_result) == 0:
                    describe_query_result = "Sorry, I could not answer that question."

        except Exception as e:
            query = "Following exception was received. Please try again.\n\n" + str(e)

        # STEP 7 - Set Django session variables which will be used in the HTML template to display the answer
        context = {
            "question": question,
            "query": query,
            "is_query_generated": is_query_generated,
            "describe_query_result": describe_query_result,
        }

    # STEP 8 - Re-direct to the same page. Now the page will display the generated answer from Bedrock.
    return render(request, 'store/question.html', context)



#### FEATURE 6 - VECTOR SEARCH ####



####################### END SECTION - IMPLEMENT GENAI FEATURES FOR WORKSHOP ##########################

