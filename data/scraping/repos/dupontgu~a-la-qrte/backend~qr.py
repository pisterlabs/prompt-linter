import pytesseract
from PIL import Image, ImageOps
from os import listdir, makedirs
from playwright.async_api import async_playwright
import openai
from time import time
from asyncio import sleep, timeout
from pdf2image import convert_from_path
from unidecode import unidecode
from urllib.parse import urlparse
import aiohttp

openai.api_key = "[lol I almost committed mine - put yours here!]"
img_dir = "img"
temp_dir = "temp"

# if the file is on drive, force a download of the actual file
# by ensuring the export parm is set to download
def check_drive_url(url):
    pr = urlparse(url)
    if pr.netloc != 'drive.google.com':
        return url
    if 'export=download' in pr.query:
        return url
    print("converting drive url to direct download")
    file_id = pr.path.split('/')[3]
    new_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(new_url)
    return new_url

# process this url as a normal webpage
# click_ui bool determines whether or not to click on any recognized buttons
# such as "Order Online", which is statistically likely to lead to a menu :)
async def web_page(url, img_path, click_ui):
    filename = f"{img_path}/out.jpg"
    async with async_playwright() as p:
        for browser_type in [p.firefox]:
            browser = await browser_type.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            try:
                await page.goto(url)
            except Exception as e:
                # if this url leads to a download, assume it's a PDF
                if "download" in e.message.lower():
                    return await pdf(url, img_path)
            await page.wait_for_load_state("load")
            # fucking animations
            await sleep(2)
            if click_ui:
                try:
                    async with timeout(5):
                         # if there's a button that says "order online", click it
                        for t in ["order online"]: # TODO add other tags?
                            element_to_click = page.get_by_text(t)
                            count = await element_to_click.count()
                            if count > 0:
                                async with context.expect_event("page") as event_info:
                                    await element_to_click.nth(count - 1).click()
                                await event_info.value
                                # if a new page was opened, switch to it
                                page = context.pages[-1]
                                break
                except TimeoutError:
                    print("click button timed out")

            await sleep(2)
            await page.screenshot(path=filename, full_page=True, type="png")
            await browser.close()

async def pdf(pdf_url, img_path):
    print(f"downloading pdf {pdf_url}")
    makedirs(temp_dir, exist_ok=True)
    # trust me, I'm a browser bro
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}) as session:
        async with session.get(pdf_url) as response:
            if "content-disposition" in response.headers:
                header = response.headers["content-disposition"]
                filename = header.split("filename=")[1]
            else:
                filename = pdf_url.split("/")[-1]
            op = f"{temp_dir}/{filename}"
            with open(op, mode="wb") as file:
                while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    file.write(chunk)
            print(f"Downloaded PDF file {filename}")
            images = convert_from_path(op)
            # convert PDF pages to jpeg images for, OCR step does not accept pdfs
            for idx, img in enumerate(images):
                img.save(f"{img_path}/{idx}.png", 'PNG')


async def run_query(url: str, click_ui: bool, paper_save: bool, menu_holder):
    # OPEN A WEB BASED MENU, SCREENSHOT IT
    start = time()
    url = check_drive_url(url)
    section_limit = 1 if paper_save else 30
    img_path = f"{img_dir}/{hash(url)}"
    makedirs(img_path, exist_ok=True)
    if url.endswith(".pdf"):
        await pdf(url, img_path)
    else:
        await web_page(url, img_path, click_ui)
    playwright_time = time()
    print(f"playwright: {playwright_time - start}")

    # SCRAPE THE TEXT OUT OF THE SCREENSHOT
    ocr_lines = []
    for img_file in sorted(listdir(img_path)):
        print(f"img_file: {img_file}")
        img = Image.open(f"{img_path}/{img_file}")
        ocr = pytesseract.image_to_string(img)
        ocr_lines.extend(ocr.splitlines())


    with open("out.txt", "w") as tf:
        tf.writelines(ocr_lines)

    # TODO do I still need to truncate this? OpenAI was yelling at me at some point
    text = '\n'.join(ocr_lines[:600])

    tess_time = time()
    print(f"tesseract: {tess_time - playwright_time}")

    # ASK CHATGPT TO MAKE SENSE OF THE SCRAPED MENU TEXT
    messages = [
        {
            "role" :  "user", 
            "content" : "Format the following menu as a CSV with semi-colons as seperators. The columns should be CATEGORY, ITEM, PRICE. Sort rows by CATEGORY. Return only valid ascii strings. \n\n" + text
        }
    ]

    valid_lines = 0
    line = ""
    category_index = 0
    reader = await openai.ChatCompletion.acreate(model="gpt-4", messages=messages, temperature=0, stream=True)

    # this nasty bit will parse the csv from ChatGPT and update a dict with the menu,
    # which will be served to API clients
    # menu holder format:
    # {
    #    "category name" : {
    #        "_index" : {category index number},
    #        "food item 1" : {price},
    #        ...
    #     },
    #     ...
    # }
    async for r in reader:
        if r.choices[0].delta.get("content") is None:
            continue
        line += r.choices[0].delta.content
        if (line.endswith('\n')):
            line = line[:-1]
            print(line)
            vals = line.split(";")
            # sometimes it ignores request to use semicolons, try commas
            if len(vals) != 3:
                vals = line.split(",")
                if len(vals) != 3:
                    print(f"invalid row: {line}")
                    line = ""
                    continue
            line = ""
            valid_lines += 1
            cat = unidecode(vals[0].title())
            if menu_holder.get(cat) is None:
                menu_holder[cat] = { "_index" : category_index }
                category_index += 1
            if len(menu_holder[cat]) < section_limit + 1:
                price = unidecode(vals[2].title())
                if not price:
                    price = "?"
                menu_holder[cat][unidecode(vals[1].title())] = price
    print(f"gpt time: {time() - tess_time}")
    return valid_lines

# run menu process with given url, in paper_save mode (or not)
# and inject results into menu_holder dict
async def run_main(url, paper_save, menu_holder):
    start = time()
    menu_map = {}
    # first try, allowing playwright to click buttons
    valid_lines = await run_query(url, True, paper_save, menu_holder)
    if valid_lines == 0:
        # if that doesn't work, try without pressing buttons
        valid_lines = await run_query(url, False, paper_save, menu_holder)
    menu_holder["completed"] = True
    return valid_lines