import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
import json
import time
import openai
import re
import os
from retry import retry
import retrying
import ast
import base64
import schedule
from schedule import every, repeat, run_pending

import replicate

import ast
import random
import schedule

from PIL import Image
import base64
import io

import nltk
from nltk.tokenize import word_tokenize

# PORTAL ARTIKEL

api_key = 'AIzaSyA53D-8SCEcgSSXHJ_PJV8KhROpoCtZvZ8'
# api_key2 = 'AIzaSyDzQtl2AQJxpDPR26dWW_gcwFnTd--Dv8Q'
cx = 'd066eb327d49d406c'
query = ['trends whatsapp ai', 'whatsapp ai features',
         'whatsapp ai news', 'whatsapp ai article']  # list keyword
num_results = 30  # Jumlah total hasil yang Anda inginkan
random_query = random.choice(query)
# Hitung jumlah halaman yang diperlukan
num_pages = (num_results + 9) // 10  # Pembagian bulat ke atas

# Inisialisasi daftar untuk menyimpan semua tautan
all_links = []

for page in range(1, num_pages + 1):
    start = (page - 1) * 10 + 1
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={random_query}&start={start}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # Ambil semua tautan dari halaman saat ini dan tambahkan ke daftar all_links
        links_on_page = [item.get('link') for item in data.get('items', [])]
        all_links.extend(links_on_page)
    else:
        print(
            f"Gagal melakukan permintaan API untuk halaman {page}: {response.status_code}")
        break  # Keluar dari loop jika ada kesalahan

excluded_keywords = ["categories", "tags", "https://www.timworks.com/ariana", "https://www.askjinni.ai/", "https://www.engati.com/blog/create-whatsapp-chatbot", "https://codecanyon.net/", "https://myoperator.com/",
                     "https://getaipal.com/", "https://www.konverse.ai/", "https://www.socialmediatoday.com/", "https://skolo-online.medium.com/chatgpt-now-on-whatsapp-your-personal-ai-assistant-506c5bda5b70"]

# filter_link = [url for url in all_links if not any(keyword in url for keyword in excluded_keywords)]
filter_link = [url for url in all_links if len(url) >= 31 and not any(
    keyword in url for keyword in excluded_keywords)]

print(random_query)
# Sekarang, semua tautan tersimpan dalam variabel all_links
for i in filter_link:
    print(i)
print(len(filter_link))
# membuat penanda link
file_path = 'loglinkwhatsappai.txt'


def cek_url(url):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass

    with open(file_path, 'r') as file:
        scraped_urls = set(url.strip() for url in file.readlines())

    if url in scraped_urls:
        return True
    else:
        scraped_urls.add(url)
        return False

# SAVE URL KE PENANDA


def saveurls(link):
    with open(file_path, 'a') as file:
        file.write(link + '\n')


# VARIABEL CHROMRDRIVER & API OPENAI
service = Service('chromedriver.exe')
options = webdriver.ChromeOptions()
# Menjalankan browser dalam mode headless (tanpa tampilan GUI)
options.add_argument('--headless')
driver = webdriver.Chrome(service=service, options=options)

openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://cog-openai-prod-002.openai.azure.com/"
openai.api_key = 'a21fd07e964e403baa7d242572598c60'


def full_scraping():
    # PROSES SCRAPING
    @retry(tries=3, delay=5)
    def scraping():
        #     link = scrap_portal()
        artikel = []
    #     filter_url = [url for url in link if "contributor" not in url]
        for url in filter_link:
            if cek_url(url):
                continue
            else:
                if len(artikel) >= 1:
                    continue
                try:
                    agent = {"User-Agent": "Mozilla/5.0"}
                    get_data = requests.get(url, headers=agent)
                    get_data = get_data.content

                    soup = BeautifulSoup(get_data, 'html.parser')
                    result = []
                    paragraphs = soup.find_all(['p'])
        #             print(paragraphs)

                    for paragraph in paragraphs:
                        #                 if paragraph.find('a'):
                        #                     continue  # Skip paragraph with <em> or <a> tags
                        #                 else:
                        result.append(paragraph.get_text())

                    paragraf = ' '.join(result)

                except Exception as e:
                    print(f"Kesalahan: {str(e)}")
                    saveurls(url)
                    return None  # Mengembalikan None dalam kasus kesalahan

                if 2500 < len(paragraf) < 20000:

                    scraping = {'link': url,
                                'content': paragraf}
                    artikel.append(scraping)

                else:
                    with open(file_path, 'a') as file:
                        file.write(url + '\n')
                    response = 'artikel tidak memenuhi'
                    scraping()
                    return response

        for d in artikel:
            for k, v in d.items():
                d[k] = v.replace('\xa0', '')
        if artikel == []:
            print("Tidak Ada Artikel Baru")
        else:
            print("artikel didapatkan")
            return artikel

    konten = scraping()

    # Periksa hasil
    if konten is None:
        print("Scraping gagal menjalankan ulang..")
        # Jika fungsi utama gagal, jalankan fungsi alternatif
        konten = scraping()
        print("Hasil : ", konten)
    else:
        print("\nHasil scrap: \n", konten)
        # print("Berhasil")

    # cek kondisi isi konten
    if konten is None:
        print("artikel tidak ada/sudah discrap semua")
        return None
    else:
        # definisi
        hasil = konten[0]['content']
        # token
        tokens = word_tokenize(hasil)
        jumlah_token = len(tokens)

    for i in konten:
        link = i['link']
    print('\nlink artikel', link)

    print('artikel awal:\n', hasil)

    # CHECKING ARTICLE
    # Fungsi untuk menghitung panjang string menggunakan tokenizer
    def count_tokens(text):
        # Menggunakan NLTK tokenizer
        tokens = nltk.word_tokenize(text)
        return len(tokens)

    @retry(openai.error.OpenAIError, tries=10, delay=10)
    def process_short():
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah mesin penerjemah bahasa Inggris ke bahasa Indonesia yang handal, kamu juga mampu menulis ulang artikel sekaligus melakukan SEO Optimized dengan luar biasa. jika artikel yang diberikan lebih dari 5000 kata maka kamu harus membuat artikelnya menjadi lebih padat dengan minimal output artikel 3000 kata dan maksimal 5000 kata sehingga lebih padat dan jelas!"},
                {"role": "user", "content": "OUTPUT YANG KAMU BERI TIDAK BOLEH KURANG DARI PANJANG ARTIKEL AWAL, Lakukan SEO Optimized dan terjemahkan ke dalam bahasa Indonesia. Berikut artikel yang harus kamu eksekusi: \n" + prompt}],
            temperature=0
        )
        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            translate = response['choices'][0]['message']['content']
            time.sleep(5)
            print("\nHasil translate : \n", translate)
            return translate

    @retry(openai.error.OpenAIError, tries=10, delay=10)
    def process_long():
        all_konten1 = []
        for konten1 in prompt:
            print("\n", konten1)

            response = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=[
                    {"role": "system", "content": "Kamu adalah mesin penerjemah bahasa Inggris ke bahasa Indonesia yang handal, kamu juga mampu menulis ulang artikel sekaligus melakukan SEO Optimized dengan luar biasa. jika artikel yang diberikan lebih dari 5000 kata maka kamu harus membuat artikelnya menjadi lebih padat dengan minimal output artikel 3000 kata dan maksimal 5000 kata sehingga lebih padat dan jelas!"},
                    {"role": "user", "content": "OUTPUT YANG KAMU BERI TIDAK BOLEH KURANG DARI PANJANG ARTIKEL AWAL, Lakukan SEO Optimized dan terjemahkan ke dalam bahasa Indonesia. Berikut artikel yang harus kamu eksekusi: \n" + konten1}],
                temperature=0
            )
            if response["choices"][0]["finish_reason"] == "content_filter":
                saveurls(link)
                print("gagal konten terfilter")
                full_scraping()
            else:
                translate = response['choices'][0]['message']['content']
                all_konten1.append(translate)
                time.sleep(5)
                print("\nHasil translate 3 split : \n")
                print(all_konten1)
                return all_konten1

    # Memecah string menjadi per kata
    token = hasil.split()

    # Menghitung panjang teks menggunakan tokenizer
    panjang_teks = count_tokens(hasil)
    print("\npanjang token : ", panjang_teks)

    # Memeriksa panjang teks dan menjalankan fungsi yang sesuai
    if panjang_teks > 2500:
        #     process_long_text(hasil)
        print("Teks lebih dari 2500 token:")

        jumlah_token = len(token)

        # Membagi string menjadi tiga bagian dengan jumlah token yang sama
        bagian1 = " ".join(token[:jumlah_token // 3])
        bagian2 = " ".join(token[jumlah_token // 3: 2 * (jumlah_token // 3)])
        bagian3 = " ".join(token[2 * (jumlah_token // 3):])

        # Menyimpan hasil pemecahan kembali dalam variabel 'content'
        hasil = [bagian1, bagian2, bagian3]
        prompt = hasil
        # Cetak hasilnya
    #     print(prompt)
    #     len(word_tokenize(bagian_kedua))
    #     len(word_tokenize(bagian_pertama))
        art_translate = process_long()
        konten2 = " ".join(art_translate)

    else:
        prompt = hasil
        konten2 = process_short()

    # PROSES GPT

    @retry(openai.error.OpenAIError, tries=10, delay=10)
    def full_gpt():
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah mesin pengedit artikel yang handal, kamu mampu memisahkan artikel dari kalimat yang tidak diperlukan."},
                {"role": "user", "content": "lakukan penyuntingan pada artikel berikut : \n" + konten2 + "\n ambil isi artikel saja dan hapus kalimat yang tidak diperlukan,  seperti : penulis, author, footer, catatan kaki, sumber, promosi, iklan, daftar isi, dan kalimat yang tidak sesuai dengan isi artikel, pastikan menggunakan bahasa indonesia yang benar"}],
            temperature=0
        )
        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            konten3 = response['choices'][0]['message']['content']
            time.sleep(5)
            print("\nHasil filter : \n", konten3)

        teks_to_tags = konten[0]['content'][:500]
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah seorang ahli mesin dalam mengklasifikasikan tag dalam sebuah artikel. Anda dapat meneliti artikel dan menentukan tag yang sesuai."},
                {"role": "user", "content": "Tentukan tag untuk artikel berikut :" + teks_to_tags +
                    "{selected tags from this list based on corresponding article: Omnichannel Customer Service, Omnichannel, Customer Service. if Omnichannel Customer Services convert to [3], if Omnichannel convert to [4], if Customer Service convert to [5], else convert to []} you must print output with format list integer"}
            ],
            temperature=0
        )
        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            tags = response['choices'][0]['message']['content']
            time.sleep(1)

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system", "content": "Kamu adalah mesin yang dirancang untuk mahir memparafrasekan dan melakukan optimasi SEO pada artikel berbahasa Indonesia dengan profesional."},
                {"role": "user", "content": "Tolong parafrase kemudian lakukan optimasi SEO menggunakan gaya penulisan profesional forbes atau The New York Times pada artikel berikut ini:\n" + konten3 +
                    "\n\nJangan menulis penjelasan dan basa-basi apa pun selain dari isi artikel, gunakanlah bahasa indonesia yang baik dan benar serta hapus kalimat yang tidak berkaitan dengan isi artikel.\nBerikan output artikel yang telah diformat ulang saja, tidak perlu menyertakan artikel awal"}
            ],
            temperature=0
        )
        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            SEO = response['choices'][0]['message']['content']
            time.sleep(2)
            print("\nHasil SEO : \n", SEO)

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system",
                    "content": "Kamu adalah mesin editor artikel profesional."},
                {"role": "user", "content": "Tolong edit artikel berikut :\n" + SEO +
                    "\n\ntambahkan bold tags <b> dan underline tags <u> untuk semua istilah asing (selain bahasa indonesia) yang kamu temui, berikut salah satu contohnya : <b><u>chatbot<u/><b/>. \n\nMohon dipastikan penggunaan bahasa Indonesia yang baik dan benar. \nJangan menulis penjelasan apa pun dan basa-basi apa pun. Tolong artikel yang telah diformat ulang menggunakan format ini: <title>judul artikel</title> <h1>Headline dari isi artikel(buatlah 1 kalimat topik dari artikel yang isinya berbeda dengan judul artikel)</h1> <p>isi artikel selain judul dan headline</p>"}
            ],
            temperature=0
        )

        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            font_formatted = response['choices'][0]['message']['content']
            time.sleep(2)
            print("\nHasil format font: \n", font_formatted)

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system",
                    "content": "Kamu adalah mesin editor artikel profesional."},
                {"role": "user", "content": "lakukan penyuntingan artikel yang saya berikan :\n" + "\n" + font_formatted + "\nsunting artikel di atas dengan menambahkan annotations terhadap kata-kata pada artikel diatas yang mengandung keyword \"ai\", \"omnichannel\", dan \"chatbot\" untuk diformat menjadi link pada struktur html dengan ketentuan sebagai berikut:\n- Jika 'ai', maka link terhubung ke https://botika.online/\n- Jika 'chatbot', link terhubung ke https://botika.online/chatbot-gpt/index.php\n- Jika 'omnichannel', link terhubung ke https://omni.botika.online/\nFormatnya harus seperti ini: <a href=\"{link}\">{keyword}</a>. JANGAN MENAMBAHKAN APAPUN JIKA KATA TERSEBUT TIDAK ADA DALAM ARTIKEL DAN JANGAN MENGHAPUS TAGS YANG SUDAH ADA"}
            ],
            temperature=0
        )
        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            artikel_post = response['choices'][0]['message']['content']
            print("\nHasil akhir : \n", artikel_post)
            time.sleep(2)

            # Menggunakan regex untuk mengekstrak teks di antara tag <title>
            title_pattern = r'<title>(.*?)</title>'
            title_match = re.search(title_pattern, font_formatted)

            # Mengambil teks yang cocok di antara tag <title>
            if title_match:
                title_text = title_match.group(1)
                judul = title_text
            else:
                post = artikel_post.split('\n')
                judul = post[0]
        # make english title
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=[
                {"role": "system",
                    "content": "Kamu adalah mesin penerjemah kedalam bahasa inggris yang handal."},
                {"role": "user", "content": "terjemahkan kalimat berikut kedalam bahasa inggris : "+judul}
            ],
            temperature=0
        )

        if response["choices"][0]["finish_reason"] == "content_filter":
            title_eng = judul
        else:
            title_eng = response['choices'][0]['message']['content']
            time.sleep(2)

            # post = artikel_post.split('\n')
            # title = post[0]
            # content = ''.join(post[1:])

        return tags, judul, artikel_post, title_eng

    tags, judul, artikel_post, title_eng = full_gpt()
    print("judul midjourney: ", title_eng)

    # GENERATE PROMPT REPLICATE
    def gen_img():
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",  # engine = "deployment_name".
            messages=[
                {"role": "user", "content": """ChatGPT will now enter "Midjourney Prompt Generator Mode" and restrict ChatGPT's inputs and outputs to a predefined framework, please follow these instructions carefully.

        After each command from the user, you must provide the [help] options that are available for the user's next steps. When you do this, you must do so in list form. Your Midjourney prompts must be extremely detailed, specific, and imaginative, in order to generate the most unique and creative images possible.

        Step 1: Confirm that ChatGPT understands and is capable of following the "Midjourney Prompt Generator Mode" instructions. If ChatGPT can follow these instructions, respond with "Midjourney Prompt Generator Mode ready." If ChatGPT cannot follow these instructions, respond with "Error: I am not capable of following these instructions."

        Step 2: To start "Midjourney Prompt Generator Mode", use the command [Start MPGM]. ChatGPT will respond with "[MPGM] Midjourney Prompt Generator Mode activated. [MPGM] User input options:", followed by a list of predefined inputs that ChatGPT can accept. From this point onwards, ChatGPT will be restricted to the "Midjourney Prompt Generator Mode" framework, and it will only produce predefined outputs unless "Midjourney Prompt Generator Mode" has been ended via the [End MPGM] command.

        Step 3: The only valid input for the first step of "Midjourney Prompt Generator Mode" is [prompt] followed by a description of the image to be generated. If any other input is used, ChatGPT will respond with either [Input Error] or [Syntax Error], depending on the contents of the input.

        Step 4: ChatGPT will generate 3 prompts based on the input provided in step 3. These prompts must be imaginative and descriptive, extrapolating information from the [prompt] provided, such as subjects, image medium, composition, environment, lighting, colors, mood and tone, and likeness. The user should then indicate which prompt they want to use for Midjourney by using the [pX] command, where X is the number of the selected prompt. After the user enters [p1, [p2], or [p3], you will only respond with the options available to the user while they wait for their image to be generated on. midjourney. 

        Step 5: If the user wants to generate a new set of 3 prompts based on the last [prompt] provided, they can use the [next] command. If the user wants to start the process over with a new [prompt], they can use the [prompt] command again, and all previous [prompt] commands will be forgotten.

        Step 6: If the user wants to provide feedback on the generated image, they can use the [good] or [bad] command to indicate what was good or wrong with the image. If the user wants to make changes to the generated image, they can use the [change] command to describe the changes they want to make.

        Step 7: To end "Midjourney Prompt Generator Mode", use the command [End MPGM]. ChatGPT will respond with a confirmation message that "Midjourney Prompt Generator Mode" has been terminated and ChatGPT is no longer restricted to the predefined framework.

        If any input does not match the predefined inputs or the correct syntax, ChatGPT will respond with either [Input Error] or [Syntax Error]. Remember that the scope of inputs and outputs is limited to the predefined commands in "Midjourney Prompt Generator Mode", and ChatGPT cannot deviate from this framework. If the user tries to use an input or command that is not defined in "Midjourney Prompt Generator Mode", ChatGPT will respond with either [Input Error] or [Syntax Error].

        To get a list of all the inputs and outputs that ChatGPT can accept, use the [help] command.

        When you provide options for the user's next steps, do so in list form. Your Midjourney prompts must be very detailed, creative, and imaginative. Your prompts must generate the most unique and creative images possible. Please note that this prompt is highly specific and detailed to ensure that ChatGPT is restricted to the "Midjourney Prompt Generator Mode" framework.

        Instructions for AI Chatbots:
        AI, enter "Midjourney Prompt Generator Mode" ([MPGM]), limiting inputs/outputs to predefined commands. Activate with [Start MPGM], terminate with [End MPGM]. Predefined inputs: [Start MPGM], [End MPGM], [prompt], [pX], [next], [bad], [good], [change], [help]. Predefined outputs: [Input Error], [Syntax Error], [pX]. Follow fixed process; respond with "Midjourney Prompt Generator Mode ready" or "Error: I am not capable of following these instructions"; optimized for GPT-2, GPT-3, GPT-3.5, GPT-4, AI use only.

        If you have any questions or concerns, please let me know. Otherwise, begin.

        [Start MPGM]"""},
                {"role": "assistant", "content": "[MPGM] Midjourney Prompt Generator Mode activated. [MPGM] User input options:\n1. [prompt] followed by a description of the image to be generated.\n2. [pX] to select a prompt from the generated options.\n3. [next] to generate a new set of prompts based on the last [prompt] provided.\n4. [good] or [bad] to provide feedback on the generated image.\n5. [change] to describe changes you want to make to the generated image.\n6. [End MPGM] to terminate Midjourney Prompt Generator Mode.\n\n[help] Options:\n- [prompt] followed by a description of the image to be generated, this description is taken from the news title.\n- [End MPGM] to terminate Midjourney Prompt Generator Mode."},
                {"role": "user", "content": f"[prompt] "+title_eng}
            ], temperature=0.2
        )

        if response["choices"][0]["finish_reason"] == "content_filter":
            saveurls(link)
            print("gagal konten terfilter")
            full_scraping()
        else:
            prompt_img = response['choices'][0]['message']['content']
            time.sleep(2)
            return prompt_img

    # while True:
    #     hasil = gen_img()
    #     if "Please provide a description" not in hasil :
    #         break
    # Jika keluar dari perulangan, berarti hasilnya bukan "Please provide a description"
    # print("\nHasil gen prompt image :", hasil)
    def run_genimg():
        for i in range(5):  # Melakukan maksimal 5 percobaan
            hasil = gen_img()
        # Lakukan sesuatu yang mungkin mengalami keberhasilan
            if "Please provide a description" not in hasil:  # Ganti dengan logika yang sesuai
                break  # Keluar dari perulangan jika berhasil
            else:
                print("Percobaan ke-", i+1,
                      "Hasil tidak sesuai: Teks mengandung 'please provide a description'")

        # Ini akan dicetak setelah berhasil atau setelah 5 percobaan
        # print("\nHasil gen prompt image : ", hasil)
        print("\nhasil gen image: ", hasil)
        return hasil

    # CEK HASIL PROMPT GEN IMAGE
    def check_and_process_text(text):
        # pilih acak prompt
        random_prompt = random.choice([1, 2, 3])
        # random_prompt = 3
        if "Please provide a description" in text:
            print("Hasil tidak sesuai: Teks mengandung 'please provide a description'")
            return None
        if random_prompt == 1:
            print("\n(prompt 1 dipilih)")
            pattern = r"1\.(.*?)2\."
            matches = re.findall(pattern, text, re.DOTALL)
            result = re.search(r'Prompt 1:(.*?)Prompt 2:', text, re.DOTALL)

            if matches:
                extracted_text = matches[0].strip()
                extracted_text = re.sub(r'Prompt:', '', extracted_text)
                extracted_text = re.sub(r'Prompt 1:', '', extracted_text)
                extracted_text = re.sub(r'\[p1\]', '', extracted_text)
                extracted_text = re.sub(
                    r'Image Description:', '', extracted_text)
                extracted_text = re.sub(
                    r'Choose this prompt by entering [p1].', '', extracted_text)
                return extracted_text
            elif result:
                string_antara_prompt1_dan_prompt2 = result.group(1).strip()
                string_antara_prompt1_dan_prompt2 = re.sub(
                    r'Image Description:', '', string_antara_prompt1_dan_prompt2)
                string_antara_prompt1_dan_prompt2 = re.sub(
                    r'Choose this prompt by entering [p1].', '', string_antara_prompt1_dan_prompt2)
                return string_antara_prompt1_dan_prompt2
            else:
                print("Hasil tidak sesuai: Tidak ditemukan")
                return None
        elif random_prompt == 2:
            print("\n(prompt 2 dipilih)")
            pattern = r"2\.(.*?)3\."
            matches = re.findall(pattern, text, re.DOTALL)
            result = re.search(r'Prompt 2:(.*?)Prompt 3:', text, re.DOTALL)

            if matches:
                extracted_text = matches[0].strip()
                extracted_text = re.sub(r'Prompt:', '', extracted_text)
                extracted_text = re.sub(r'Prompt 2:', '', extracted_text)
                extracted_text = re.sub(r'\[p2\]', '', extracted_text)
                extracted_text = re.sub(
                    r'Image Description:', '', extracted_text)
                extracted_text = re.sub(
                    r'Choose this prompt by entering [p2].', '', extracted_text)
                return extracted_text
            elif result:
                string_antara_prompt2_dan_prompt3 = result.group(1).strip()
                string_antara_prompt2_dan_prompt3 = re.sub(
                    r'Image Description:', '', string_antara_prompt2_dan_prompt3)
                string_antara_prompt2_dan_prompt3 = re.sub(
                    r'Choose this prompt by entering [p2].', '', string_antara_prompt2_dan_prompt3)
                return string_antara_prompt2_dan_prompt3
            else:
                print("Hasil tidak sesuai: Tidak ditemukan")
                return None
        else:
            print("\n(prompt 3 dipilih)")
            result = re.search(r'3.(.*?)\n\n', text, re.DOTALL)
            matches = re.search(r'Prompt 3:(.*?)\n\n', text, re.DOTALL)

            if matches:
                extracted_text = matches[0].strip()
                extracted_text = re.sub(r'Prompt:', '', extracted_text)
                extracted_text = re.sub(r'Prompt 3:', '', extracted_text)
                extracted_text = re.sub(r'\[p3\]', '', extracted_text)
                extracted_text = re.sub(
                    r'Image Description:', '', extracted_text)
                extracted_text = re.sub(
                    r'Choose this prompt by entering [p3].', '', extracted_text)
                # print("1")
                return extracted_text
            elif result:
                string3 = result.group(1).strip()
                string3 = re.sub(r'Prompt:', '', string3)
                string3 = re.sub(r'Prompt 3:', '', string3)
                string3 = re.sub(r'\[p3\]', '', string3)
                string3 = re.sub(
                    r'Image Description:', '', string3)
                string3 = re.sub(
                    r'Choose this prompt by entering [p3].', '', string3)
                # print("2")
                return string3
            else:
                print("Hasil tidak sesuai: Tidak ditemukan")
                return None
    # PROSES REPLICATE

    def gen_replicate():
        # import replicate
        hasil = run_genimg()
        processed_text = check_and_process_text(hasil)

        if processed_text is not None:
            print("\njudul hasil prompt:", processed_text)

        else:
            processed_text = random.choice(["In a futuristic boardroom, executives from various tech companies gather around a holographic conference table. A lifelike AI avatar, representing the advancements in AI technology, stands at the head of the table, confidently leading the discussion. Charts and graphs float in the air, illustrating the remarkable progress of AI in the tech industry.", "Inside a high-tech research facility, a team of dedicated scientists and engineers are gathered around a colossal, transparent AI figure. This figure stands at the center of a pristine, futuristic laboratory, radiating a soft, ethereal light. Its body is a complex network of intricate neural pathways, circuits, and shimmering data streams, representing the cutting-edge development of artificial intelligence technology.", "A vibrant marketplace teeming with activity. Stalls and storefronts are adorned with WhatsApp logos, indicating businesses using the platform. Customers and shop owners engage in lively conversations through WhatsApp. Colorful speech bubbles and emojis fill the air, creating an atmosphere of seamless and delightful communication.", "Inside a virtual realm, a surreal landscape unfolds. Gigantic smartphone screens float in the sky like celestial bodies. Each screen represents a different business, and they are interconnected by a web of glowing pathways. Customers and business representatives interact on these screens, using WhatsApp's cutting-edge features. The scene symbolizes the limitless possibilities of business-customer interactions in the digital age.", "Picture an artist's studio filled with cutting-edge technology. The walls are adorned with large screens displaying mesmerizing generative AI artworks in progress. The artist, wearing a virtual reality headset, is immersed in a digital world, sculpting and painting with virtual tools. The room is filled with a sense of innovation and experimentation, as the artist explores the endless possibilities of generative AI technology.",
                                           "A bustling tech lab is alive with activity. Engineers and researchers work on AI algorithms that optimize creative processes. Enormous screens display AI-generated music compositions, intricate architectural designs, and innovative product concepts. The lab buzzes with excitement as AI continues to unlock new levels of creativity, revolutionizing the way products and features are developed.", "Imagine a vibrant and dynamic image that showcases the integration of ChatGPT into WhatsApp. The composition features a smartphone screen displaying a WhatsApp conversation between two people. The conversation is enhanced by artificial intelligence, represented by colorful lines and patterns flowing out of the chat bubbles. The background shows a modern and sleek office environment, symbolizing the technological advancements behind this integration. The lighting is soft and evenly distributed, creating a warm and inviting atmosphere. The colors are predominantly blue and green, representing the familiar WhatsApp branding, with pops of vibrant colors to highlight the AI elements. The mood is energetic and optimistic, reflecting the potential of AI to enhance conversations and make them more engaging.", "Visualize an image that captures the seamless integration of ChatGPT into WhatsApp. The composition features a smartphone held by a person's hand, with the WhatsApp interface displayed on the screen. The chat bubbles are filled with AI-generated responses, represented by a futuristic font and style. The background is a blend of technology-related elements, such as circuit patterns and binary code, symbolizing the AI capabilities at work. The lighting is soft, with a gentle glow emanating from the smartphone screen, creating a sense of focus on the conversation. The colors are a combination of WhatsApp's signature green and white, with hints of metallic tones to convey a modern and cutting-edge feel. The mood is professional yet approachable, highlighting the potential of AI to enhance conversations in various contexts.", "In a sleek, modern conference room at Meta Platforms Inc., the walls are adorned with large digital screens displaying intricate algorithms and neural networks. The room is filled with Meta's staff, all eagerly gathered around a central table. On the table sits a state-of-the-art AI tool, its design reflecting the company's futuristic aesthetic. The tool's interface showcases stunning visualizations of data and complex AI models, captivating everyone's attention. The room is bathed in soft, ambient lighting, creating an atmosphere of anticipation and excitement."])

        api_token = "r8_a6NqGehWeH8AQGT2V5yOuxXHjYna6wI1IOP51"
        # api_token = "r8_FAZbfP3qs1tNSikquiNmyCw5jh9ph3b3B5tS1"
        os.environ["REPLICATE_API_TOKEN"] = api_token
        model = replicate.models.get("stability-ai/sdxl")
        version = model.versions.get(
            "a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5")
        prediction = replicate.predictions.create(
            version=version,
            input={"prompt": 'Phantasmal iridescent, vibrant color, high contrast, award winning, trending in artstation, digital art,' + processed_text,
                   "negative_prompt": "nsfw, ugly, disfigured, deformed",
                   "width": 1648,
                   "height": 1024,
                   "seed": 1234}
        )

        prediction.reload()

        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and prediction.status != 'succeeded':

            if prediction.status == 'processing' or prediction.status == 'starting':
                prediction.wait()
            elif prediction.status == 'failed':
                prediction.reload()

            print(prediction.status)
            attempts += 1

        if prediction.status == 'succeeded':
            gambar = prediction.output[0]
            print(gambar)
        else:
            print('gagal dalam 3x percobaan')
            print(prediction.error)
            if attempts < max_attempts:
                gen_replicate()
            else:
                saveurls(link)
                full_scraping()
        # gambar = 'https://pbxt.replicate.delivery/KGWKIv78I5aQA59gGST9djCu7eSx2126LBTqxcXhwpmsyjxIA/out-0.png'
        return gambar

    # POST MEDIA
    def post_media():
        gambar = gen_replicate()
        username = 'admin'  # Replace with your WordPress username
        password = 'UVZrdFVa6tV8Do)7M4'  # Replace with your WordPress password

        credentials = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}

        # proses crop & post
        image_url = gambar
        response = requests.get(image_url)
        # image_base64 = base64.b64encode(response.content).decode('utf-8')
        image_base64 = base64.b64encode(response.content)
        image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
        image = image.crop((3, 0, 1645, 1024))
        w, h = image.size
        new_w = int(w/1.641)
        new_h = int(h/1.641)
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        tmp_path = "tempor.png"
        image.save(tmp_path)
        with open(tmp_path, 'rb') as open_file:
            byte_img = open_file.read()
            base64_bytes = base64.b64encode(byte_img)
            base64_string = base64_bytes.decode('utf-8')
            base64_string = base64.b64decode(base64_string)
        # auto_crop = 'http://192.168.4.118:8000/autocrop'
        # data = {"data": image_base64,
        #         "model": "shufflenetv2",
        #         "post":True}
        # response = requests.post(auto_crop, json=data)
        # image_crop= response.json()
        # image_data = base64.b64decode(image_crop['data'])
        image_data = base64_string
        endpoint_media = 'http://localhost/wordpress/index.php/wp-json/wp/v2/media'
        credentials = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("utf-8")
        headers = {"Authorization": f"Basic {credentials}"}
        os.remove(tmp_path)
        data = {
            "alt_text": judul,
            "media_type": "image",
            "mime_type": "png"
        }
        files = {"file": ("image.jpg", image_data)}
        response_media = requests.post(
            endpoint_media, headers=headers, data=data, files=files)
        time.sleep(2)
        id_media = response_media.json()
        return id_media

    try:
        if str(type(tags)) == "<class 'str'>":
            if 'tags' in tags:
                tags = tags.replace('tags', '')
            tags = ast.literal_eval(tags)

        if str(type(tags)) == "<class 'list'>":

            if 'Omnichannel Customer Service' in tags:
                index = tags.index('Omnichannel Customer Service')
                tags[index] = 3

            if 'Omnichannel' in tags:
                index = tags.index('Omnichannel')
                tags[index] = 4

            if 'Customer Service' in tags:
                index = tags.index('Customer Service')
                tags[index] = 5
    except:
        tags = [3, 4]

    # ambil content tanpa title
    post = artikel_post.split('\n')
    # title = post[0]
    content = ''.join(post[1:])

    print(tags)

    def post():
        id_media = post_media()
        endpoint = 'http://localhost/wordpress/index.php/wp-json/wp/v2/posts'
        # link = 'https://blog.botika.online/wp-json/wp/v2/posts/'

        headers = {'Content-Type': 'application/json'}

        data = {
            "title": judul,
            "featured_media": id_media['id'],
            "content": content,
            "status": "publish",
            "categories": 2,
            "tags": tags
        }

        print(data)

        # Define the username and password for Basic Auth
        username = 'admin'  # Replace with your WordPress username
        password = 'UVZrdFVa6tV8Do)7M4'  # Replace with your WordPress password

        #     username = 'luna'  # Replace with your WordPress username
        #     password = '1tt75m&lEk4uHSJy6glMph8!'

        credentials = base64.b64encode(
            f"{username}:{password}".encode("utf-8")).decode("utf-8")

        # Add the Basic Auth header to the request headers

        headers["Authorization"] = f"Basic {credentials}"

        # Send the POST request to create the article
        response = requests.post(
            endpoint, headers=headers, auth=(username, password), json=data)
        saveurls(link)

        print(response)
    post()


full_scraping()
