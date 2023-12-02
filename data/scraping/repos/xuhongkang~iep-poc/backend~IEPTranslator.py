import fitz, base64, requests
from openai import OpenAI
from io import BufferedReader, BytesIO

class IEPTranslator:
    def __init__(self, client:OpenAI, api_key: str) -> None:
        self.client = client
        self.api_key = api_key
        self.pdf = None
        self.full_translation = ''
    
    def add_iep(self, iep: BytesIO):
        self.pdf = fitz.open(stream=iep, filetype="pdf")
    
    def get_total_page_num(self) -> int:
        return len(self.pdf)

    def _image_to_text(self, image):
        print('19')
        base64_image = base64.b64encode(image).decode('utf-8')
        print('21')
        headers = {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": "gpt-4-vision-preview",
                "messages": [{"role": "user","content": [
                    {"type": "text",
                    "text": "This is a page from a student's (possibly redacted) Individualized Education Plan. Try to provide a text transcription of the image while preserving its structure and logic but excluding any redacted information. Use the 'the student' equivalent in the translation to refer to the student."},
                    {"type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                "max_tokens": 300}
        print('32')
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json().get('choices')[0].get('message').get('content')

    def _translate_text(self, text: str, language: str='Spanish'):
        return self._get_chatgpt_response('Please Translate the following text to '+ language +' without loss of information:', text)

    def _summarize_text(self, text: str, language: str='Spanish'):
        return self._get_chatgpt_response('Please Summarize the Following IEP Breakdown in ' + language, text)

    def _get_chatgpt_response(self, prompt:str, text: str) -> str:
        response = self.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": text}])
        return response.choices[0].message.content

    def get_page_translation(self, page_num:int, language:str='Spanish', image_format:str='jpeg'):
        # Get the page
        page = self.pdf[page_num]
        print('50')
        # Render page to an image
        pix = page.get_pixmap()
        text = self._image_to_text(pix.tobytes(image_format))
        print('61')
        translated_text = self._translate_text(text,language)
        self.full_translation += f"Page {page_num + 1}:\n" + translated_text
        return translated_text
    
    def get_summary(self, language:str='Spanish'):
        return self._summarize_text(self.full_translation, language)
    
    # Close Method Needed
