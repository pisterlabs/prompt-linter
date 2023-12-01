import re
import openai
import requests
import time

from fastapi import HTTPException

from src.blog.image_controller import ImageController
from src.blog.schemas import OutputFormat
from src.config import Config

OPEN_API_KEY = Config.OPENAI_API_KEY
openai.api_key = OPEN_API_KEY


class BlogController:
    def __init__(
        self,
        title: str,
        keyword: str,
        title_and_headings: str,
        length: int,
        tone_of_voice: str,
        language: str,
        format: str,
        spellings_format: str,
        project_id: int,
        number_of_images: int,
        width_of_image: int,
        height_of_image: int,
        version: str,
    ) -> None:
        self.title = title
        self.keyword = keyword
        self.title_and_headings = title_and_headings
        self.spellings_format = spellings_format
        self.introduction = ""
        self.improved_title = ""
        self.outlines = ""
        self.body_paragraphs = ""
        self.conclusion = ""
        self.faq = []
        self.length = length
        self.tone_of_voice = tone_of_voice
        self.language = language
        self.format = format
        self.meta_description = ""
        self.project_id = project_id
        self.response = {}
        self.number_of_images = number_of_images
        self.main_image = ""
        self.headings_images = []
        self.headings = []
        self.prompt = ""
        self.width_of_image = width_of_image
        self.height_of_image = height_of_image
        self.version = version

    def call_openai(self, prompt, model="gpt-4", temperature=0.8):
        """
        It is used to call openai and get response from it
        """

        try:
            response = openai.ChatCompletion.create(
                model=model, messages=prompt, temperature=temperature
            )

            # Extract desired output from JSON object
            content = response["choices"][0]["message"]["content"]
        except:
            try:
                time.sleep(10)
                response = openai.ChatCompletion.create(model=model, messages=prompt)

                # Extract desired output from JSON object
                content = response["choices"][0]["message"]["content"]
            except:
                raise HTTPException(detail="GPT error", status_code=400)

        return content

    def improve_title(self):
        """
        It improves the title and makes it more appealing
        """

        messages = [
            {
                "role": "system",
                "content": "Act as a blog title generator and your task is to generate the final title. You would be either provided with title or keyword. If it is title then, improve it but if it is keyword then, simply paste it in the output without modifying it at all. Remember you only have to improve the title and you are not allowed to change the keyword even if it seems like a question. Do not make any changes in Keyword. Even if the keyword seems grammatically unconnected or incorrect then also, return it in the output without making any changes. Strictly make sure to copy the keyword as it is in the output without even changing the grammar or tense. Do not mention format or anything irrelevant in the output. There should be nothing other than the actual title in the output.\n\nCharacter count: Keep the generated title between 50 to 60 characters\n\nKeywords: It contains SEO friendly keywords\n\nLanguage: Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. Strictly follow the specified language to generate title, if given. Important Note: Strictly do not mention region in the title as the blog would be available worldwide. If headings are given then, use the headings while generating Title otherwise generate title normally without the reference of Headings. Do not enclose the title in quotation marks. Do not give any indication by writing 'Title:' or anything irrelevant like that before title. Lastly, take the language parameter into consideration, if given.",
            },
            {
                "role": "user",
                "content": f"Title: {self.title}\nKeyword: {self.keyword}\nHeadings: {self.outlines}\nLanguage: {self.language}\nEnglish and spellings Format: {self.spellings_format}",
            },
        ]

        improved_title = self.call_openai(messages).replace("Title: ", "").strip()
        h1_pattern = r"<h1>(.*?)</h1>"
        h1_replacement = r"\1"
        self.improved_title = re.sub(h1_pattern, h1_replacement, improved_title)

    def introduction_generation(self):
        """
        It is used to generate introduction
        """
        messages = [
            {
                "role": "system",
                "content": "Act as a content writer and your task is to write an introduction that meets following requirements:\nIt must be a small introduction that contains no more than 2 or 4 sentences. If needed, create paragraphs otherwise it's fine.\n\nHook: Begin with a compelling hook that resonates with readers.\n\nWord Count: Keep the whole introduction between 100 to 200 words, maintaining brevity while conveying the main idea effectively.\n\nProblem: Highlight the challenges related to title.\n\nSolution: Introduce the content as a comprehensive guide designed to provide practical strategies and insights for overcoming these challenges.\nKeywords: Ensure to use SEO friendly keywords.\n\nTransition: End the introduction with a smooth transition to the body content, promising valuable guidance to help readers navigate.\n\nLanguage: Strictly Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. Strictly follow the specified language to generate introduction, if given. Do not mention US or British English or any other type in the output. Important Note: Strictly do not mention any region in the introduction as the blog would be available worldwide. If outlines are there then, use those outlines to generate introduction and if not, then generate on your own without the reference of outlines. Lastly, take the tone of voice and language parameters into consideration, if given.",
            },
            {
                "role": "user",
                "content": f"Title: {self.improved_title}\nOutlines: {self.outlines}\nTone of Voice: {self.tone_of_voice}\nLanguage: {self.language}\nEnglish and spellings format: {self.spellings_format}\nIntroduction:\n",
            },
        ]

        self.introduction = self.call_openai(messages)

    def outline_generation(self):
        """
        It generates the outlines for the blog
        """
        outlines = 3 if self.length <= 600 else 5 if self.length <= 900 else 6 if self.length <= 1200 else 8 if self.length <= 1500 else 10
        messages = [
            {
                "role": "system",
                "content": "Act as a blog outline generator. Craft a concise, single-sentence numerical outline for the body paragraph using the provided title and introduction. Add suboutlines to some outlines only, it is not necessary for each outline to have sub outlines. Strictly do not include sub outlines for all the outlines. Number outlines as 1,2 and so on.. and sub outlines as a,b and so on... Do not insert full stop at the end of any outline or sub outline. Follow the number of outlines parameter and generate same number of outlines. Important Note: Strictly Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. When there are any steps to achieve something, then include the actual steps in the suboutline. Body paragraphs of each outline would be descriptive so number of outlines depend on the provided length of blog parameter. One outline can't contain more than 10 words. Focus solely on the content relevant to the body paragraph; strictly avoid generating outlines for the introduction and conclusion. Take the tone of voice and language parameters into consideration, if given. Strictly follow the specified language to generate outline, if given. Do not leave any space before suboutlines.\nOutput Format:\n1.\na.\n2.\na.\nso on..",
            },
            {
                "role": "user",
                "content": f"Title: {self.improved_title}\n\n{self.introduction}\nLength of the Blog: {self.length}\nTone of Voice: {self.tone_of_voice}\nLanguage: {self.language}\nNumber of Outlines: {outlines}\nEnglish and spellings format: {self.spellings_format}\nBody paragraph outlines:",
            },
        ]
        self.outlines = self.call_openai(messages)

    def meta_description_generation(self):
        """
        It generates a Meta Description with the help of Title, Introduction and Outlines
        """

        messages = [
            {
                "role": "system",
                "content": "Act as a blog meta description generator. Your task is to generate a small meta description using the given title, introduction, and outlines. Important Note: If keyword is provided then for the SEO purpose make sure that is used as it is in the Meta Description. Do not make any changes in Keyword. Even if the keyword seems grammatically unconnected or incorrect then also, copy it in the output without making any changes. Strictly make sure to copy the keyword as it is in the output without even changing the grammar or tense. The purpose of Meta Description is to entice the readers on Google search to click on the article.\nCharacter Count: The Meta Description must be strictly between 15-20 words, do not exceed the limit in any case.\nLanguage: Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. Strictly follow the specified language to generate meta description, if given. Do not give any indication by writing 'Meta Description:'",
            },
            {
                "role": "user",
                "content": f"Title: {self.improved_title}\nKeyword: {self.keyword}Introduction: {self.introduction}\nOutlines: {self.outlines}\nTone of Voice: {self.tone_of_voice}\nLanguage: {self.language}\nEnglish and spellings format: {self.spellings_format}\nMeta Description:",
            },
        ]

        self.meta_description = self.call_openai(messages).replace('"', '')

    def paragraph_generation(self):
        """
        It is used to generate body paragraphs, conclusion, frequently asked questions and meta description
        """

        messages = [
            {
                "role": "system",
                "content": "Act as a blog writer and your task is to generate body paragraph for each outline and sub outline. I am providing you with Title, Introduction and Outlines with sub outlines and your task is to generate body paragraph for each outline and sub outline. Generate a paragraph for the Outline first and then generate paragraph for sub outlines. Create separate paragraphs for sub outline. Provided outlines and sub outlines would have numbers but do not mention the numbers in the output. An outline can have multiple body paragraphs. When generating multiple paragraphs, do not give numbering to the paragraphs. When outline mentions steps to achieve a particular objective, then strictly create multiple paragraphs and explain each step in detail by strictly numbering each step as step-1, step-2 and so on.. only. If there are only outlines and no sub outlines, then generate multiple body paragraphs for each outline only. In this case, there is no need to generate body paragraphs for suboutlines or even there is no need to generate suboutlines. When sub outlines are absent, then do not write sub outlines on your own, just generate multiple body paragraphs for given the Outlines according to the given length. Whenever there is a short form or acronym used in the paragraph, be sure to write its full form in the bracket right next to it. Use simplified language. Do not generate random paragraphs. Generate according to the outlines. Do not insert full stop at the end of any outline or sub outline Make sure the content of paragraphs is related to the provided data. Use numbers, statistics and data as much as possible. Remember to Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. Do not include any fluff words and use active voice where possible. There should not be plagiarised content and it must pass AI detection. Write the respective outline as well as sub outline before each paragraph. Also take the length of the blog into account. Lastly, also generate a conclusion at the end. Make sure conclusion recaps the main points covered. The output must be in HTML format for outlines and paragraphs as shown in the output format. Do not insert any extra tags on your own. Take the tone of voice and language parameters into consideration, if given. Strictly follow the specified language to generate paragraph, if given. \nStrictly follow\nOutput Format:\n<h2>Input the actual Outline</h2>\nDescription on the main outline\n<h3>Input the actual Sub Outline</h3>\nDescription about that sub outline outline\n\n<h2>Conclusion</h2>",
            },
            {
                "role": "user",
                "content": f"Title: {self.improved_title}\n\nMaximum Length of the blog: {self.length} words\n\nEnglish and spellings format: {self.spellings_format}\n{self.introduction}\nOutlines: {self.outlines}\nTone of Voice: {self.tone_of_voice}\nLanguage: {self.language}\n",
            },
        ]

        return self.call_openai(messages)

    def generate_faqs(self, response):
        messages = [
            {
                "role": "system",
                "content": "Act as a frequently asked questions generator. Your task is to generate exactly 4 Frequently Asked Questions with the help of provided paragraphs and conclusion. Make sure that the answers of FAQs are at least 2 to 4 sentences long. Do not give numberings to the FAQs.\nLanguage: Follow the given English format. Word choices, spellings, and grammar should be according to the provided English conventions. Strictly follow the specified language to generate faqs, if given.\nOutput Format:\nFrequently Asked Questions:\nQ:\nA:",
            },
            {
                "role": "user",
                "content": f"Paragraphs and Conclusion: {response}\nFrequently Asked Questions:\nTone of Voice: {self.tone_of_voice}\nLanguage: {self.language}\nEnglish and spellings Format: {self.spellings_format}",
            },
        ]

        return self.call_openai(messages)

    def post_process(self, response, faqs):
        """
        It carries out the extraction of required fields like conclusion, outlines and paragraphs, and FAQs from the generated content
        """

        paragraphs = re.split(r"<h2>\s?Conclusion\:?\s?</h2>", response)

        self.body_paragraphs = paragraphs[0]

        if len(paragraphs) == 1:
            self.conclusion = response.split("\n")[-1]
        else:
            self.conclusion = paragraphs[1]
    

        self.faq = [
            {"Question": q.strip(), "Answer": a.strip()}
            for q, a in re.findall(
                r"Q:\s(.*?)(?=\nA:)\s+A:\s(.*?)(?=\n\nQ:|\Z)", faqs, re.DOTALL
            )
        ]

        if (
            self.body_paragraphs
            and self.conclusion
            and self.faq
            and self.meta_description
        ):
            return (
                self.body_paragraphs,
                self.conclusion,
                self.faq,
                self.meta_description,
            )
        else:
            raise HTTPException(
                detail="GPT generated insufficient content", status_code=400
            )

    def get_markdown_format(self):
        h2_pattern = r"<h2>(.*?)</h2>"
        h3_pattern = r"<h3>(.*?)</h3>"

        h2_replacement = r"## \1"
        h3_replacement = r"### \1"

        self.body_paragraphs = re.sub(
            h2_pattern, h2_replacement, self.body_paragraphs
        ).replace("<h2>", "")
        self.body_paragraphs = re.sub(h3_pattern, h3_replacement, self.body_paragraphs)
        self.conclusion = self.conclusion.replace("</h2>", " ")

    def get_simple_text_format(self):
        self.body_paragraphs = re.sub(r"<h2>", "", self.body_paragraphs)
        self.body_paragraphs = re.sub(r"</h2>", "\n", self.body_paragraphs)
        self.body_paragraphs = re.sub(r"<h3>", "", self.body_paragraphs)
        self.body_paragraphs = re.sub(r"</h3>", "\n", self.body_paragraphs)

        self.conclusion = self.conclusion.replace("</h2>", " ")

    def separate_title_and_headings(
        self,
    ):
        """
        It is used to separate out title and headings from the input
        """

        input_string = self.title_and_headings
        input_list = input_string.split(",")
        self.title = input_list[0]

        headings_list = input_list[1:]
        self.outlines = [f"{index + 1}. {heading}" for index, heading in enumerate(headings_list)]
        self.outlines = "\n".join(self.outlines)

    def send_response(self):
        if self.version == "test":
            response = requests.post(
                "https://rebecca-29449.bubbleapps.io/version-test/api/1.1/wf/article",
                json=self.response,
            )
        elif self.version == "live":
            response = requests.post(
                "https://rebecca-29449.bubbleapps.io/api/1.1/wf/article",
                json=self.response,
            )

        if response.status_code == 200:
            return {
                "message": "Blog generated successfully",
                "title": self.response["request_word"],
                "project_id": self.response["project_id"],
            }
        else:
            return {"message": "Error sending the blog"}

    def generate_images(self):
        self.headings = re.split(r"\d+\.\s", self.outlines)
        self.headings = [heading.strip() for heading in self.headings if heading.strip()][: self.number_of_images - 1]

        image_gen_obj = ImageController(
            self.improved_title,
            self.headings,
            self.width_of_image,
            self.height_of_image,
        )

        self.main_image, self.headings_images = image_gen_obj.generate_images()

    def generate_response(self):
        """
        It is used to generate the final response and gives the output in desired format
        """

        if not (self.title or self.keyword or self.title_and_headings):
            raise HTTPException(
                detail="Please provide any of Title or Keyword or Title and Headings.",
                status_code=400,
            )

        if self.title_and_headings:
            self.separate_title_and_headings()

        self.improve_title()
        self.introduction_generation()
        if not self.outlines:
            self.outline_generation()

        response = self.paragraph_generation()

        faqs = self.generate_faqs(response)

        self.meta_description_generation()
        self.post_process(response, faqs)

        if self.format == OutputFormat.markdown:
            self.get_markdown_format()

        elif self.format == OutputFormat.text:
            self.get_simple_text_format()

        if self.number_of_images > 0:
            self.generate_images()

        if (
            self.introduction
            and self.body_paragraphs
            and self.conclusion
            and self.faq
            and self.meta_description
        ):
            self.response = {
                "seo_title": self.improved_title.strip(),
                "content": f"{self.introduction.strip()}\n{self.body_paragraphs.strip()}\n{self.conclusion.strip()}",
                "frequently_asked_questions": self.faq,
                "meta_description": self.meta_description.strip(),
                "project_id": self.project_id,
                "request_word": self.title or self.keyword or self.title_and_headings,
                "heading_images_prompt": self.headings,
                "main_image_url": self.main_image,
                "header_images_url": self.headings_images,
            }
        else:
            self.response = {"error": "GPT generated insufficient content"}

        return self.send_response()
