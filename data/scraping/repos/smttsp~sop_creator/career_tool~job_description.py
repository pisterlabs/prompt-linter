import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from career_tool.utils.file_utils import read_text_from_file
from career_tool.utils.http_utils import get_text_from_html
from career_tool.utils.string_utils import remove_extra_spaces
from career_tool.utils.word_utils import get_word_cloud, get_word_freq


class JobDescription:
    def __init__(self, jd_file, jd_link, jd_text):
        self.jd_file = jd_file
        self.jd_link = jd_link
        self.jd_text = jd_text
        # self.ori_content = self._get_jd_from_inputs()
        # self.content = self._remove_extra_wording_from_jd()

        self.content = self._get_jd_from_inputs()

        self.wc = get_word_cloud(self.content)
        self.wf = get_word_freq(self.wc)

        extracted_info = self._remove_extra_wording_from_jd_v2()
        self.clean_jd = extracted_info.get("clean_job_description", "")
        self.company = extracted_info.get("company", "")
        self.job_title = extracted_info.get("job_title", "")
        self.location = extracted_info.get("location", "")

    def convert_to_json_jd(self):
        pass

    def convert_to_pdf_jd(self, template_name):
        pass

    def _get_jd_from_inputs(self):
        """Process the JD (Job Description) based on the provided inputs.

        This function takes in the JD file, JD link, and JD text as inputs.
        It reads the text from the JD file if provided, otherwise, it tries to extract
        the text from the JD link if provided, and finally falls back to the JD text.
        The extracted or provided JD text is then processed to remove any extra spaces.

        Returns:
            str: Processed JD text with extra spaces removed.
        """

        jd = read_text_from_file(self.jd_file)
        jd = jd if jd is not None else get_text_from_html(self.jd_link)
        jd = jd if jd is not None else self.jd_text

        if isinstance(jd, str):
            jd = remove_extra_spaces(jd)

        return jd

    def _remove_extra_wording_from_jd_v2(self, llm_model="gpt-3.5-turbo"):
        """Remove the common wording from the JD, such as equal opportunity employer,
        non-discrimination, benefits etc.
        """

        chat = ChatOpenAI(temperature=0.0, model=llm_model)

        template_string = """Here is a {job_description}.
            Can you remove the common wording such as
            - equal opportunity employer
            - non-discrimination
            - benefits
            that are at the end of the from the job description?
            
            ---
            Along with that, can you also extract `job_title`, `company_name`, 
            `location` from the job description?
             
            Format the output as JSON with the following keys:
                clean_job_description
                job_title
                company_name
                location
        """

        prompt_template = ChatPromptTemplate.from_template(template_string)

        service_messages = prompt_template.format_messages(
            job_description=self.content
        )
        response = chat(service_messages)
        clean_content = json.loads(response.content)

        return clean_content
