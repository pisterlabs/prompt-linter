import unittest
import json
import os
from PsyProfile_HumanticAI import *
from ResumeParser_OpenAI import *

class TestPsyProfile(unittest.TestCase):

    def test_create_user_profile(self):
        file_path = "./Materials/SOP.pdf"
        user_id = "unittest"
        API_KEY = "chrexec_96effa3d5b5f3c0de52193e04e91e087"
        statu_code, text = create_user_profile(API_KEY, user_id, file_path)
        self.assertEqual(statu_code, 200, 'Cannot create user profile.')
    
    def test_modify_user_persona(self):
        user_id = "unittest"
        API_KEY = "chrexec_96effa3d5b5f3c0de52193e04e91e087"
        PERSONA = "hiring"
        statu_code, text = modify_user_persona(API_KEY, user_id, PERSONA)
        self.assertEqual(statu_code, 200, 'Cannot modify user persona.')
    
    def test_load_file(self):
        file_path = "./Materials/SOP.pdf"
        files = load_file(file_path)
        self.assertNotEqual(len(files), 0, 'Fail to load document files.')
    
    def test_convert_to_json_file(self):
        text = "{}"
        user_id = "unittest"
        output_dir = "./Materials/"
        output_file = convert_to_json_file(text, user_id, output_dir)
        self.assertTrue(os.access(output_file, os.W_OK), 'Fail to create json file.')
    
    def test_check_json_file(self):
        output_file = "./Materials/PsyProfile_jiaqili.json"
        text = check_json_file(output_file)
        self.assertNotEqual(len(text["results"]["personality_analysis"]), 0, 'Fail to check json file.')
    
    def test_profile_psychometric(self):
        user_id = "jiaqili"
        file_path = "./Materials/SOP.pdf"
        output_dir = "./Materials/"
        API_KEY = "chrexec_96effa3d5b5f3c0de52193e04e91e087"
        PERSONA = "hiring"
        output_file = profile_psychometric(file_path, user_id, output_dir, API_KEY, PERSONA)
        self.assertTrue(os.access(output_file, os.W_OK), 'Fail to profile psychometric for ' + user_id + '.')
    
class TestResumeParser(unittest.TestCase): 

    def test_load_json_template(self):
        template_path = "./Materials/ResumeTemplate.json"
        json_template_string = load_json_template(template_path)
        self.assertIsNotNone(json_template_string, 'Fail to load json template for resume parsing.')

    def test_load_resume(self):
        resume_path = "./Materials/Resume.pdf"
        resume_text = load_resume(resume_path)
        self.assertIsNotNone(resume_text, 'Fail to extract text from resume.')

    def test_call_openal_api(self):
        json_template_string = "{'Name': ''}"
        resume_text = "Name: hack"
        openai_api_key = "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"
        generated_text = call_openal_api(json_template_string, resume_text, openai_api_key)
        self.assertIsNotNone(generated_text, 'Fail to fetch result from openai api.')

    def test_parse_resume(self):
        resume_path = "./Materials/Resume.pdf"
        template_path = "./Materials/ResumeTemplate.json"
        user_id = "hack"
        output_dir = "./Materials/"
        api_key = "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"
        output_file = parse_resume(resume_path, template_path, user_id, output_dir, api_key)
        self.assertTrue(os.access(output_file, os.W_OK), 'Fail to parse the resume')

def run_unit_tests():
    test_classes_to_run = [TestResumeParser, TestPsyProfile]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)

if __name__ == '__main__':
    run_unit_tests()
