import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from resumeparser import ResumeParser
import json
import openai

app = Flask(__name__)

OPENAI_API_KEY = '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'
parser = ResumeParser(OPENAI_API_KEY)

# Define weights for different factors
weights = {
    'similarity': 0.4,
    'cgpa': 0.3,
    'experience': 0.2,
    'projects': 0.1
}

@app.route('/calculate-score', methods=['POST'])
def calculate_score():
    try:
        resume_url = "https://uploadthing.com/f/d4149a99-09d5-4b75-a380-65012fefbfeb_file_7bd3fc.pdf"
        
        # Download the resume PDF from the provided URL
        pdf_file = request.json.get(resume_url)
        with open('uploads/resume.pdf', 'wb') as f:
            f.write(pdf_file.content)
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            
            response_text = parser.query_resume('uploads/resume.pdf')
            response_text = response_text.replace("Link\n\n", "")

            candidate = json.loads(response_text)


            
            job_description = request.json.get('job_description')
            if not job_description:
                return jsonify({"error": "Job description is required."}), 400

            # Preprocess and tokenize the job description and candidate's information
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            job_desc_tfidf = tfidf_vectorizer.fit_transform([job_description])

            candidate_info = " ".join([
                candidate['basic_info']['full_name'],
                candidate['basic_info']['education_level'],
                " ".join([exp['job_title'] for exp in candidate['work_experience']]),
                " ".join([proj['project_name'] for proj in candidate['project_experience']])
            ])
            candidate_tfidf = tfidf_vectorizer.transform([candidate_info])

            # Calculate similarity score
            similarity_score = cosine_similarity(job_desc_tfidf, candidate_tfidf)[0][0]

            # Calculate CGPA score
            cgpa_score = float(candidate['basic_info']['GPA']) / 10 * weights['cgpa']

            # Calculate experience score
            # experience_years = int(candidate['experience_years'].split()[0])
            experience_score = 3 * weights['experience']

            # Calculate projects score
            projects_score = len(candidate['project_experience']) * weights['projects']

            # Calculate the final score out of 100
            total_score = (similarity_score * weights['similarity'] +
                           cgpa_score + experience_score + projects_score) * 100

            return jsonify({
                "candidate_name": candidate['basic_info']['full_name'],
                "similarity_score": similarity_score,
                "cgpa_score": cgpa_score,
                "experience_score": experience_score,
                "projects_score": projects_score,
                "total_score": total_score
            })
        else:
            return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
def generate_question(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        stop=None,
        temperature=0.7
    )
    return response
    
@app.route('/generate-interview-questions', methods=['POST'])
def generate_interview_questions():
    try:
        
        
        # Download the resume PDF from the provided URL
        pdf_file = request.get('resume_url')
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            pdf_path = os.path.join('uploads', pdf_file.filename)
            pdf_file.save(pdf_path)
            
            response_text = parser.query_resume(pdf_path)
            response_text = response_text.replace("Link\n\n", "")

            candidate = json.loads(response_text)


            
            job_description = request.get('job_description')
            jd_tech_keywords = ["react", "frontend", "javascript"]
            
            # Extract tech stack from candidate's project experience
            # tech_stack = candidate['Skills'][0]
            
            # Combine JD and tech stack keywords for generating questions
            # keywords = jd_tech_keywords
            
            # Generate questions using OpenAI GPT-3
            prompt1 = f"Generate 5 questions based on {candidate['work_experience']}"
            response1 = generate_question(prompt1)
            generated_question1 = response1.choices[0].text.strip()
            
            prompt2 = f"generate 5 technical questions based on common techstack between {job_description}"
            response2 = generate_question(prompt2)
            generated_question2 = response2.choices[0].text.strip()
            
            # Combine generated questions
            generated_questions = generated_question1 + "\n\n" + generated_question2
            
            # Return JSON response
            return jsonify({"questions": generated_questions})
        else:
            return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(port=3000, debug=True)
