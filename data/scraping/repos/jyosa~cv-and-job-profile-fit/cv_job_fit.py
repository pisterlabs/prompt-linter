import PyPDF2
import openai
import tkinter as tk
from tkinter import filedialog, messagebox

# Set your OpenAI API key in a secure way instead of hardcoding it into the script.
openai.api_key = 'here your openai key'


def pdf_to_text(pdf_path):
    # Open the PDF file in binary read mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        # Iterate over each page in the PDF
        for page in range(len(pdf_reader.pages)):
            # Extract text from each page and concatenate it to the text variable
            text += pdf_reader.pages[page].extract_text()
    return text

def analyze_cv(cv_text, job_profile):
    # Construct a prompt for the OpenAI model to follow for analysis
    prompt = f"""
### Job Profile
{job_profile}

### CV
{cv_text}

### Analysis
Analyze the CV with respect to the job profile.

#### Strong Points
- Identify the areas where the candidate's CV aligns well with the job profile requirements.

#### Weak Points & Recommendations for Improvement
- List the areas where the candidate's CV does not meet the job profile requirements, and suggest how the candidate might improve these aspects.

#### Conclusion
- Summarize the overall fit of the candidate for the position based on the CV and job profile.

#### Percentage Fit
- Estimate a percentage fit of the candidate to the job profile, based on the comparison of the CV and job profile.

### Table of Results
- Provide a table summarizing the strong points, weak points, recommendations for improvement, conclusion, and percentage fit.
"""

    # Call the OpenAI API using the created prompt for text analysis
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=3500
    )

    # Extract the content from the response received from the API
    analysis = response['choices'][0]['message']['content'].strip()
    return analysis

def upload_file():
    global file_path
    # Use a file dialog to select a PDF and store the path
    file_path = filedialog.askopenfilename()

def submit_analysis():
    # Check if the file path and job profile are not empty
    if not file_path or not job_profile_text.get("1.0", tk.END).strip():
        messagebox.showerror("Error", "Please upload a CV and enter a job profile before submitting.")
        return
    # Extract text from the PDF file
    cv_text = pdf_to_text(file_path)
    # Get the job profile text from the text widget
    job_profile = job_profile_text.get("1.0", tk.END).strip()
    # Analyze the CV using the extracted text and job profile
    analysis = analyze_cv(cv_text, job_profile)
    
    # Update the result_text widget with the new analysis
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, analysis)

# Initialize the main application window
app = tk.Tk()
app.title("CV Analyzer")
app.geometry("800x600")

# Set up UI elements for job profile entry
job_profile_label = tk.Label(app, text="Enter Job Profile:")
job_profile_label.pack()

job_profile_text = tk.Text(app, height=10, width=80)
job_profile_text.pack()

# Button to upload a CV PDF file
upload_button = tk.Button(app, text="Upload CV PDF", command=upload_file)
upload_button.pack()

# Button to submit the CV for analysis
submit_button = tk.Button(app, text="Submit Analysis", command=submit_analysis)
submit_button.pack()

# Label and text widget to display the analysis result
result_label = tk.Label(app, text="Analysis Result:")
result_label.pack()

result_text = tk.Text(app, height=20, width=80)
result_text.pack()

# Initialize file_path as None to store the path of the uploaded file
file_path = None

# Start the main loop of the application
app.mainloop()
