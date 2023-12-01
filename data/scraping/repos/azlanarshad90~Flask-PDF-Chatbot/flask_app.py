from flask import Flask, request, render_template, flash, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from io import BytesIO
import os


app = Flask(__name__)


app.secret_key = 'chatbot'
env_config = os.getenv("APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)
os.environ["OPENAI_API_KEY"] = app.config.get("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

class UploadForm(FlaskForm):
    file = FileField('PDF File', validators=[InputRequired()])
    submit = SubmitField('Submit')

@app.route('/',methods=["GET","POST"])
def home():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash ('File upload successfully!')
        session["filename"]=filename
    return render_template('index.html', form=form, filename=session.get("filename", None))

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    filename = session.get("filename",None)
    loader = PyPDFLoader(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    docs = docsearch.vectorstore.similarity_search(userText)
    prompt = f"""
        Answer the question that is given under triple round brackets with maximum detail.
        Make the answer maximum detailed with respect to your capability.
        Follow the format of the answer that is given in the question, if no information is provided, you can use the format you think is best equipped according to question.
        ((({userText})))
        """
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answer = chain.run(input_documents=docs, question=prompt)
    return str(answer)


if __name__ == "__main__":
    app.run()
    
