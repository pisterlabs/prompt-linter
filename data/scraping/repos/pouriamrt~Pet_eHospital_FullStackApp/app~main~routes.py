from app.main import bp
from flask import render_template, request, url_for, jsonify
from flask_login import login_required, current_user
from openai import OpenAI
import os
from app.models.ContactForm import ContactForm
from app.extensions import db

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
departments = ["general", "dental", "orthopedic", "surgery", "ophthalmology"]

@bp.route('/')
@login_required
def index():
    global messages
    messages=[
        {"role": "system", "content": "You are an intelligent assistant for a pet online hospital app that gives a short general and unprofessional solution that they can try."},
        {"role": "system", "content": "You are an intelligent assistant for a pet online hospital app that recommends a suitable online hospital department webpage in the app to the customer and nothing more."},
        {"role": "system", "content": "You are an intelligent assistant for a pet online hospital which its departments are only [general, dental, orthopedic, surgery, ophthalmology] ."}
    ]
    
    return render_template('index.html', name=current_user.name)

@bp.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form["msg"]
    reply, recommended_department = get_Chat_response(msg)
    if recommended_department:
        link_str = ' * Here is the link to the department: <a style="color:red;" href="' + url_for('AI_suggestion.get_suggestion_page', department=recommended_department, _external=True) + '">' + f'The {recommended_department} department</a>'
        reply += link_str
        messages.append({"role": "assistant", "content": link_str})
    return reply

def get_Chat_response(text):
    if text:
        messages.append({"role": "user", "content": text})
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.5
        )

        reply = completion.choices[0].message.content
        messages.extend([{"role": "assistant", "content": text}, {"role": "assistant", "content": reply}])
        
        recommended_department = ""
        for department in departments:
            if department in reply.lower():
                recommended_department = department
                break
        return reply, recommended_department

@bp.route('/about')
@login_required
def about():
    return render_template('About.html')

@bp.route('/contact')
@login_required
def contact():
    return render_template('Contact.html')

@bp.route('/submit_contact_request', methods=['POST'])
@login_required
def submit_contact_request():
    name = request.form.get('name')
    phone = request.form.get('phone')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    new_request = ContactForm(name=name, phone=phone, email=email, subject=subject, message=message)
    db.session.add(new_request)
    db.session.commit()

    return jsonify({'success': True})