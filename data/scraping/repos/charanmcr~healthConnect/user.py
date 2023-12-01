import datetime
import bcrypt
import pymongo
import os
import openai
from blueprints.getTokens import addEvent
from blueprints.ibm_connection import cos, cosReader
from blueprints.user.generate_slots import generate_slots
from bson import ObjectId
from flask import Blueprint, jsonify, make_response, redirect, render_template, request, session, url_for
from blueprints.database_connection import users, hospitals, appointments, doctors,tokens
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from blueprints.redis_connection import r 
from blueprints.blockChainLogging import blockChain
from blueprints.confidential import OPEN_AI_KEY
import xgboost as xgb
import pandas as pd
import numpy as np


user = Blueprint("user", __name__, template_folder="templates")
specialties = ['Cardiology', 'Dermatology', 'Endocrinology', 'Gastroenterology', 'General Practice', 'Infectious Diseases', 'Neurology', 'Oncology', 'Pediatrics', 'Psychiatry', 'Pulmonology', 'Radiology', 'Rheumatology']

@user.before_request
def check_session():
    if request.endpoint not in ['user.login', 'user.register','user.hello_world','user.doc_out' ] and '_id' not in session:
        return redirect(url_for('user.login'))


@user.route('/')
def hello_world():
   session.clear()
   res = make_response(render_template('user/indexMain.html'))
   res.set_cookie('FindDoctor', 'False')
   res.set_cookie('Book','False')
   return res

# Route for user registration
@user.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        aadharnumber = request.form.get('aadharnumber')
        password = request.form.get('password') 
        name = request.form.get('name')
        phone = request.form.get('phone')
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        
        #generating pem
        private_key_pem = private_key.private_bytes(encoding=serialization.Encoding.PEM,format=serialization.PrivateFormat.PKCS8,encryption_algorithm=serialization.NoEncryption())
        public_key_pem = public_key.public_bytes(encoding=serialization.Encoding.PEM,format=serialization.PublicFormat.SubjectPublicKeyInfo)
        
        # Validate required fields
        if not aadharnumber or not password or not name or not phone:
            return render_template('login.html', message='All Fields are required')

        # Check duplicate usernames
        existing_user = users.find_one({'aadharnumber': aadharnumber})

        if existing_user:
            return render_template('user/login.html', message='User Already exists')
    
        # Hash password 
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Insert user
        user = {
        'aadharnumber':aadharnumber,
        'name': name,
        'password': hashed_password,
        'phone': phone,
        'private_key': private_key_pem.decode('utf-8'),
        'public_key': public_key_pem.decode('utf-8'),
        'streak':'-1'
        }
        
        try:
            result = users.insert_one(user)
        
            if result.inserted_id:
                session['aadharnumber'] = aadharnumber
                return redirect(url_for('user.stepsform'))
            else:
                return render_template('user/login.html', message='User not created')

        except Exception as e:
            print(e)
            return jsonify({'message': 'Unknown error'}), 500
    else:
        # GET request - show signup form
        return render_template('user/register.html')

#stepsform
@user.route('/stepsform',methods=['GET','POST'])
def stepsform():
    if request.method == "POST":
        fname = request.form['fName']
        lname = request.form['lName']
        email = request.form['email']
        phno = request.form['phno']
        city = request.form['city']
        dob = request.form['dob']
        gender = request.form['gender'] 
        marstat = request.form['marstat']
        ch = request.form['children']
        cargiv = request.form['cargiv']
        hndcp = request.form['handicapp']
        occ = request.form['occupation'] 
        bg = request.form['bloodgroup']
        alc = request.form['alcohol']
        currmed = request.form['currmed']
        emercon = request.form['emercon']
        chcom = request.form['chcom'] 
        aadharnumber = session.get('aadharnumber')
        #Insert form details
        details = {
        'firstName': 1,
        'lastname': 1,
        'email': 1,
        'phno': 1,
        'email': 1,
        'city': 1,
        'dob': 1,
        'noofchildren' : 0,
        'cargivers' : 1,
        'gender': 1,
        'marriage_status': 1,
        'bloodgroup': 1,
        'smokingConsumerHabits': 1,
        'is_handicapped': 1,
        'occupation': 1,
        'currentMedications': 1,
        'emergency_contact': 1,
        'cheif_complaint': 1,
        'aadharnumber': 1,
        'pdfReports': 0,    
        }

        data = {
        'firstName': fname,
        'lastname': lname,
        'email': email,
        'phno': phno,
        'email': email,
        'city': city,
        'dob': dob,
        'noofchildren' : ch,
        'cargivers' : cargiv,
        'gender': gender,
        'marriage_status': marstat,
        'bloodgroup': bg,
        'smokingConsumerHabits': alc,
        'is_handicapped': hndcp,
        'occupation': occ,
        'currentMedications': currmed,
        'emergency_contact': emercon,
        'cheif_complaint': chcom,  
        'emergency_profile' : details  
          
        }
        result = users.update_one(
        {'aadharnumber': aadharnumber},
        {'$set': data},
        upsert=True
    )
        return "Form Submitted Successfully"
    return render_template('user/stepreq.html')


@user.route('/login', methods=['GET', 'POST'])
def login():
    session.clear()
    if request.method == 'POST':
        aadharnumber = request.form['aadharnumber']
        password = request.form['password']
        user = users.find_one({'aadharnumber': aadharnumber})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['aadharnumber'] = aadharnumber
            session['_id'] = str(user['_id'])
            message = "User ID: "+ str(user['_id']) + " logged into his account"
            blockChain(message)
            if request.cookies.get('FindDoctor') == 'True':
                res = make_response(redirect(url_for('user.get_doctors')))
                res.set_cookie('FindDoctor', 'False')
                return res
            if request.cookies.get('Book') == 'True':
                res = make_response(redirect(url_for('user.get_doctors')))
                res.set_cookie('FindDoctor', 'False')
                return res
            return redirect(url_for('user.user_dashboard'))
        else:
            return render_template('user/login.html', message='Incorrect aadharnumber/password combination')
            
    return render_template('user/login.html')

@user.route('/search',methods=['GET'])
def search():
    keyword = request.args.get('keyword', '')
    suggestions = get_autocomplete_suggestions(keyword)
    return jsonify(suggestions)

def get_autocomplete_suggestions(keyword):
    # Perform MongoDB query for autocomplete suggestions
    regex_pattern = f'.*{keyword}.*'  # Construct a regex pattern
    query = {'hospital_name': {'$regex': regex_pattern, '$options': 'i'}}  # Case-insensitive regex search
    projection = {'_id': 0, 'hospital_name': 1}  

    # Query the 'hospitals' collection for suggestions
    suggestions_cursor = hospitals.find(query, projection)
    
    # Convert the cursor to a list of dictionaries
    suggestions = list(suggestions_cursor)
    return suggestions


# user dashboard
@user.route('/user_dashboard',methods=['GET'])
def user_dashboard():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        user_id = users.find_one({'aadharnumber':session['aadharnumber']},{"_id":1})
        userdata = users.find_one({'_id':user_id['_id']},{'password':0})
        today, _ = str(datetime.datetime.now()).split()
        appointment_id = appointments.find({'user_id':ObjectId(user_id['_id']), 'appointment_date': today},{'_id':1})
        code = ''
        for i in appointment_id:
            print(i)
            if r.get(str(i['_id'])):
                code = r.get(str(i['_id']))
                # code = code.decode('utf-8')
                if len(code)!=4:
                    code = None
                print(code)
                break
        query ={'user_id':user_id['_id']}
        res=[]
        appointments_data = appointments.find(query)
        appointments_data= list(appointments_data)
        for appointment_data in appointments_data:
            doctor_id = appointment_data['doctor_id']
            doctor_data = get_doc_details(doctor_id)
            
            combined_data = {
                'doctor': {
                    'name': doctor_data['name'],
                    'email': doctor_data['email'],
                    'hospital_name': doctor_data['hospital']
                },
                'appointment': {
                    'appointment_date': appointment_data['appointment_date'],
                    'appointment_time': appointment_data['appointment_time'],
                    'status': appointment_data['status'],
                    'issue': appointment_data['issue'],
                    'reviews': appointment_data['reviews'],
                }
            }
            res.append(combined_data)

    requests = users.find_one({'_id':ObjectId(user_id['_id'])},{'caregivers':1})     
    return render_template('user/user-dashboard.html',appointments=res,userdata=userdata, code=code , requests=requests)



@user.route('/my_appointments',methods=['GET'])
def my_appointements():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        user_id = users.find_one({'aadharnumber':session['aadharnumber']},"_id")
        query ={'user_id':user_id['_id']}
        res=[]
        appointments_data = appointments.find(query,{}).sort([("timestamp",pymongo.DESCENDING)])
        appointments_data= list(appointments_data)
        for appointment_data in appointments_data:
            doctor_id = appointment_data['doctor_id']
            doctor_data = get_doc_details(doctor_id)
            
            combined_data = {
                'doctor': {
                    'name': doctor_data['name'],
                    'email': doctor_data['email'],
                    'hospital_name': doctor_data['hospital_address'],
                    'location': doctor_data['location'],
                    'recommendation_score': doctor_data['recommendation_score'],
                    'speciality': doctor_data['speciality'],
                    'experience': doctor_data['experience'],
                    'fees': doctor_data['fees'],
                    # Other doctor fields

                },
                'appointment': {
                    'appointment_date': appointment_data['appointment_date'],
                    'appointment_time': appointment_data['appointment_time'],
                    # Other appointment fields
                }
            }
            
            res.append(combined_data)
        return render_template('user/my-appointments.html',appointments_data=res)
    
@user.route('/get_doc_details',methods=['GET'])
def get_doc_details(doctor_id):
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        doctor_details = doctors.find_one({'_id':doctor_id})
        return  doctor_details
    
@user.route('/my_reports',methods=['GET'])
def my_reports():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        user = users.find_one({'_id': ObjectId(session['_id'])})
        return render_template('user/my-reports.html', reports=user['pdfReports'])

@user.route('/my_profile',methods=['GET'])
def my_profile():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        user=users.find_one({'aadharnumber':session['aadharnumber']})
        PDFreports=[{'filetype':'x-ray','filename':'tdt.txt'},{'filetype':'prescription','filename':'tdt.txt'}] 
        return render_template('user/my-profile.html',reports=PDFreports ,user=user)
    
@user.route('/search_doctors',methods=['POST','GET'])
def search_docotors():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        hospital_name = request.form.get('hospital')
        location = request.form.get('location')
        symptoms = request.form.get('symptoms')
        # Create a query based on the provided criteria
        query = {
            'hospital': hospital_name,
            'location': location,
        }
        
        # Find doctors matching the query and retrieve their IDs
        doctors_cursor = doctors.find(query)
        doctors_cursor = list(doctors_cursor)
        hospitals_loc_data = get_hospitals_locations()
        hospitals_names = hospitals.distinct('hospital_name')
        # Extract doctor IDs from the cursor
        return render_template('user/doctors.html',doctors_data=doctors_cursor , hospitals_names=hospitals_names ,locations=hospitals_loc_data)


@user.route('/doctor_appointments1')
def doctor_appointments1():
    appointment1 = appointments.find({"user_id": ObjectId(session['_id']),'status':'completed'}).sort('timestamp',-1)
    appointments_with_users = appointment1
    return render_template('user/prescriptions_list.html', appointments_with_users=appointments_with_users)
 
@user.route('/doctor_reviews1/<appointment_id>/<doctor_id>')
def doctor_reviews1(appointment_id,doctor_id):
    user_id = session.get('_id')
    plist=appointments.find({'_id':ObjectId(appointment_id)},{'prescription':1})
    report_review1=appointments.find({'_id':ObjectId(appointment_id)},{'_id':0,'notes':1})
    report_review=report_review1[0]['notes']
    return render_template('user/app-invoice.html',plist=plist,appointment_id=appointment_id,user_id=user_id,report_review=str(report_review))

@user.route('/get_doctors',methods=['GET'])
def get_doctors():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        doctors_data = doctors.find().limit(100)
        doctors_data= list(doctors_data)
        hospitals_loc_data = doctors.distinct('location')
        hospitals_names = hospitals.distinct('hospital_name')
        return render_template('user/doctors.html',doctors_data=doctors_data , hospitals_names=hospitals_names ,locations=hospitals_loc_data)
    
@user.route('/prescriptions_list',methods=['GET','POST'])
def prescriptions_list():
     if request.method == 'POST':
        medicine_id = request.form['charge'].split(')')[0]
        days = request.form['days']
        evn = request.form.get("evn") != None
        aft = request.form.get("aft") != None
        mor = request.form.get("mor") != None

@user.route('/recommendMydoctor',methods=['GET','POST'])
def recommendMydoctor():
    if request.method=='POST':
        hospitals_loc_data = doctors.distinct('location')
        hospitals_names = hospitals.distinct('hospital_name')
        hospital_name = request.form['hospital']
        location = request.form['location']
        doctors_data = doctors.find().limit(100)
        doctors_data= list(doctors_data)
        symptoms = request.form.getlist('symptoms[]')
        if symptoms!=[]  and hospital_name != 'Select Hospital' and location!='Select Location':
            specialist = str(get_specialist(symptoms, session['age'], session['gender'])).strip()
            sorted_doctors= doctors.find({'hospital':hospital_name,'speciality': specialist,'location': location}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        elif symptoms!=[]  and location!='Select Location':
            specialist = str(get_specialist(symptoms, session['age'], session['gender'])).strip()
            sorted_doctors= doctors.find({'speciality': specialist,'location': location}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        elif  hospital_name !='Select Hospital' and location!='Select Location':
             sorted_doctors= doctors.find({'hospital':hospital_name,'location': location}).sort('recommendation_score',-1)
             sorted_doctors=list(sorted_doctors)
             return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        elif hospital_name != 'Select Hospital' and symptoms!=[]:
            specialist = str(get_specialist(symptoms, session['age'], session['gender'])).strip()
            sorted_doctors= doctors.find({'hospital':hospital_name,'speciality': specialist}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data) 
        elif symptoms!=[]:
            specialist = str(get_specialist(symptoms, session['age'], session['gender'])).strip()
            sorted_doctors= doctors.find({'speciality': specialist}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        elif hospital_name != 'Select Hospital':
            sorted_doctors= doctors.find({'hospital':hospital_name}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        elif location!='Selected Location':
            sorted_doctors=  doctors.find({'location': location}).sort('recommendation_score',-1)
            sorted_doctors=list(sorted_doctors)
            return render_template('user/doctors.html',doctors_data=sorted_doctors,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
        else:
            return render_template('user/doctors.html',doctors_data=doctors_data,hospitals_names=hospitals_names ,locations=hospitals_loc_data)
    
def get_specialist(symptoms, age, gender):
  openai.api_key = OPEN_AI_KEY
  prompt = f"Based on these symptoms: {symptoms}, for a {gender} aged {age}, the most accurate initially needed medical specialty from this list: {specialties} is:"

  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    temperature=0.5, 
    max_tokens=60,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].text.strip()

@user.route('/book_appointment/<doctor_id>/<user_id>',methods=['GET'])
def book_appointment(doctor_id, user_id):
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        doctor_data = get_doc_details(ObjectId(doctor_id))
        return render_template('user/book-appointment.html',doctor_data=doctor_data)


@user.route('/user_logout',methods=['GET'])
def user_logout():
    session.clear()
    return redirect(url_for('user.login'))


@user.route('/get_hospitals_locations',methods=['GET'])
def get_hospitals_locations():
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        hospitals_data = hospitals.distinct('location')
        hospitals_loc_data= hospitals_data
        return hospitals_loc_data


@user.route('/confirm_booking/<string:doctor_id>', methods=['POST'])
def confirm_booking(doctor_id):
    if 'aadharnumber' in session:
        accessToken = doctor_id + session['_id']
        doctor_id = ObjectId(doctor_id)
        doctor_data = get_doc_details(ObjectId(doctor_id))
        user_id = ObjectId(session['_id'])
        selected_date = request.form['appointment_date']
        selected_time_slot = request.form['time_slot']  
        reason = request.form['reason']
        # Convert selected date and time to datetime objects
        selected_datetime = datetime.datetime.strptime(selected_date + ' ' + selected_time_slot, '%Y-%m-%d %I:%M %p')
        current_datetime = datetime.datetime.now()

        # Calculate the difference in days between selected date and current date
        days_difference = (selected_datetime.date() - current_datetime.date()).days
        # Check booking conditions
        if days_difference <= 15:
            if days_difference >= 0:
                # Booking is within 15 days, proceed with time check
                if selected_datetime > current_datetime + datetime.timedelta(minutes=10):
                    start_time = datetime.datetime.strptime(selected_time_slot, "%I:%M %p")
                    end_time = start_time + datetime.timedelta(minutes=30)
                    _ ,start_time = str(start_time).split()
                    _ ,end_time = str(end_time).split()
                    start_time = start_time[:-3]
                    end_time = end_time[:-3]
                    start_datetime = datetime.datetime.strptime(selected_date + " " + str(start_time), "%Y-%m-%d %H:%M").isoformat() + "+05:30"
                    end_datetime = datetime.datetime.strptime(selected_date + " " + str(end_time), "%Y-%m-%d %H:%M").isoformat() + "+05:30"
                    location = doctor_data['hospital_address'] + " " + doctor_data['location']
                    description = "Appointment with doctor "+ doctor_data['name'] + " (" + doctor_data['speciality'] +") @ " + str(selected_time_slot)
                    event = {
                        "summary": "Doctor Appointment",
                        "location": location,
                        "description": description,
                        
                        "start": {
                            "dateTime": start_datetime, 
                            "timeZone": "Asia/Kolkata"
                        },
                        
                        "end": {
                            "dateTime": end_datetime,
                            "timeZone": "Asia/Kolkata"
                        },

                        "reminders": {
                            "useDefault": False,
                            "overrides": [
                            {"method": "email", "minutes":180},
                            {"method": "popup", "minutes": 30}
                        ]
                        },
                        "visibility":"public",
                        "sendNotifications": True,
                        "sendUpdates": "all"
                    }

                    event_id  = addEvent(session['_id'],1,event=event)
                    booking_data = {
                        'user_id': user_id,
                        'doctor_id': doctor_id,
                        'appointment_date': selected_date,
                        'appointment_time': selected_time_slot,
                        'accessToken': accessToken,
                        'accessed':'0',
                        'timestamp': datetime.datetime.now(),
                        'issue': reason,
                        'reviews': '',
                        'notes': '',
                        'status': 'booked',
                        'lab_tests': [],
                        'lab_report': [],
                        'calendar_event_id':str(event_id)
                    }
            
                    # Insert the booking data into the database
                    appointments.insert_one(booking_data)

            
                    return render_template('user/book-appointment.html',message ="Appointment confirmed successfully!",type="success", doctor_data = doctor_data)
                else:
                    return render_template('user/book-appointment.html',message ="You can only book appointments that are more than 10 minutes away from the current time.", type="error" ,doctor_data = doctor_data)
            else:
                return render_template('user/book-appointment.html',message ="You cannot book appointments for a past date.", type="error" ,doctor_data = doctor_data)
        else:
            return render_template('user/book-appointment.html',message ="You can only book appointments up to 15 days from the current date", type="error" ,doctor_data = doctor_data)
    else:
        return redirect(url_for('login'))

@user.route('/check_appointments/<string:doctor_id>/<string:selected_date>/<int:dayOfWeek>')
def check_appointments(doctor_id, selected_date, dayOfWeek):
    days_of_week = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
    doctor_id = ObjectId(doctor_id)
    get_schedule = doctors.find_one({"_id":doctor_id},{"schedule":1})
    schedule = get_schedule['schedule'][days_of_week[dayOfWeek]]
    if 'surgery' in schedule:
        del schedule['surgery']
    available_slots = generate_slots(schedule, selected_date)
    get_appointments = list(appointments.find({"doctor_id":doctor_id,"appointment_date":selected_date},{"appointment_time":1, "_id":0}))
    get_appointments = [appointment['appointment_time'] for appointment in get_appointments]
    return jsonify({"available_slots":available_slots, "booked_slots":get_appointments})

@user.route('/already_in_appointment/<string:doctor_id>/<string:selected_date>')
def already_in_appointment(doctor_id, selected_date):
    doctor_id = ObjectId(doctor_id)
    if 'aadharnumber' in session:
        user_id = users.find_one({"aadharnumber":session['aadharnumber']},{"_id":1})
    else:
        return redirect(url_for('login'))
    get_appointment = list(appointments.find({"user_id":user_id,"doctor_id":doctor_id,"appointment_date":selected_date},{}))
    if len(get_appointment) >= 1:
        return jsonify({"message": "True"})
    return jsonify({"message":"False"})

@user.route('/doctor-profile/<string:doctor_id>',methods=['GET'])
def doctor_profile(doctor_id):
    if 'aadharnumber' not in session:
        return redirect(url_for('login'))
    else:
        doctor_data = get_doc_details(ObjectId(doctor_id))
        return render_template('user/doctor-profile.html',doctor_data=doctor_data)
    
# ----------------------==========Upload File =========------------------------

@user.route('/upload', methods=['POST'])
def upload_file():
    report_type = request.form['report_type']
    uploaded_file = request.files['file']

    if uploaded_file:
        user_name = session['_id']
        filename = f"{user_name}_{report_type}.pdf"

        # Save the uploaded file in the current folder
        uploaded_file.save(filename)

        try:
            # Upload the file to COS
            cos.upload_file(Filename=filename, Bucket='healthconnectibm', Key=filename)
        except Exception as e:
            os.remove(filename)
            return f"Error uploading to COS: {e}"
        else:
            message = "User ID: " + str(session['_id']) + " uploaded file(" + str(filename) +") into COS"
            blockChain(message)
            os.remove(filename)
            report_info = {'reportType': report_type, 'filename': filename}
            query = {"_id": ObjectId(session['_id'])}
            update = {"$push": {"pdfReports": report_info}}
            users.update_one(query, update)
            return render_template('user/my-reports.html',message ="File uploaded Succesfully !",type="success", reports = users.find_one(query)['pdfReports'])
    else:
        return render_template('user/my-reports.html',message ="File not uploaded",type="error", reports = users.find_one(query)['pdfReports'])

    
@user.route('/viewreports')
def view_reports():
  if '_id' in session:
    user = users.find_one({'_id': ObjectId(session['_id'])})
    if user:
      return render_template('my-reports.html', reports=user['pdfReports'])
  else: 
    return redirect(url_for('login'))
  
@user.route('/display_pdf/<filename>')
def display_pdf(filename):
    bucket_name = 'healthconnectibm'
    key_name = filename
    http_method = 'get_object'
    expiration = 600
    try:
        signedUrl = cosReader.generate_presigned_url(http_method, Params={'Bucket': bucket_name, 'Key': key_name}, ExpiresIn=expiration)
        message = "User ID: " + str(session['_id']) + " viewed file " + str(key_name)
        blockChain(message)
    except Exception as e:
        print(e)
        return "Cannot load data"
    return render_template('user/display-report.html', pdfUrl = signedUrl)

@user.route('/doc_out')
def doc_out():
    res = make_response(redirect(url_for('user.login')))
    res.set_cookie('FindDoctor', 'True')
    return res 

@user.route('/fit_data')
def fit_data():
    today = datetime.datetime.now()
    value = addEvent(session['_id'],2,date=today)
    user_age = users.find({'_id':ObjectId(session['_id'])},{'age':1,'streak':1,'_id':0})
    age= user_age[0]['age']
    streak = user_age[0]['streak']
    return render_template('user/fitness_data.html', fitness_data=value, age=age , streak=str(streak))

@user.route('/update_streak/<string:streak>', methods=['GET'])
def update_streak(streak):
    streak3 = users.update_one({'userID':ObjectId(session['_id'])},{'$set':{'streak':streak}})
    if streak3:
        response = {'message': 'Streak updated successfully'}
        return jsonify(response)

@user.route('/chatbot')
def chatbot():
    user=users.find_one({'_id':ObjectId(session['_id'])},{'name':1})
    return render_template('user/chatbot.html',user=user['name'])


appointments2 = {}
@user.route("/get_bot", methods=['GET'])
def get_bot():
    user_id=session['_id']
    user=users.find({'_id':ObjectId(user_id)},{'_id':0})
    age= user[0]['age']
    gender= user[0]['gender']
    symptoms_list = ["asthma","bronchitis","pneumonia","emphysema","hypertension","heart-disease","atherosclerosis","arrhythmia","ulcer","gastritis","crohns-disease","migraine","epilepsy","arthritis","osteoporosis","fibromyalgia","diabetes",
    "thyroid",
    "cushing-syndrome",
    "psoriasis",
    "eczema",
    "acne",
    "covid",
    "influenza",
    "hiv",
    "hepatitis",
    "lyme-disease",
    "breast-cancer",
    "lung-cancer",
    "prostate-cancer",
    "colon-cancer",
    "leukemia",
    "kidney-stones",
    "kidney-failure",
    "polycystic-kidney",
    "glaucoma",
    "cataracts",
    "macular-degeneration",
    "depression",
    "anxiety",
    "addiction",
    "ptsd"
    ]
    current_date = datetime.datetime.now().date()

    # Create a list to store the next seven days
    next_seven_days = [current_date + datetime.timedelta(days=i) for i in range(7)]

    # Convert the dates to strings in a desired format (e.g., 'YYYY-MM-DD')
    formatted_dates = [date.strftime('%Y-%m-%d') for date in next_seven_days]
    locations = hospitals.distinct('location')
    user_text = request.args.get('msg')

    if user_text.lower() == "book appointment":
        return jsonify(symptoms_list) # return symptoms list

    if user_text in symptoms_list:
        appointments2['symptoms']=user_text
        speciality = get_specialist(user_text,age,gender)
        appointments2['speciality']=speciality
        return jsonify(locations)
    if user_text in locations:
        appointments2['location']=user_text
        data = doctors.find({'location':appointments2['location'],'speciality':appointments2['speciality']},{'name':1,'_id':0})
        doctor_names = [item['name'] for item in data]
        appointments2['doctors']=doctor_names
        return jsonify(doctor_names)  

    if user_text in appointments2['doctors']:
        appointments2['doctor'] = user_text
        return jsonify(formatted_dates)

    if user_text in  formatted_dates:
        appointments2['date']=str(user_text)
        date2=str(user_text)
        date2 = datetime.datetime.strptime(date2, "%Y-%m-%d").date()
        dow = date2.weekday()
        doctorid=doctors.find({'name':appointments2['doctor'],'speciality':appointments2['speciality']},{'_id':1})
        doctor_id=doctorid[0]['_id']
        appointments2['doctor_id']=doctor_id
        check_appointments = check_appointments1(doctor_id,appointments2['date'],dow)
        appointments2['check_appointments']=check_appointments
        return jsonify(check_appointments)
    if user_text in appointments2['check_appointments']:
         time = user_text
         appointments2['time']=time
         status=confirm_booking1()
         if status == 'booked':
             return f'''Thank You for Choosing AI Online Booking Chatbot!!     
                                     Hope you get well soon!!         
                                     Booked successfully!!
                                     Your Slot Details:
                                     NAME OF THE DOCTOR:{appointments2['doctor']},
                                     date of booking:{appointments2['date']},
                                     Time Slot:{appointments2['time']},
                                     Speciality:{appointments2['speciality']},
                                     location:{appointments2['location']},
                                     You can Go Back Now, by clicking 'back' button.
             '''
         else:
             return 'booked not successfully'
    else:
        return "I didn't understand, say 'book appointment' to start booking"
    

def get_specialist2(symptoms, age, gender):
  openai.api_key = OPEN_AI_KEY
  prompt = f"Based on these symptoms: {symptoms}, for a {gender} aged {age}, the most accurate initially needed medical specialty from this list: {specialties} is:"

  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    temperature=0.5, 
    max_tokens=60,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].text.strip()


def check_appointments1(doctor_id, selected_date, dayOfWeek):
    days_of_week = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
    doctor_id = ObjectId(doctor_id)
    get_schedule = doctors.find_one({"_id":doctor_id},{"schedule":1})
    schedule = get_schedule['schedule'][days_of_week[dayOfWeek]]
    if 'surgery' in schedule:
        del schedule['surgery']
    available_slots = generate_slots(schedule, selected_date)
    get_appointments = list(appointments.find({"doctor_id":doctor_id,"appointment_date":selected_date},{"appointment_time":1, "_id":0}))
    get_appointments = [appointment['appointment_time'] for appointment in get_appointments]
    print(available_slots)
    return list(set(list(available_slots))-set(list(get_appointments)))

def get_doc_details1(doctor_id):
    doctor_details = doctors.find_one({'_id':doctor_id})
    return  doctor_details
def confirm_booking1():
    doctor_id=appointments2['doctor_id']
    accessToken = str(doctor_id)+ str(session['_id'])

    doctor_id = ObjectId(doctor_id)
    doctor_data = get_doc_details1(ObjectId(doctor_id))
    user_id = ObjectId(session['_id'])
    selected_date = appointments2['date']
    selected_time_slot = appointments2['time']  
    reason = appointments2['symptoms']
    # Convert selected date and time to datetime objects
    selected_datetime = datetime.datetime.strptime(selected_date + ' ' + selected_time_slot, '%Y-%m-%d %I:%M %p')
    current_datetime = datetime.datetime.now()

    # Calculate the difference in days between selected date and current date
    days_difference = (selected_datetime.date() - current_datetime.date()).days
    # Check booking conditions
    if days_difference <= 15:
        if days_difference >= 0:
            # Booking is within 15 days, proceed with time check
            if selected_datetime > current_datetime + datetime.timedelta(minutes=1):
                start_time = datetime.datetime.strptime(selected_time_slot, "%I:%M %p")
                end_time = start_time + datetime.timedelta(minutes=30)
                _ ,start_time = str(start_time).split()
                _ ,end_time = str(end_time).split()
                start_time = start_time[:-3]
                end_time = end_time[:-3]
                start_datetime = datetime.datetime.strptime(selected_date + " " + str(start_time), "%Y-%m-%d %H:%M").isoformat() + "+05:30"
                end_datetime = datetime.datetime.strptime(selected_date + " " + str(end_time), "%Y-%m-%d %H:%M").isoformat() + "+05:30"
                location = doctor_data['hospital_address'] + " " + doctor_data['location']
                description = "Appointment with doctor "+ doctor_data['name'] + " (" + doctor_data['speciality'] +") @ " + str(selected_time_slot)
                event = {
                    "summary": "Doctor Appointment",
                    "location": location,
                    "description": description,
                    
                    "start": {
                        "dateTime": start_datetime, 
                        "timeZone": "Asia/Kolkata"
                    },
                    
                    "end": {
                        "dateTime": end_datetime,
                        "timeZone": "Asia/Kolkata"
                    },

                    "reminders": {
                        "useDefault": False,
                        "overrides": [
                        {"method": "email", "minutes":180},
                        {"method": "popup", "minutes": 30}
                    ]
                    },
                    "visibility":"public",
                    "sendNotifications": True,
                    "sendUpdates": "all"
                }

                event_id  = addEvent(session['_id'],1,event=event)
                booking_data = {
                    'user_id': user_id,
                    'doctor_id': doctor_id,
                    'appointment_date': selected_date,
                    'appointment_time': selected_time_slot,
                    'accessToken': accessToken,
                    'accessed':'0',
                    'timestamp': datetime.datetime.now(),
                    'issue': reason,
                    'reviews': '',
                    'notes': [],
                    'status': 'booked',
                    'lab_tests': [],
                    'lab_report': [],
                    'calendar_event_id':str(event_id)
                }
        
                # Insert the booking data into the database
                appointments.insert_one(booking_data)
                return "booked"
@user.route('/update_emgergency_visibility' , methods=['POST' , 'GET'])
def update_emgergency_visibility():
    data = request.get_json()
    user_id = data.get('user_id')
    visibility = data.get('visibility')
    availability = data.get('availability')
    update_query = {'$set': {f'emergency_profile.{visibility}': availability}}
    res = users.update_one({'_id':ObjectId(user_id)},update_query)
    if res.modified_count == 1 :
        return jsonify({'message':'success'})
    else:
        return jsonify({'message':'failed'})
    
@user.route('/emergency_profile',methods=['GET'])
def emergency_profile():
    if '_id' in session:
        user =  users.find_one({'_id': ObjectId(session['_id'])},{'private_key':0,'public_key':0,'password':0,'password':0 , '_id':1})
        print(user)
        return render_template('user/emergency_profile.html',user=user , details = user)           


@user.route('/care_givers' , methods=['POST','GET'])
def care_givers():
    if '_id' in session:
        return render_template('user/care-givers.html',care_giverdetails = None)


@user.route('/send-care-giver-request',methods=['GET','POST'])
def send_care_giver_request():
    if '_id' in session:
        req = request.form.get('user_id')
        if req == session['aadharnumber']:
            return render_template('user/care-givers.html',care_giverdetails = None , message = 'You cannot send request to yourself')
        if req == None:
            return render_template('user/care-givers.html',care_giverdetails = None , message = 'Please enter a valid aadhar number')
        req_id =  users.find_one({'aadharnumber' : req},{'_id':1})
        print(req_id)
        user_data = users.find_one({'_id':ObjectId(session['_id'])})  
        print(user_data)
        for i in user_data['caregivers']:
            if i['care_giver_id'] == req_id['_id']:
                return render_template('user/care-givers.html',care_giverdetails = user_data , message = 'You have already sent a request to this caregiver')
        cid = users.find_one({'aadharnumber':str(req)},{'_id':1})
        query = {'$push': {'caregivers': {'care_giver_id': cid['_id'], 'status': 'booked'}}}     
        users.update_one({'_id': ObjectId(session['_id'])}, query)
        users.update_one({'_id': ObjectId(cid['_id'])}, {'$push': {'caregivers': {'care_giver_id': ObjectId(session['_id']), 'status': 'booked'}}})
        userdata = users.find_one({'_id': ObjectId(session['_id'])})
        res=[]
        for i in userdata['caregivers']:
            name = users.find_one({'_id':i['care_giver_id']},{'name':1,'_id':0})
            res.append({'name':name['name'],'status':i['status']})
        
        return render_template('user/care-givers.html',care_giverdetails = res , message = 'Request Sent Successfully')


@user.route('/approve-care-giver-request/<care_id>',methods=['GET','POST'])
def approve_care_giver_request(care_id):
    if '_id' in session :
        query = {'$set': {'caregivers.$.status': 'approved'}}
        cid = users.find_one({'aadharnumber':str(care_id)},{'_id':1})
        users.update_one({'_id': ObjectId(session['_id']), 'caregivers.care_giver_id': cid['_id']}, query)
        userdata = users.find_one({'_id': ObjectId(session['_id'])})
        return redirect(url_for('user_dashboard'))
    
@user.route('/reject-care-giver-request/<care_id>',methods=['GET','POST'])
def reject_care_giver_request(care_id):
    if '_id' in session :
        query = {'$set': {'caregivers.$.status': 'rejected'}}
        cid = users.find_one({'aadharnumber':str(care_id)},{'_id':1})
        users.update_one({'_id': ObjectId(session['_id']), 'caregivers.care_giver_id': cid['_id']}, query)
        userdata = users.find_one({'_id': ObjectId(session['_id'])})
        return  redirect(url_for('user_dashboard'))
    
@user.route('/diabeticPrediction', methods=['POST'])
def diabeticPrediction():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        pred =predict_diabetes([preg,glucose,bp,st,insulin,bmi,dpf,age])
        today = datetime.datetime.now()
        value = addEvent(session['_id'],2,date=today)
        user_age = users.find({'_id':ObjectId(session['_id'])},{'age':1,'streak':1,'_id':0})
        age= user_age[0]['age']
        streak = user_age[0]['streak']
        if  pred==1:
            message = 'Had Diabetic'
        else:
            message = 'No Diabetic'
        return render_template('user/fitness_data.html', fitness_data=value, age=age , streak=str(streak),prediction=message)


# Create prediction function 
def predict_diabetes(input_data):
    model = xgb.XGBClassifier()
    model.load_model('blueprints//user//Models//diabetes_model.json')
    input_data = np.array(input_data).reshape(1, -1) 
    input_df = pd.DataFrame(input_data, columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    prediction = model.predict(input_df)
    return prediction[0]



