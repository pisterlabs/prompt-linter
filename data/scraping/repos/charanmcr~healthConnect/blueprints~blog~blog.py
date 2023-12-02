import datetime
import bcrypt
import pymongo
import os
import openai
from bson import ObjectId
from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for,Flask,  current_app
from blueprints.database_connection import users, hospitals, appointments, doctors , blogVar
import math,random
from datetime import datetime

blog = Blueprint("blog", __name__, template_folder="templates")

@blog.route('/blogDetails<string:blog_id>')
def blogDetails(blog_id):
   doctor_id = session.get('doctor_id')
   user_is_doctor = False
   if doctor_id:
       user_is_doctor = True
   try:
        blog_object_id = ObjectId(blog_id)
        blog = blogVar.find_one({'_id': blog_object_id})
        if blog:
            return render_template('blog/blogdetails.html', blog=blog, user_is_doctor=user_is_doctor)
        else:
            # Handle the case where the blog doesn't exist
            return render_template('error.html', error_message="Blog not found")
   except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return render_template('error.html', error_message=error_message)

@blog.route('/blogs')
def blogs():
    try:
        all_blogs = blogVar.find()
        doctor_id = session.get('doctor_id')
        doctor_data = None
        user_is_doctor = False
        if doctor_id:
            doctor_data = doctors.find_one({"_id": ObjectId(doctor_id)})
            print(doctor_data)
            user_is_doctor = True
        return render_template('blog/blogposts.html', blogs=all_blogs, doctor_data=doctor_data, user_is_doctor=user_is_doctor)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return "raising error"
    
@blog.route('/myblogs')
def myblogs():
    try:
        doctor_id = session.get('doctor_id')
        if(doctor_id):
            user_is_doctor = True
            doctor_blogs = blogVar.find({"doctor_id": doctor_id})
            return render_template('blog/blogposts.html', blogs=doctor_blogs,user_is_doctor=user_is_doctor)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return "raising error"

@blog.route('/blogForm', methods=['POST','GET'])
def blogForm():
    doctor_id = session.get('doctor_id')
    if doctor_id:
        doctor_data = doctors.find_one({"_id": ObjectId(doctor_id)})
        user_is_doctor = True
        if request.method == 'POST':
            try:
                title = request.form.get('blog_title')
                content = request.form.get('blog_description')
                category = request.form.get('blog_category')
                creation_date = datetime.now().date().strftime('%Y-%m-%d')
                
                doctor_name = None
                if doctor_id:
                    doctor = doctors.find_one({'_id': ObjectId(doctor_id)})
                    if doctor:
                        doctor_name = doctor['name']
                
                words_per_minute = 60  # Adjust as needed
                words = content.split()
                total_words = len(words)
                reading_time = math.ceil(total_words / words_per_minute)
                
                image_path = None
                random_views = random.randint(100, 2000)
                random_comments = random.randint(1, 20)
                
                if 'blog_image' in request.files:
                    blog_image = request.files['blog_image']
                    if blog_image.filename != '':
                        filename = os.path.join(current_app.config['UPLOAD_FOLDER'], blog_image.filename)
                        blog_image.save(filename)
                        image_path = os.path.join('uploads', blog_image.filename)
            
                blog_data = {
                    'doctor_id': doctor_id,
                    'doctor_name': doctor_name,
                    'title': title,
                    'content': content,
                    'category': category,
                    'image_path': image_path,
                    'reading_time': reading_time,
                    'posted_date': creation_date,
                    'post_views': random_views,
                    'post_comments': random_comments
                }
                blogVar.insert_one(blog_data)
                
                return redirect(url_for('blog.blogs'))

            except Exception as e:
                # Handle the exception
                error_message = f"An error occurred: {str(e)}"
                return redirect(url_for('blog.blogForm'))
        return render_template('blog/blogform.html',doctor_data = doctor_data)
    else:
        return redirect(url_for('blog.blogs'))


