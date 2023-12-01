from customtkinter import *
import pymysql
from datetime import * 
import tkinter as tk
from tkinter import ttk
from PIL import Image
import time 
import wikipedia
import random
import speech_recognition as sr
import pyttsx3
from twilio.rest import Client
import tkinter.messagebox as messagebox
import openai
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import socket
from tkdial import ScrollKnob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pyautogui
from reportlab.lib import colors
import webbrowser

#       ___          _____          ___                       ___           ___           ___           ___     
#      /  /\        /  /::\        /__/\          ___        /  /\         /  /\         /  /\         /__/|    
#     /  /:/_      /  /:/\:\       \  \:\        /  /\      /  /::\       /  /::\       /  /:/        |  |:|    
#    /  /:/ /\    /  /:/  \:\       \  \:\      /  /:/     /  /:/\:\     /  /:/\:\     /  /:/         |  |:|    
#   /  /:/ /:/_  /__/:/ \__\:|  ___  \  \:\    /  /:/     /  /:/~/:/    /  /:/~/::\   /  /:/  ___   __|  |:|    
#  /__/:/ /:/ /\ \  \:\ /  /:/ /__/\  \__\:\  /  /::\    /__/:/ /:/___ /__/:/ /:/\:\ /__/:/  /  /\ /__/\_|:|____
#  \  \:\/:/ /:/  \  \:\  /:/  \  \:\ /  /:/ /__/:/\:\   \  \:\/:::::/ \  \:\/:/__\/ \  \:\ /  /:/ \  \:\/:::::/
#   \  \::/ /:/    \  \:\/:/    \  \:\  /:/  \__\/  \:\   \  \::/~~~~   \  \::/       \  \:\  /:/   \  \::/~~~~ 
#    \  \:\/:/      \  \::/      \  \:\/:/        \  \:\   \  \:\        \  \:\        \  \:\/:/     \  \:\     
#     \  \::/        \  \/        \  \::/          \  \:\   \  \:\        \  \:\        \  \::/       \  \:\    
#      \__\/          \_/          \__\/            \__\/    \__\/         \__\/         \__\/         \__\/    
#
#



#-------------------------------------------database connection------------------------------------------------------------------------------------

db=pymysql.connect(host='localhost',user='root',password='root',database="student_database",charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)

#-------------------------------------------------Global-----------------------------------------------------------------------------

current_image = 0
query="select * from student"
admin_stat=False
student_stat=True
teacher_stat=False
student_id=1
teacher_id=1
flag_bot=True
first_enter=True
#-----------------------------------------------Common Functions----------------------------------------------------------------------------------------

#flash message for adding form
def flash_message(text, flashlb):
    def clear_flash_message(flashlb):
        flashlb.configure(text="")
        flashlb.update()
    flashlb.configure(text=text)
    flashlb.update()
    flashlb.after(3000, clear_flash_message, flashlb)


def reverse_animation( frame_to_animate, speed):
    target_width = frame_to_animate.cget("width")
    target_height = frame_to_animate.cget("height")
    current_width = target_width
    current_height = target_height

    while current_width > 0 or current_height > 0:
        current_width = max(current_width - speed, 0)
        current_height = max(current_height - speed, 0)
        frame_to_animate.configure(width=current_width, height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)

def animate_frame(frame_to_animate):
    target_height = frame_to_animate.cget("height")
    frame_to_animate.configure(height=0)
    
    current_height = 0

    while current_height < target_height:
        current_height = min(current_height + 30, target_height)
        frame_to_animate.configure( height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)

def animate_text( element_to_animate, speed):
    sentence = element_to_animate.cget("text")
    element_to_animate.configure(text="")
    element_to_animate.update()

    def animate(index):
        if index < len(sentence):
            element_to_animate.configure(text=element_to_animate.cget("text") + sentence[index])
            element_to_animate.update()
            element_to_animate.after(speed, animate, index + 1)

    animate(0)

def animate_bot_frame( frame_to_animate, speed):
    target_width = frame_to_animate.cget("width")
    target_height = frame_to_animate.cget("height")
    frame_to_animate.configure(width=0, height=0)
    current_width = 0
    current_height = 0

    while current_width < target_width or current_height < target_height:
        current_width = min(current_width + speed, target_width)
        current_height = min(current_height + speed, target_height)
        frame_to_animate.configure(width=current_width, height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)


def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    
    engine.say(text)
    engine.runAndWait()





#---------------------------------------login pages code starts here-----------------------------------------------------------------------------------------------




def login_page():
    # Toggle password visibility
    def toggle_password():
        if e2.cget("show") =="*":
            e2.configure(show="")
            peek1.configure(image=img5)
        elif e2.cget("show")=="":
            e2.configure(show="*")
            peek1.configure(image=img4, fg_color="#e0e0eb")

    
    # validation of login and password  
    def validate_login():
        global query,db,admin_stat,student_stat,teacher_stat,student_id,teacher_id
        cursor=db.cursor()
        cursor.execute(query)
        data=cursor.fetchall()
        num=len(data)
        username = e1.get()
        password = e2.get()
        if username=="" :
            flash_message("Please enter a username",mass_lab)
            #tk.messagebox.showerror("WARNING","PLEASE! ENTER A USERNAME")
            return 0
        elif password=="" :
            flash_message("Please enter a password",mass_lab)
            #tk.messagebox.showerror("WARNING","PLEASE! ENTER A PASSWORD")  
            return 0  

        access=False
        for i in range(0,num):
                
            user=data[i]
            username_data=user['username']
            password_data=user['password']
            if username == username_data and password == password_data:
                access=True
                if teacher_stat==True:
                    teacher_id=user["teacher_id"]
            
                if student_stat==True:
                    student_id=user["stud_id"]
                    
                break

        # Massagebox username and password
        if access == True:
            #Opening main pages
            tk.messagebox.showinfo("Login successful!", "Welcome back, " + username + " !")
            window.destroy()
            if teacher_stat:
                teacher_page(teacher_id)
            elif admin_stat:
                admin_page()
            else :
                student_page(student_id)
        else:
            tk.messagebox.showerror("Invalid username or password", "Please try again.")
            e2.delete(0, END)


    def clear_entry():
        if len(e1.get())!=0  or len(e2.get())!=0:
            e1.delete(0, END)
            e2.delete(0, END)


    def blind_password():
        if e2.cget("show")=="":
            e2.configure(show="*")
            peek1.configure(image=img4, fg_color="#e0e0eb")



    def change_image(direction):
        global current_image,query,admin_stat,student_stat,teacher_stat
        
        current_image += direction
        
        if current_image > 2:
            current_image = 0
        elif current_image < 0:
            current_image = 2
        
        # Change the image in the label widget
        if current_image == 0:
            b2.configure(state="normal",fg_color="#3b8ed0",text="SIGN UP")
            b1.place(x=50,y=240)
            image_label.configure(image=img1)
            clear_entry()
            student_stat=True
            admin_stat=False
            teacher_stat=False
            blind_password()
            query="select * from student"
            l1.configure(text="STUDENTS LOGIN")
            
        
        elif current_image == 1:
            b2.configure(state="disabled",fg_color="#e0e0eb",text="")
            b1.place(x=127,y=240)
            image_label.configure(image=img2)
            clear_entry() 
            student_stat=False
            admin_stat=True
            teacher_stat=False   
            blind_password()
            query="select * from admin"
            l1.configure(text="ADMIN LOGIN")
            
        else:
            b2.configure(state="disabled",fg_color="#e0e0eb",text="")
            b1.place(x=127,y=240)
            image_label.configure(image=img3)
            clear_entry() 
            admin_stat=False
            student_stat=False
            teacher_stat=True
            blind_password()
            query="select * from teacher"
            l1.configure(text="TEACHERS LOGIN")


    def sign_up():
    
        def submit_form():
            
            try:
                exist=False
                stud_id = stud_id_entry.get()
                username = user_entry.get()
                password = pass_entry.get()
                Cpass = Cpass_entry.get()
                cursor = db.cursor()
                cursor.execute("select stud_id from student")
                data_id=cursor.fetchall()

                for i in range(0,len(data_id)):
                    user=data_id[i]
                    user_id=user["stud_id"]  
                    if int(stud_id)==int(user_id):
                        exist=True
                
                if password==Cpass and username!="" and password!="" and Cpass!="" and stud_id!=""and exist==True:
                    query1 = "update student set username=%s,password=%s where stud_id=%s"
                    val = (username, password,stud_id)
                    cursor = db.cursor()
                    cursor.execute(query1, val)
                    db.commit()
                    success_label.configure(text_color='green')
                    flash_message("Registration successful",success_label)
                    
                    stud_id_entry.delete(0, END)
                    user_entry.delete(0, END)
                    pass_entry.delete(0, END)
                    Cpass_entry.delete(0, END)
                    
                elif password!=Cpass:
                    success_label.configure(text_color='red')
                    flash_message("Passwords don't match. Please check and try again.",success_label)
                elif exist==False:
                    success_label.configure(text_color='red')
                    flash_message("Invaild Sdudent ID",success_label)
                else:
                    success_label.configure(text_color='red')
                    flash_message("Please Try Again.",success_label)
                
            except:
                success_label.configure(text_color='red')
                flash_message("Try again",success_label)



        def check_strength(password):
            score = 0
            length = len(password)
            
            if length >= 8 and length <= 15:
                score += 1
            #uppercase letters
            if any(c.isupper() for c in password):
                score += 1
            #lowercase letters
            if any(c.islower() for c in password):
                score += 1
            #check digits
            if any(c.isdigit() for c in password):
                score += 1
            #check symbols
            if any(c in "!@#$%^&*()_+{}[];:'\"<>,.?/|\\" for c in password):
                score += 1
            
            return score

        def show_strength(event):
            password = pass_entry.get()
            strength = check_strength(password)
            if strength < 2:
                strength_label.configure(text="Weak",text_color="red")
            elif strength < 4:
                strength_label.configure(text="Moderate",text_color="#ff9900")
            else:
                strength_label.configure(text="Strong",text_color="green")
        
        signup_win = CTk()
        signup_win.title("SIGN UP FORM")
        signup_win.iconbitmap("logo_icon.ico")
        window_width = 390
        window_height = 470
        signup_win.geometry(f"{window_width}x{window_height}")


        screen_width = signup_win.winfo_screenwidth()
        screen_height = signup_win.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        signup_win.geometry(f"+{x}+{y}")


        screen=CTkFrame(signup_win,width=390,height=470,fg_color="#c2c2d6")
        screen.place(relx=0.5,rely=0.5,anchor=CENTER)

        # create header label
        header_label = CTkLabel(screen, text="SIGN UP FORM", font=("Arial", 18))
        header_label.place(relx=0.5,y=30,anchor=CENTER)

        # create input fields
        stud_id_label = CTkLabel(screen, text="Student_ID :", font=("Arial", 12))
        stud_id_label.place(x=100, y=100,anchor=CENTER)
        stud_id_entry = CTkEntry(screen, font=("Arial", 12))
        stud_id_entry.place(x=250, y=100,anchor=CENTER)

        user_label = CTkLabel(screen, text="Username :", font=("Arial", 12))
        user_label.place(x=100, y=150,anchor=CENTER)
        user_entry = CTkEntry(screen, font=("Arial", 12))
        user_entry.place(x=250, y=150,anchor=CENTER)
        
        strength_label=CTkLabel(screen,text="",width=130,height=11,bg_color="#c2c2d6")
        strength_label.place(x=250,y=224,anchor=CENTER)

        pass_lb = CTkLabel(screen, text="Password :", font=("Arial", 12))
        pass_lb.place(x=100, y=200,anchor=CENTER)
        pass_entry = CTkEntry(screen, font=("Arial", 12))
        pass_entry.place(x=250, y=200,anchor=CENTER)
        pass_entry.bind("<KeyRelease>", show_strength)
        

        Cpass_lb = CTkLabel(screen, text="Confirm Password :", font=("Arial", 12))
        Cpass_lb.place(x=100, y=250,anchor=CENTER)
        Cpass_entry = CTkEntry(screen,show="*",font=("Arial", 12))
        Cpass_entry.place(x=250, y=250,anchor=CENTER)

        # create submit button
        submit_button = CTkButton(screen, text="Submit", command=submit_form, font=("Arial", 12))
        submit_button.place(relx=0.5, y=360,anchor=CENTER)

        success_label = CTkLabel(screen, text="",text_color="green" ,font=("Arial", 12))
        success_label.place(relx=0.5,y=430,anchor=CENTER)
        
        signup_win.mainloop()

    set_appearance_mode("light")
    set_default_color_theme("blue")
    window=CTk()
    screen_width = window.winfo_screenwidth()
    screen_height= window.winfo_screenheight()
    window_width = screen_width
    window_height = screen_height
    window.geometry(f"{window_width}x{window_height}")
    window.iconbitmap("logo_icon.ico")
    window.geometry("+0+0")
    # window.maxsize(width=1400,height=750)
    # window.minsize(width=1400,height=750)
    window.attributes('-fullscreen',True)
    window.iconbitmap("logo_icon.ico")
    window.title("Login page")

    frame= CTkFrame(window,fg_color="white",width=1400,height=750)
    frame.place(x=0,y=0)

    img1 = CTkImage(Image.open("student.png"),size=(600,600))
    img2 = CTkImage(Image.open("admin.png"),size=(600,600))
    img3 = CTkImage(Image.open("teacher.png"),size=(600,600))

    img4=CTkImage(Image.open("closed_eye.png"),size=(27,27))
    img5=CTkImage(Image.open("open_eye.png"),size=(30,30))

    img6=CTkImage(Image.open("left_arrow.png"),size=(20,20))
    img7=CTkImage(Image.open("right_arrow.png"),size=(20,20))

    # label to display the images
    image_label = CTkLabel(frame,text="", image=img1)
    image_label.place(x=400,rely=0.5,anchor=CENTER)


    frame1=CTkFrame(frame,width=330,height=390,fg_color="#e0e0eb",corner_radius=20)
    frame1.place(relx=0.78,rely=0.48,anchor=tk.CENTER)
    l1=CTkLabel(frame1,text="STUDENTS LOGIN",font=("Century Gothic",20))
    l1.place(x=160,y=60,anchor=CENTER)

    e1=CTkEntry(frame1,height=30,width=220,placeholder_text="Username",font=("Microsoft YaHei UI light",15))
    e1.place(x=50,y=110)
    e2=CTkEntry(frame1,show = "*",height=30,width=220,placeholder_text="Password",font=("Microsoft YaHei UI light",15))
    e2.place(x=50,y=165)
    
    b2=CTkButton(frame1,width=80,text="SIGN UP",cursor="hand2",font=("Microsoft YaHei UI light",15),command=sign_up)
    b2.place(x=190,y=240)

    b1=CTkButton(frame1,width=80,text="LOGIN",cursor="hand2",font=("Microsoft YaHei UI light",15),command=validate_login)
    b1.place(x=50,y=240)
    window.bind('<Return>', lambda event=None: b1.invoke())
    
    b3=CTkLabel(frame1,width=80,text="Forget Password ?",text_color="blue", cursor="hand2",font=("Microsoft YaHei UI light",12))
    b3.place(x=50,y=293)
    peek1= CTkButton(frame1,width=20, text="",fg_color="#e0e0eb",image=img4,hover_color="#e0e0eb",font=("Arial", 12),
                    cursor="hand2", bg_color="#e0e0eb", command=toggle_password)
    peek1.place(x=270,y=160)

    mass_lab=tk.Label(frame1, text="",bg="#e0e0eb",fg="red",font=("Microsoft YaHei UI light",11))
    mass_lab.place(x=160,y=370,anchor=CENTER)

    prev_button = CTkButton(frame1,width=6,text="",fg_color="#e0e0eb" ,image=img6,hover_color="#c2c2d6",command=lambda: change_image(-1))
    prev_button.place(x=10,y=45)
    window.bind('<Left>', lambda event=None: prev_button.invoke())
    next_button = CTkButton(frame1,width=6, text="",fg_color="#e0e0eb",image=img7,hover_color="#c2c2d6",command=lambda: change_image(1))
    next_button.place(x=280,y=45)
    window.bind('<Right>', lambda event=None: next_button.invoke())
    window.mainloop()   





#-------------------------------------------Admin_pages code starts here-------------------------------------------------------------








def admin_page():

    #datetime
    def date_time_display():
            
        def ct_time():
            now = datetime.now()
            ct_string = now.strftime("%H:%M:%S")
            return ct_string

        def ct_change():
            ct_string = ct_time()
            time_lb.configure(text=ct_string)
            f0.after(1000, ct_change)  # update every 1 second
        #logout_frame
        def logout_frame():
            delete_frames()
            date_time_display()
            def destroy_window():
                admin_win.destroy()
                login_page()
            #asking to logout
            log_lb=CTkLabel(f0,text="Do you want to logout ?",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
            log_lb.place(relx=0.5,y=200,anchor=CENTER)
            log_bu=CTkButton(f0,text="Yes",height=45,command=destroy_window,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            log_bu.place(relx=0.5,y=300,anchor=CENTER)

        today = datetime.today()
        t_date= today.strftime("%B %d, %Y")
        #date and time 
        d_f=CTkFrame(f0,width=350,height=50,border_color="black",border_width=3,fg_color="white",corner_radius=40)
        d_f.place(x=730,y=5)
        time_lb=CTkLabel(d_f,width=110,height=30,text="",font=CTkFont("Helvetica",19),fg_color="white",corner_radius=40,text_color="black")
        time_lb.place(relx=0.8,rely=0.5,anchor=CENTER)
        date_lb=CTkLabel(d_f,text=t_date,width=150,height=30,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="white")
        date_lb.place(relx=0.3,rely=0.5,anchor=CENTER)
        ct_change()

        photo1=CTkImage(Image.open("logout1.png"),size=(50,50))
        logout_b=CTkButton(f0,image=photo1,command=logout_frame,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
        logout_b.place(x=1100,y=0)

    #to indicate
    def indicate(lb,frame):
        hide_indicators()
        lb.configure(fg_color="#0066ff")
        delete_frames()
        frame()

    #for removing attendence
    def delete_attendance_record(student_id):
        #data access from database
        cursor=db.cursor()
        cursor.execute("Select * from student")
        s_data=cursor.fetchall()
        id=s_data[-1]["stud_id"]
        
        if int(student_id)>100 and int(student_id)<=int(id):
            cursor=db.cursor()
            cursor.execute("Select * from student where stud_id=%s",(student_id,))
            s_data=cursor.fetchall()
            if len(s_data)!=0:
                data1=s_data[0]
                Std=data1.get("std_code")
                
                cursor = db.cursor()
                cursor.execute("select count(*) from student where std_code=%s and stud_id<%s", (Std, student_id))
                s_data = cursor.fetchall()

                roll_index = s_data[0]["count(*)"]       
                cursor.execute("select attendance_date,attend_sheet from attendance where stand_code=%s",(Std))
                data_sheet=cursor.fetchall()
                new_sheet=""
                for i in range(len(data_sheet)):
                    date=data_sheet[i].get("attendance_date")
                    sheet=data_sheet[i].get("attend_sheet")
                    new_sheet=sheet[:roll_index]+sheet[int(roll_index)+1:]

                    cursor = db.cursor()
                    cursor.execute(" update attendance set attend_sheet=%s where attendance_date=%s and stand_code=%s", (new_sheet,date,Std))
                    db.commit()

    #def for removing record
    def remove(entry,table,id,flash):
        entry=entry.get()
        flag_alpha=True
        for i in entry:
                if i.isalpha():
                    flag_alpha=False
        if len(entry)==0:
            flash.configure(text_color="red")
            flash_message("Enter a "+table+" id",flash)
        elif flag_alpha==False:
            flash.configure(text_color="red")
            flash_message("Invalid "+table+" id",flash)
        else:
            try:
                cursor=db.cursor()
                cursor.execute("Select "+id+" from "+table+" where "+id+"=%s",(entry))
                data=cursor.fetchall()
                if id=="stud_id":
                    cursor.execute("delete from complain where stud_code=%s",(entry))
                    db.commit()
                    cursor.execute("delete from grade where stud_code=%s",(entry))
                    db.commit()
                    delete_attendance_record(entry)
                if len(data)!=0:
                    cursor.execute("delete from "+table+" where "+id+"=%s",(entry))
                    db.commit()
                    flash.configure(text_color="green")
                    flash_message(table+" Removed Successfully",flash)
                else:
                    flash.configure(text_color="red")
                    flash_message(table+" doesn't exist of the given id",flash)
            except pymysql.Error:
                flash.configure(text_color="red")
                flash_message("Teacher assinged to class",flash)

    #def for validating dob
    def validate_dob(year,month,date,dob,flag_dob):
        flag_alpha=True
        for i in dob:
            if i.isalpha():
                flag_alpha=False
                flag_dob=False
        if flag_alpha==True:
            year=int(year)
            month=int(month)
            date=int(date)
            if len(dob)==10 and year<=2020 :
                    if month<=12:
                        if month in (1,3,5,7,8,10,12):
                            if 1<=date<=31:
                                flag_dob=True
                            else:
                                flag_dob=False
                        elif month in (4,6,9,11):
                            if 1<=date<=30:
                                flag_dob=True
                            else:
                                flag_dob=False
                        elif month==2:
                            if year%4==0:
                                if 1<=date<=29:
                                    flag_dob=True
                                else:
                                    flag_dob=False
                            else:
                                if 1<=date<=28:
                                    flag_dob=True
                                else:
                                    flag_dob=False
                    else:
                        flag_dob=False
            else:
                flag_dob=False
        return flag_dob

    #hide indicators
    def hide_indicators():
        home_indicate.configure(fg_color="white")
        student_indicate.configure(fg_color="white")
        teacher_indicate.configure(fg_color="white")
        timetable_indicate.configure(fg_color="white")
        complain_indicate.configure(fg_color="white")

    #to delete frames
    def delete_frames():
        for f in f0.winfo_children():
            f.destroy()



    #---------------------------------------------------------------Home Frame----------------------------------------------------------------



    #home_frame
    def home_frame():
        global departments
        date_time_display()
        #welcome label
        l2=CTkLabel(f0,text="Welcome Admin!",width=50,font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=0)

        #no of students
        cursor=db.cursor()
        cursor.execute("Select count(*) count from student")
        data=cursor.fetchall()
        d=data[0]
        tot_st=d["count"]
        f3=CTkFrame(f0,width=240,height=130,fg_color="#FFFFCC",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        st_no=CTkLabel(f3,text=tot_st,font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        st_no.place(x=20,y=10)
        t_st=CTkLabel(f3,text="Total students",font=CTkFont(family="Helvetica",weight="bold",size=25),text_color="black")
        t_st.place(x=20,y=90)

        #no of teachers
        cursor=db.cursor()
        cursor.execute("Select count(*) count from teacher")
        data=cursor.fetchall()
        d=data[0]
        tot_te=d["count"]
        f4=CTkFrame(f0,width=240,height=130,fg_color="#C7FAC7",border_width=3,corner_radius=12,border_color="black")
        f4.place(x=40,y=245)
        t_no=CTkLabel(f4,text=tot_te,font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        t_no.place(x=20,y=10)
        t_te=CTkLabel(f4,text="Total Teachers",font=CTkFont(family="Helvetica",weight="bold",size=25),text_color="black")
        t_te.place(x=20,y=90)

        #Pending Complains
        departments=[]
        cursor=db.cursor()
        cursor.execute("SELECT distinct(depart) FROM complain WHERE solution IS NULL and `to`=%s",("Admin"))
        data=cursor.fetchall()
        f5=CTkScrollableFrame(f0,width=210,height=245,fg_color="#F2C6C6",border_width=3,corner_radius=12,border_color="black")
        f5.place(x=40,y=390)
        pen_lb=CTkLabel(f5,fg_color="#dac292",height=40,width=150,corner_radius=20,text="Pending",font=CTkFont(family="Helvetica",weight="bold",size=20),text_color="black")
        pen_lb.grid(row=0,column=1,padx=10)
        for i in range(len(data)):
            new=data[i]
            departments.append(new['depart'])

        if len(departments)==0:
            new_b=CTkLabel(f5,text="No Complains",height=45,width=180,corner_radius=20,text_color="black",fg_color="#F2C6C6",font=CTkFont(family="Helvetica",weight="bold",size=20))
            new_b.grid(row=3,column=1,padx=10,pady=10)

        r=1
        y_pad=2
        for i in departments:
            new_b=CTkButton(f5,text=i,height=45,width=180,border_width=2,border_color="black",corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            new_b.configure(command=lambda new=new_b: indicate(complain_indicate,complain_frame))
            new_b.grid(row=r,column=1,padx=10,pady=y_pad+7)
            r+=1

        #recent students
        f6=CTkFrame(f0,height=305,width=885,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f6.place(x=300,y=100)
        animate_frame(f6)
        rec_stu_lb=CTkLabel(f0,text="Recently added students",font=CTkFont(family="Helvetica",size=20),text_color="white")
        rec_stu_lb.place(x=940,y=70)
        animate_text(rec_stu_lb,20)
        
        def treeview():
            cursor=db.cursor()
            cursor.execute("Select stud_id,name,gen,std_code,dob,phone_no from Student order by(stud_id)desc limit 5")
            data=cursor.fetchall()
            stud_id=[]
            name=[]
            gen=[]
            std=[]
            dob=[]
            mobile=[]

            for i in data:
                    id=i.get("stud_id")
                    na=i.get("name")
                    ge=i.get("gen")
                    st=i.get("std_code")
                    bir=i.get("dob")
                    mob=i.get("phone_no")
                    stud_id.append(id)
                    name.append(na)
                    gen.append(ge)
                    std.append(st)
                    dob.append(bir)
                    mobile.append(mob)

            stu_table=ttk.Treeview(f6,columns=("stu_id","name","gen","std","dob","mobile"),show="headings",height=5)
            style=ttk.Style(f6)
        
            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#dac292",fieldbackground="#dac292", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("name",text="Name")
            stu_table.heading("stu_id",text="Student id")
            stu_table.heading("gen",text="Gender")
            stu_table.heading("std",text="Class")
            stu_table.heading("dob",text="Date of Birth")
            stu_table.heading("mobile",text="Mobile No")
            stu_table.column("stu_id",width=100,anchor=CENTER)
            stu_table.column("name",width=200,anchor=CENTER)
            stu_table.column("gen",width=100,anchor=CENTER)
            stu_table.column("std",width=130,anchor=CENTER)
            stu_table.column("dob",width=150,anchor=CENTER)
            stu_table.column("mobile",width=180,anchor=CENTER)

            for i in range(len(stud_id)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(stud_id[i],name[i],gen[i],std[i],dob[i],mobile[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)
        treeview()

        #recent teachers
        f6=CTkFrame(f0,height=205,width=885,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f6.place(x=300,y=460)
        animate_frame(f6)
        rec_stu_lb=CTkLabel(f0,text="Recently added Teahcers",font=CTkFont(family="Helvetica",size=20),text_color="white")
        rec_stu_lb.place(x=940,y=430)
        animate_text(rec_stu_lb,20)
        
        def treeview():
            cursor=db.cursor()
            cursor.execute("Select teacher_id,name,gen,quali,dob from teacher order by(teacher_id) desc limit 3")
            data=cursor.fetchall()
            stud_id=[]
            name=[]
            gen=[]
            std=[]
            dob=[]

            for i in data:
                    id=i.get("teacher_id")
                    na=i.get("name")
                    ge=i.get("gen")
                    st=i.get("quali")
                    bir=i.get("dob")
                    stud_id.append(id)
                    name.append(na)
                    gen.append(ge)
                    std.append(st)
                    dob.append(bir)

            stu_table=ttk.Treeview(f6,columns=("t_id","name","gen","quali","dob"),show="headings",height=3)
            style=ttk.Style(f6)
        
            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#dac292",fieldbackground="#dac292", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("name",text="Name")
            stu_table.heading("t_id",text="Teahcer id")
            stu_table.heading("gen",text="Gender")
            stu_table.heading("quali",text="Qualification")
            stu_table.heading("dob",text="Date of Birth")
            stu_table.column("t_id",width=100,anchor=CENTER)
            stu_table.column("name",width=250,anchor=CENTER)
            stu_table.column("gen",width=100,anchor=CENTER)
            stu_table.column("quali",width=200,anchor=CENTER)
            stu_table.column("dob",width=210,anchor=CENTER)

            for i in range(len(stud_id)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(stud_id[i],name[i],gen[i],std[i],dob[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)
        treeview()



    #----------------------------------------------------------------Student Frame----------------------------------------------------------------



    #student_frame
    def student_frame():
        date_time_display()

        #student section
        l2=CTkLabel(f0,text="Student's details",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)

        #frame for buttons 
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        animate_frame(f3)
        
        def treeview():
            f9=CTkFrame(f0,width=875,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            f9.place(x=310,y=100)
            animate_frame(f9)
            cursor=db.cursor()
            cursor.execute("Select stud_id,name,gen,std_code,dob,phone_no from Student")
            data=cursor.fetchall()
            stud_id=[]
            name=[]
            gen=[]
            std=[]
            dob=[]
            mobile=[]

            for i in data:
                    id=i.get("stud_id")
                    na=i.get("name")
                    ge=i.get("gen")
                    st=i.get("std_code")
                    bir=i.get("dob")
                    mob=i.get("phone_no")
                    stud_id.append(id)
                    name.append(na)
                    gen.append(ge)
                    std.append(st)
                    dob.append(bir)
                    mobile.append(mob)
            global stu_table 
            stu_table=ttk.Treeview(f9,columns=("stu_id","name","gen","std","dob","mobile"),show="headings")
            style=ttk.Style(f9)
        
            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("name",text="Name")
            stu_table.heading("stu_id",text="Student id")
            stu_table.heading("gen",text="Gender")
            stu_table.heading("std",text="Class")
            stu_table.heading("dob",text="Date of Birth")
            stu_table.heading("mobile",text="Mobile No")
            stu_table.column("stu_id",width=100,anchor=CENTER)
            stu_table.column("name",width=200,anchor=CENTER)
            stu_table.column("gen",width=100,anchor=CENTER)
            stu_table.column("std",width=100,anchor=CENTER)
            stu_table.column("dob",width=150,anchor=CENTER)
            stu_table.column("mobile",width=200,anchor=CENTER)

            for i in range(len(stud_id)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(stud_id[i],name[i],gen[i],std[i],dob[i],mobile[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)
        

        def stu_hide_hover():
            vie_stu.configure(fg_color="#33CCFF")
            add_stu.configure(fg_color="#33CCFF")
            up_stu.configure(fg_color="#33CCFF")
            del_stu.configure(fg_color="#33CCFF")
            
        #view students
        def vie_stud():
            stu_hide_hover()
            vie_stu.configure(fg_color="#888888")
            treeview()

        vie_stu=CTkButton(f3,hover_color="#D9D9D0",command=vie_stud,text="View Students",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        vie_stu.place(x=120,y=45,anchor=CENTER)
        animate_text(vie_stu,20)


        #add student
        def add_stud():
            stu_hide_hover()
            add_stu.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)
            animate_frame(f8)

            main_lb=CTkLabel(f8,text="Student Admission",text_color="black",font=CTkFont("Helvetica",30))
            main_lb.place(relx=0.5,y=40,anchor=CENTER)
            animate_text(main_lb,20)

            id_lb=CTkLabel(f8,text="Student id",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.1,y=120,anchor=CENTER)

            id_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            id_et.place(relx=0.3,y=125,anchor=CENTER)

            cursor=db.cursor()
            cursor.execute("select max(stud_id)+1 from student")
            data=cursor.fetchall()
            auto=data[0]
            auto_id=auto.get("max(stud_id)+1")
            id_et.insert(0,auto_id)
            id_et.configure(state="disabled")
            mobi_lb=CTkLabel(f8,text="Mobile No",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            mobi_lb.place(relx=0.6,y=120,anchor=CENTER)

            mobi_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            mobi_et.place(relx=0.8,y=125,anchor=CENTER)

            name_lb=CTkLabel(f8,text="Name",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            name_lb.place(relx=0.1,y=190,anchor=CENTER)

            name_et=CTkEntry(f8,corner_radius=30,height=30,width=200,border_width=2,font=("Roboto",12))
            name_et.place(relx=0.3,y=195,anchor=CENTER)

            add_lb=CTkLabel(f8,text="Address",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            add_lb.place(relx=0.6,y=190,anchor=CENTER)

            add_et=CTkTextbox(f8,corner_radius=10,height=100,width=200,border_width=2,font=("Roboto",15))
            add_et.place(relx=0.8,y=220,anchor=CENTER)

            gen_lb=CTkLabel(f8,text="Gender",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            gen_lb.place(relx=0.1,y=250,anchor=CENTER)

            gender=StringVar()
            gen_op1=CTkRadioButton(f8,text="Male",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="M")
            gen_op1.place(relx=0.25,y=255,anchor=CENTER)

            gen_op2=CTkRadioButton(f8,text="Female",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="F")
            gen_op2.place(relx=0.35,y=255,anchor=CENTER)

            gen_op3=CTkRadioButton(f8,text="Others",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="O")
            gen_op3.place(relx=0.47,y=255,anchor=CENTER)

            std_lb=CTkLabel(f8,text="Standard",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            std_lb.place(relx=0.1,y=390,anchor=CENTER)

            std_op=CTkOptionMenu(f8,width=180,values=["5th std","6th std","7th std","8th std","9th std","10th std"])
            std_op.place(relx=0.3,y=390,anchor=CENTER)

            date_lb=CTkLabel(f8,text="Date of Birth",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            date_lb.place(relx=0.1,y=310,anchor=CENTER)

            date_et=CTkEntry(f8,placeholder_text="YYYY-MM-DD",height=30,width=200,border_width=2,font=("Roboto",12))
            date_et.place(relx=0.3,y=315,anchor=CENTER)

            flash_massa=CTkLabel(f8,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#96DED1")
            flash_massa.place(relx=0.52,y=510,anchor=CENTER)

            #def for sending sms to student
            def send_sms_student(target_no,name,id):
                client=Client("<Account SID>","<Auth Token>")
                student_id=str(id)
                message=client.messages.create(
                body=("-\n\n🎉 Congratulations!\n\nDear "+name+",\n\nWe are thrilled to inform you that your student ID has been generated.\n\nStudent id: "+student_id+"\n\nYou can use this student id to sign up and create your username and password\n\nWishing you a fantastic start to your educational endeavors!\n\nBest Regards,\nTeam Edutrack"),
                from_="<My Twilio phone number>",
                to=target_no
                )
            
            def submit():
                global db
                id=id_et.get()
                name=name_et.get()
                gen=gender.get()
                std=std_op.get()
                dob=date_et.get()
                phone=mobi_et.get()
                phone_="+91"+phone
                add=add_et.get("1.0", "end-1c")
                flag_check=True
                if len(gen)==0 or len(phone)==0 or len(dob)==0 or len(name)==0 or len(add)==0 or len(id)==0:
                    flash_massa.configure(text_color="red")
                    flash_message("Try Again",flash_massa)
                    flag_check=False
                else:
                    cursor=db.cursor()
                    flag_ph=True
                    flag_nm=True
                    flag_dob=True
                    if len(phone)!=10 or not(phone.isdigit()) or not(phone[0] in ["9","8","7"]):
                        flag_ph=False
                    for i in name:
                        if i.isnumeric():
                            flag_nm=False
                            break
                    special_characters = "!@#$%^&*()_+}~`-=;'/.,<>?|"
                    if len(name)==0 or any(char in special_characters for char in name):
                        flag_nm=False
                    year=dob[0:4]
                    date=dob[8:10]
                    month=dob[5:7]
                    flag_dob=validate_dob(year,month,date,dob,flag_dob)
                    if flag_check==True and flag_ph==True and flag_nm==True and flag_dob==True:
                        cursor.execute("insert into student(stud_id,name,gen,std_code,dob,phone_no,address) values(%s,%s,%s,%s,%s,%s,%s)",(id,name,gen,std,dob,phone,add))
                        db.commit()
                        flash_massa.configure(text_color="#04C34D")
                        flash_message("Student Added Successfully",flash_massa)
                        send_sms_student(phone_,name,id)
                    elif flag_nm==False:
                        flash_massa.configure(text_color="red")
                        flash_message("Invalid Name",flash_massa)
                    elif flag_dob==False:
                        flash_massa.configure(text_color="red")
                        flash_message("Invalid Date of Birth",flash_massa)
                    elif flag_ph==False:
                        flash_massa.configure(text_color="red")
                        flash_message("Invalid Phone No",flash_massa)


            sub_bt=CTkButton(f8,text="Submit",command=submit,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            sub_bt.place(relx=0.52,y=470,anchor=CENTER)

        add_stu=CTkButton(f3,hover_color="#D9D9D0",command=add_stud,text="Add student",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        add_stu.place(relx=0.5,y=120,anchor=CENTER)
        animate_text(add_stu,20)

        #remove student
        def del_stud():
            stu_hide_hover()
            del_stu.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)

            main_lb=CTkLabel(f8,text="Enter the detail to remove student",text_color="black",font=CTkFont("Helvetica",35))
            main_lb.place(relx=0.5,rely=0.3,anchor=CENTER)

            id_lb=CTkLabel(f8,text="Student id",width=120,height=45,corner_radius=8,font=("Helvetica",23),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.41,rely=0.44,anchor=CENTER)

            id_et=CTkEntry(f8,justify=CENTER,height=45,width=170,corner_radius=30,border_width=2,font=("Roboto",20))
            id_et.place(relx=0.58,rely=0.44,anchor=CENTER)

            flash_massa=CTkLabel(f8,fg_color="#96DED1",bg_color="#96DED1",text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20))
            flash_massa.place(relx=0.5,rely=0.7,anchor=CENTER)

            del_bt=CTkButton(f8,text="Remove",command=lambda: remove(id_et,"Student","stud_id",flash_massa),height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            del_bt.place(relx=0.5,rely=0.6,anchor=CENTER)

        del_stu=CTkButton(f3,hover_color="#D9D9D0",command=del_stud,text="Remove student",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        del_stu.place(x=120,y=195,anchor=CENTER)

        #update student
        def up_stud():
            stu_hide_hover()
            up_stu.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)
            animate_frame(f8)

            main_lb=CTkLabel(f8,text="Update student",text_color="black",font=CTkFont("Helvetica",30))
            main_lb.place(relx=0.5,y=30,anchor=CENTER)
            animate_text(main_lb,20)

            id_lb=CTkLabel(f8,text="Student id",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.1,y=120,anchor=CENTER)

            id_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            id_et.place(relx=0.3,y=125,anchor=CENTER)

            def fetch():
                stud_id=id_et.get()
                flag_check=True
                if len(stud_id)==0:
                    flash_massa.configure(text_color="red")
                    flash_message("Try Again",flash_massa)
                    flag_check=False
                if flag_check==True:
                    cursor=db.cursor()
                    cursor.execute("Select name,gen,std_code,dob,phone_no,address from student where stud_id=%s",(stud_id))
                    data=cursor.fetchall()
                    if len(data)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Student of given id doesn't exist",flash_massa)
                    if len(data)!=0:   
                        detail=data[0]
                        name=detail.get("name")
                        gen=detail.get('gen')
                        std_code=detail.get('std_code')
                        dob=detail.get('dob')
                        phone=detail.get("phone_no")
                        add=detail.get("address")
                        #wipping entry
                        mobi_et.delete(0,END)
                        name_et.delete(0,END)
                        add_et.delete(1.0,END)
                        date_et.delete(0,END)
                        #inserting entries
                        mobi_et.insert(0,phone)
                        name_et.insert(0,name)
                        add_et.insert(INSERT,add)
                        date_et.insert(0,dob)
                        gender.set(gen)
                        std_op.set(std_code)
                
            fet_bt=CTkButton(f8,text="fetch",command=fetch,height=20,width=60,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            fet_bt.place(relx=0.47,y=125,anchor=CENTER)

            mobi_lb=CTkLabel(f8,text="Mobile No",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            mobi_lb.place(relx=0.6,y=120,anchor=CENTER)

            mobi_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            mobi_et.place(relx=0.8,y=125,anchor=CENTER)

            name_lb=CTkLabel(f8,text="Name",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            name_lb.place(relx=0.1,y=190,anchor=CENTER)

            name_et=CTkEntry(f8,corner_radius=30,height=30,width=200,border_width=2,font=("Roboto",12))
            name_et.place(relx=0.3,y=195,anchor=CENTER)

            add_lb=CTkLabel(f8,text="Address",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            add_lb.place(relx=0.6,y=190,anchor=CENTER)

            add_et=CTkTextbox(f8,corner_radius=10,height=100,width=200,border_width=2,font=("Roboto",12))
            add_et.place(relx=0.8,y=220,anchor=CENTER)

            gen_lb=CTkLabel(f8,text="Gender",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            gen_lb.place(relx=0.1,y=250,anchor=CENTER)

            gender=StringVar()
            gen_op1=CTkRadioButton(f8,text="Male",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="M")
            gen_op1.place(relx=0.25,y=255,anchor=CENTER)

            gen_op2=CTkRadioButton(f8,text="Female",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="F")
            gen_op2.place(relx=0.35,y=255,anchor=CENTER)

            gen_op3=CTkRadioButton(f8,text="Others",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="O")
            gen_op3.place(relx=0.47,y=255,anchor=CENTER)

            std_lb=CTkLabel(f8,text="Standard",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            std_lb.place(relx=0.1,y=310,anchor=CENTER)

            std_op=CTkOptionMenu(f8,width=180,values=["5th std","6th std","7th std","8th std","9th std","10th std"])
            std_op.place(relx=0.3,y=315,anchor=CENTER)

            date_lb=CTkLabel(f8,text="Date of Birth",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            date_lb.place(relx=0.1,y=390,anchor=CENTER)

            date_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            date_et.place(relx=0.3,y=390,anchor=CENTER)

            flash_massa=CTkLabel(f8,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#96DED1")
            flash_massa.place(relx=0.5,y=510,anchor=CENTER)
            
            def update():
                    global db
                    id=id_et.get()
                    name=name_et.get()
                    gen=gender.get()
                    std=std_op.get()
                    dob=date_et.get()
                    phone=mobi_et.get()
                    add=add_et.get("1.0", "end-1c")
                    if len(gen)==0 or len(phone)==0 or len(dob)==0 or len(name)==0 or len(add)==0 or len(id)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Try Again",flash_massa)
                    else:
                        cursor=db.cursor()
                        flag_ph=True
                        flag_nm=True
                        flag_dob=True
                        if  len(phone)!=10 or not(phone.isdigit()) or not(phone[0] in ["9","8","7"]):
                            flag_ph=False
                        for i in name :
                            if i.isnumeric():
                                flag_nm=False
                                break
                        special_characters = "!@#$%^&*()_+}~`-=;'/.,<>?|"
                        if len(name)==0 or any(char in special_characters for char in name):
                            flag_nm=False
                        year=dob[0:4]
                        date=dob[8:10]
                        month=dob[5:7]
                        flag_dob=validate_dob(year,month,date,dob,flag_dob)
                        if flag_ph==True and flag_nm==True and flag_dob==True:
                            cursor.execute("Update student set name=%s,gen=%s,std_code=%s,dob=%s,phone_no=%s,address=%s where stud_id=%s",(name,gen,std,dob,phone,add,id))
                            db.commit()
                            flash_massa.configure(text_color="#04C34D")
                            flash_message("Student Updated Successfully",flash_massa)
                        elif flag_nm==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Name",flash_massa)
                        elif flag_dob==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Date of Birth",flash_massa)
                        elif flag_ph==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Phone No",flash_massa)
                        

            up_bt=CTkButton(f8,text="Update",command=update,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            up_bt.place(x=440,y=470,anchor=CENTER)

        up_stu=CTkButton(f3,hover_color="#D9D9D0",command=up_stud,text="Update student",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        up_stu.place(x=120,y=270,anchor=CENTER)
        animate_text(up_stu,20)
        vie_stu.invoke()



    #----------------------------------------------------------------Teacher Frame----------------------------------------------------------------


    #teacher_frame
    def teacher_frame():
        date_time_display()

        #teacher_section
        l2=CTkLabel(f0,text="Teacher's details",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        animate_frame(l2)

        #frame for buttons 
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        animate_frame(f3)

        def teach_hide_hover():
            vie_te.configure(fg_color="#33CCFF")
            add_t.configure(fg_color="#33CCFF")
            up_t.configure(fg_color="#33CCFF")
            del_t.configure(fg_color="#33CCFF")

        def treeview_teacher():
            f9=CTkFrame(f0,width=875,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            f9.place(x=310,y=100)
            animate_frame(f9)
            cursor=db.cursor()
            cursor.execute("Select teacher_id,name,gen,quali,dob from teacher")
            data=cursor.fetchall()
            stud_id=[]
            name=[]
            gen=[]
            std=[]
            dob=[]

            for i in data:
                    id=i.get("teacher_id")
                    na=i.get("name")
                    ge=i.get("gen")
                    st=i.get("quali")
                    bir=i.get("dob")
                    stud_id.append(id)
                    name.append(na)
                    gen.append(ge)
                    std.append(st)
                    dob.append(bir)

            stu_table=ttk.Treeview(f9,columns=("t_id","name","gen","quali","dob"),show="headings")
            style=ttk.Style(f9)
        

            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("name",text="Name")
            stu_table.heading("t_id",text="Teahcer id")
            stu_table.heading("gen",text="Gender")
            stu_table.heading("quali",text="Qualification")
            stu_table.heading("dob",text="Date of Birth")
            stu_table.column("t_id",width=100,anchor=CENTER)
            stu_table.column("name",width=250,anchor=CENTER)
            stu_table.column("gen",width=130,anchor=CENTER)
            stu_table.column("quali",width=180,anchor=CENTER)
            stu_table.column("dob",width=190,anchor=CENTER)

            for i in range(len(stud_id)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(stud_id[i],name[i],gen[i],std[i],dob[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)

        #view teachers
        def view_teach():
            teach_hide_hover()
            vie_te.configure(fg_color="#888888")
            treeview_teacher()

        vie_te=CTkButton(f3,text="View Teachers",hover_color="#D9D9D0",command=view_teach,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        vie_te.place(x=120,y=45,anchor=CENTER)
        animate_text(vie_te,20)

        #add teacher
        def add_teach():
            teach_hide_hover()
            add_t.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)
            animate_frame(f8)
            main_lb=CTkLabel(f8,text="Add Teacher's details",text_color="black",font=CTkFont("Helvetica",30))
            main_lb.place(relx=0.5,y=40,anchor=CENTER)
            animate_text(main_lb,20)
            id_lb=CTkLabel(f8,text="Teacher id",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.1,y=120,anchor=CENTER)

            id_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            id_et.place(relx=0.3,y=125,anchor=CENTER)

            cursor=db.cursor()
            cursor.execute("select max(teacher_id)+1 from teacher")
            data=cursor.fetchall()
            auto=data[0]
            auto_id=auto.get("max(teacher_id)+1")
            id_et.insert(0,auto_id)
            id_et.configure(state="disabled")
            mobi_lb=CTkLabel(f8,text="Mobile No",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            mobi_lb.place(relx=0.6,y=120,anchor=CENTER)

            mobi_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            mobi_et.place(relx=0.8,y=125,anchor=CENTER)

            name_lb=CTkLabel(f8,text="Name",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            name_lb.place(relx=0.1,y=190,anchor=CENTER)

            name_et=CTkEntry(f8,corner_radius=30,height=30,width=200,border_width=2,font=("Roboto",12))
            name_et.place(relx=0.3,y=195,anchor=CENTER)

            add_lb=CTkLabel(f8,text="Address",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            add_lb.place(relx=0.6,y=190,anchor=CENTER)

            add_et=CTkTextbox(f8,corner_radius=10,height=70,width=200,border_width=2,font=("Roboto",12))
            add_et.place(relx=0.8,y=200,anchor=CENTER)

            user_lb=CTkLabel(f8,text="Username",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            user_lb.place(relx=0.6,y=280,anchor=CENTER)

            user_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            user_et.place(relx=0.8,y=280,anchor=CENTER)

            pass_lb=CTkLabel(f8,text="Password",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            pass_lb.place(relx=0.6,y=360,anchor=CENTER)

            pass_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            pass_et.place(relx=0.8,y=360,anchor=CENTER)

            gen_lb=CTkLabel(f8,text="Gender",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            gen_lb.place(relx=0.1,y=250,anchor=CENTER)

            gender=StringVar()
            gen_op1=CTkRadioButton(f8,text="Male",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="M")
            gen_op1.place(relx=0.25,y=255,anchor=CENTER)

            gen_op2=CTkRadioButton(f8,text="Female",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="F")
            gen_op2.place(relx=0.35,y=255,anchor=CENTER)

            gen_op3=CTkRadioButton(f8,text="Others",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="O")
            gen_op3.place(relx=0.47,y=255,anchor=CENTER)

            quali_lb=CTkLabel(f8,text="Qualification",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            quali_lb.place(relx=0.1,y=310,anchor=CENTER)

            quali_op=CTkOptionMenu(f8,width=180,values=["D.ed","B.ed","M.ed","Phd"])
            quali_op.place(relx=0.3,y=315,anchor=CENTER)

            date_lb=CTkLabel(f8,text="Date of Birth",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            date_lb.place(relx=0.1,y=390,anchor=CENTER)

            date_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            date_et.place(relx=0.3,y=390,anchor=CENTER)

            flash_massa=CTkLabel(f8,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#96DED1")
            flash_massa.place(relx=0.52,y=510,anchor=CENTER)

            #def for sending sms to teacher
            def send_sms_teacher(target_no,name,id,user,passwd):
                client=Client("<Account SID>","<Auth Token>")
                teacher_id=str(id)
                message=client.messages.create(
                body=("-\n\n🎉 Congratulations!\n\nDear "+name+",\n\nCongratulations on joining our institution as a teacher.we would like to provide you with your teacher ID, username, and password for accessing our online systems\nTeacher Id:"+teacher_id+"\nUsername:"+user+"\nPassword:"+passwd+"\nPlease ensure to keep this information confidential and secure. Your username and password will grant you access to our online platforms, including our learning management system, attendance tracking system, grade management system and more\nShould you have any further queries or need additional information, please do not hesitate to contact the administration office.\n\nBest Regards,\nTeam Edutrack"),
                from_="<My Twilio phone number>",
                to=target_no
                )

            def submit():
                    global db
                    id=id_et.get()
                    name=name_et.get()
                    gen=gender.get()
                    quali=quali_op.get()
                    dob=date_et.get()
                    phone=mobi_et.get()
                    phone_="+91"+phone
                    address=add_et.get("1.0", "end-1c")
                    username=user_et.get()
                    password=pass_et.get()
                    if len(gen)==0 or len(name)==0 or len(gen)==0 or len(dob)==0 or len(phone)==0 or len(address)==0 or len(username)==0 or len(id)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Try Again",flash_massa)
                    else:
                        cursor=db.cursor()
                        flag_ph=True
                        flag_nm=True
                        flag_dob=True
                        flag_pass=True
                        if len(phone)!=10 or not(phone.isdigit()) or not(phone[0] in ["9","8","7"]):
                            flag_ph=False
                        for i in name:
                            if i.isnumeric():
                                flag_nm=False
                                break
                        special_characters = "!@#$%^&*()_+}~`-=;'/.,<>?|"
                        if len(name)==0 or any(char in special_characters for char in name):
                            flag_nm=False
                        year=dob[0:4]
                        date=dob[8:10]
                        month=dob[5:7]
                        flag_dob=validate_dob(year,month,date,dob,flag_dob)
                        if len(password)<5:
                            flag_pass=False       
                        if flag_pass==True and flag_ph==True and flag_nm==True and flag_dob==True:
                            cursor.execute("insert into teacher values(%s,%s,%s,%s,%s,%s,%s,%s,%s)",(id,name,gen,quali,dob,phone,address,username,password))
                            db.commit()
                            flash_massa.configure(text_color="#04C34D")
                            flash_message("Teacher Added Successfully",flash_massa)
                            send_sms_teacher(phone_,name,id,username,password)
                        elif flag_nm==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Name",flash_massa)
                        elif flag_dob==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Date of Birth",flash_massa)
                        elif flag_ph==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Phone No",flash_massa)
                        elif flag_pass==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Password length",flash_massa)

            sub_bt=CTkButton(f8,text="Submit",command=submit,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            sub_bt.place(relx=0.52,y=470,anchor=CENTER)

        add_t=CTkButton(f3,text="Add Teacher",hover_color="#D9D9D0",command=add_teach,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        add_t.place(x=120,y=120,anchor=CENTER)
        animate_text(add_t,20)
        #remove teacher
        def del_teach():
            teach_hide_hover()
            del_t.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)
            animate_frame(f8)

            main_lb=CTkLabel(f8,text="Enter the detail to remove Teacher",text_color="black",font=CTkFont("Helvetica",35))
            main_lb.place(relx=0.5,rely=0.3,anchor=CENTER)
            animate_text(main_lb,20)
            id_lb=CTkLabel(f8,text="Teacher id",width=120,height=35,corner_radius=8,font=("Helvetica",23),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.41,rely=0.44,anchor=CENTER)

            id_et=CTkEntry(f8,justify=CENTER,height=45,width=170,corner_radius=30,border_width=2,font=("Helvetica",23))
            id_et.place(relx=0.58,rely=0.44,anchor=CENTER)

            flash_massa=CTkLabel(f8,fg_color="#96DED1",bg_color="#96DED1",text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20))
            flash_massa.place(relx=0.5,rely=0.7,anchor=CENTER)

            del_bt=CTkButton(f8,text="Remove",command=lambda: remove(id_et,"Teacher","teacher_id",flash_massa),height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            del_bt.place(relx=0.5,rely=0.6,anchor=CENTER)

        del_t=CTkButton(f3,text="Remove Teacher",hover_color="#D9D9D0",command=del_teach,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        del_t.place(x=120,y=195,anchor=CENTER)
        animate_text(del_t,20)
        #update teacher
        def up_teach():
            teach_hide_hover()
            up_t.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=875,height=560,fg_color="#96DED1",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=310,y=100)
            animate_frame(f8)

            main_lb=CTkLabel(f8,text="Update Teacher",text_color="black",font=CTkFont("Helvetica",30))
            main_lb.place(relx=0.5,y=30,anchor=CENTER)
            animate_text(main_lb,20)

            id_lb=CTkLabel(f8,text="Teacher id",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            id_lb.place(relx=0.1,y=120,anchor=CENTER)

            id_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            id_et.place(relx=0.3,y=125,anchor=CENTER)

            def fetch():
                t_id=id_et.get()
                if len(t_id)==0:
                    flash_massa.configure(text_color="red")
                    flash_message("Try Again",flash_massa)
                else:
                    t_id=id_et.get()
                    cursor=db.cursor()
                    cursor.execute("Select name,gen,quali,dob,phone_no,address,username,password from teacher where teacher_id=%s",(t_id))
                    data=cursor.fetchall()
                    if len(data)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Teacher of given id doesn't exist",flash_massa)
                    detail=data[0]
                    name=detail.get("name")
                    gen=detail.get('gen')
                    std_code=detail.get('quali')
                    dob=detail.get('dob')
                    phone=detail.get("phone_no")
                    add=detail.get("address")
                    user=detail.get("username")
                    passwd=detail.get("password")

                    #wipping data
                    mobi_et.delete(0,END)
                    name_et.delete(0,END)
                    add_et.delete(0,END)
                    date_et.delete(0,END)
                    user_et.delete(0,END)
                    pass_et.delete(0,END)

                    #inserting data
                    mobi_et.insert(0,phone)
                    name_et.insert(0,name)
                    add_et.insert(0,add)
                    date_et.insert(0,dob)
                    gender.set(gen)
                    quali_op.set(std_code)
                    user_et.insert(0,user)
                    pass_et.insert(0,passwd)
                
            fet_bt=CTkButton(f8,text="fetch",command=fetch,height=20,width=60,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            fet_bt.place(relx=0.47,y=125,anchor=CENTER)

            mobi_lb=CTkLabel(f8,text="Mobile No",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            mobi_lb.place(relx=0.6,y=120,anchor=CENTER)

            mobi_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            mobi_et.place(relx=0.8,y=125,anchor=CENTER)

            name_lb=CTkLabel(f8,text="Name",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            name_lb.place(relx=0.1,y=190,anchor=CENTER)

            name_et=CTkEntry(f8,corner_radius=30,height=30,width=200,border_width=2,font=("Roboto",12))
            name_et.place(relx=0.3,y=195,anchor=CENTER)

            add_lb=CTkLabel(f8,text="Address",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            add_lb.place(relx=0.6,y=190,anchor=CENTER)

            add_et=CTkEntry(f8,corner_radius=10,height=80,width=200,border_width=2,font=("Roboto",12))
            add_et.place(relx=0.8,y=200,anchor=CENTER)

            gen_lb=CTkLabel(f8,text="Gender",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            gen_lb.place(relx=0.1,y=250,anchor=CENTER)

            gender=StringVar()
            gen_op1=CTkRadioButton(f8,text="Male",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="M")
            gen_op1.place(relx=0.25,y=255,anchor=CENTER)

            gen_op2=CTkRadioButton(f8,text="Female",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="F")
            gen_op2.place(relx=0.35,y=255,anchor=CENTER)

            gen_op3=CTkRadioButton(f8,text="Others",fg_color="black",font=CTkFont("Helvetica",20),variable=gender,value="O")
            gen_op3.place(relx=0.47,y=255,anchor=CENTER)

            quali_lb=CTkLabel(f8,text="Qualification",width=120,height=35,corner_radius=8,font=CTkFont("Helvetica",20),text_color="black",bg_color="#96DED1")
            quali_lb.place(relx=0.1,y=310,anchor=CENTER)

            quali_op=CTkOptionMenu(f8,width=180,values=["D.ed","B.ed","M.ed","Phd"])
            quali_op.place(relx=0.3,y=315,anchor=CENTER)

            date_lb=CTkLabel(f8,text="Date of Birth",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            date_lb.place(relx=0.1,y=390,anchor=CENTER)

            date_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            date_et.place(relx=0.3,y=390,anchor=CENTER)

            user_lb=CTkLabel(f8,text="Username",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            user_lb.place(relx=0.6,y=280,anchor=CENTER)

            user_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            user_et.place(relx=0.8,y=280,anchor=CENTER)

            pass_lb=CTkLabel(f8,text="Password",width=120,height=35,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="#96DED1")
            pass_lb.place(relx=0.6,y=360,anchor=CENTER)

            pass_et=CTkEntry(f8,height=30,width=200,border_width=2,font=("Roboto",12))
            pass_et.place(relx=0.8,y=360,anchor=CENTER)

            flash_massa=CTkLabel(f8,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#96DED1")
            flash_massa.place(relx=0.5,y=510,anchor=CENTER)
            
            def update():
                    global db
                    id=id_et.get()
                    name=name_et.get()
                    gen=gender.get()
                    std=quali_op.get()
                    dob=date_et.get()
                    phone=mobi_et.get()
                    add=add_et.get()
                    cursor=db.cursor()
                    user=user_et.get()
                    passwd=pass_et.get()
                    if len(gen)==0 or len(name)==0 or len(gen)==0 or len(dob)==0 or len(phone)==0 or len(add)==0 or len(user)==0 or len(id)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Try Again",flash_massa)
                    else:
                        cursor=db.cursor()
                        flag_ph=True
                        flag_nm=True
                        flag_dob=True
                        flag_pass=True
                        if len(phone)!=10 or not(phone.isdigit()) or not(phone[0] in ["9","8","7"]):
                            flag_ph=False
                        for i in name:
                            if i.isnumeric():
                                flag_nm=False
                                break
                        special_characters = "!@#$%^&*()_+}~`-=;'/.,<>?|"
                        if len(name)==0 or any(char in special_characters for char in name):
                            flag_nm=False
                        year=dob[0:4]
                        date=dob[8:10]
                        month=dob[5:7]
                        flag_dob=validate_dob(year,month,date,dob,flag_dob)
                        if len(passwd)<5:
                            flag_pass=False       
                        if flag_pass==True and flag_ph==True and flag_nm==True and flag_dob==True:
                            cursor.execute("Update teacher set name=%s,gen=%s,quali=%s,dob=%s,phone_no=%s,address=%s,username=%s,password=%s where teacher_id=%s",(name,gen,std,dob,phone,add,user,passwd,id))
                            db.commit()
                            flash_massa.configure(text_color="#04C34D")
                            flash_message("Teacher Updated Successfully",flash_massa)
                        elif flag_nm==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Name",flash_massa)
                        elif flag_dob==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Date of Birth",flash_massa)
                        elif flag_ph==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Phone No",flash_massa)
                        elif flag_pass==False:
                            flash_massa.configure(text_color="red")
                            flash_message("Invalid Password length",flash_massa)

            up_bt=CTkButton(f8,text="Update",command=update,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            up_bt.place(x=440,y=470,anchor=CENTER)

        up_t=CTkButton(f3,text="Update Teacher",hover_color="#D9D9D0",command=up_teach,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        up_t.place(x=120,y=270,anchor=CENTER)
        animate_text(up_t,20)
        vie_te.invoke()



    #----------------------------------------------------------------Teacher Assignment Frame----------------------------------------------------------------


    #teacher_assign_frame
    def teacher_assign_frame():
        date_time_display()
        #timetable_section
        l2=CTkLabel(f0,text="Classwise teacher's details",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)

        #frame for buttons 
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        animate_frame(f3)

        def treeview(std_code):
            f9=CTkFrame(f0,width=625,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            f9.place(x=290,y=100)
            animate_frame(f9)
            cursor=db.cursor()
            cursor.execute("Select teacher.name,subject.sub_name From teacher,class,teach_class,subject where teacher_id=teacher_code and sub_id=sub_code and std_id=std_code and std_id=%s",(std_code))
            data=cursor.fetchall()
            teach_name=[]
            subject_name=[]
            for i in data:
                    sub_name=i.get("name")
                    teacher_name=i.get("sub_name")
                    teach_name.append(teacher_name)
                    subject_name.append(sub_name)

            stu_table=ttk.Treeview(f9,columns=("subject_name","teach_name"),show="headings")
            style=ttk.Style(f9)
            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("subject_name",text="Subject")
            stu_table.heading("teach_name",text="Teacher Name")
            stu_table.column("teach_name",width=300,anchor=CENTER)
            stu_table.column("subject_name",width=300,anchor=CENTER)

            for i in range(len(teach_name)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(teach_name[i],subject_name[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)
        def std_hide_hover():
            std_5.configure(fg_color="#33CCFF")
            std_6.configure(fg_color="#33CCFF")
            std_7.configure(fg_color="#33CCFF")
            std_8.configure(fg_color="#33CCFF")
            std_9.configure(fg_color="#33CCFF")
            std_10.configure(fg_color="#33CCFF")
        #list with all teacher's names
        cursor=db.cursor()
        cursor.execute("Select name from teacher")
        data=cursor.fetchall()
        teach_name=[]
        for i in data:
            teacher_name=i.get("name")
            teach_name.append(teacher_name)
        #def for getting the exact subject teacher
        def get_teacher_nm(sub_no,std_co):
            cursor=db.cursor()
            cursor.execute("select name from teacher where teacher_id=(select teacher_code from teach_class where sub_code=%s and std_code=%s)",(sub_no,std_co))
            data=cursor.fetchall()
            teacher=data[0]
            t_nm=teacher.get("name")
            return t_nm
        #def for edit button
        def edit_tt(std_code):
            f8=CTkFrame(f0,width=605,height=540,fg_color="#96DED1",bg_color="white",border_width=0,corner_radius=12,border_color="black")
            f8.place(x=300,y=110)
            animate_frame(f8)
            eng=CTkLabel(f8,text="English",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            eng.place(x=100,y=20)
            option_eng=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_eng.set(get_teacher_nm(1,std_code))
            option_eng.place(x=300,y=23)
            mara=CTkLabel(f8,text="Marathi",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            mara.place(x=100,y=90)
            option_mara=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_mara.set(get_teacher_nm(3,std_code))
            option_mara.place(x=300,y=93)
            hin=CTkLabel(f8,text="Hindi",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            hin.place(x=100,y=160)
            option_hin=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_hin.set(get_teacher_nm(2,std_code))
            option_hin.place(x=300,y=163)
            sci=CTkLabel(f8,text="Science",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            sci.place(x=100,y=230)
            option_sci=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_sci.place(x=300,y=233)
            option_sci.set(get_teacher_nm(5,std_code))
            math=CTkLabel(f8,text="Mathematics",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            math.place(x=100,y=300)
            option_math=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_math.set(get_teacher_nm(4,std_code))
            option_math.place(x=300,y=303)
            ss=CTkLabel(f8,text="Social Studies",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            ss.place(x=100,y=370)
            option_ss=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_ss.set(get_teacher_nm(6,std_code))
            option_ss.place(x=300,y=373)
            pt=CTkLabel(f8,text="Physical Training",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            pt.place(x=100,y=440)
            option_pt=CTkOptionMenu(f8,width=200,height=40,values=teach_name)
            option_pt.set(get_teacher_nm(7,std_code))
            option_pt.place(x=300,y=443)
            note=CTkLabel(f8,text="*Do not enter Same teacher for two subjects",height=25,width=170,corner_radius=20,text_color="black",fg_color="#96DED1",font=CTkFont("Helvetica",18))
            note.place(x=15,y=500)
            def save():
                t_eng=option_eng.get()
                t_mara=option_mara.get()
                t_hin=option_hin.get()
                t_sci=option_sci.get()
                t_math=option_math.get()
                t_ss=option_ss.get()
                t_pt=option_pt.get()
                def update_t(sub_t,sub_no,std_no):
                        cursor=db.cursor()
                        cursor.execute("update teach_class set teacher_code=(select teacher_id from teacher where name=%s) where sub_code=%s and std_code=%s",(sub_t,sub_no,std_no))
                        db.commit()
                update_t(t_eng,1,std_code)
                update_t(t_hin,2,std_code)
                update_t(t_mara,3,std_code)
                update_t(t_sci,5,std_code)
                update_t(t_math,4,std_code)
                update_t(t_ss,6,std_code)
                update_t(t_pt,7,std_code)
                flash_massa=CTkLabel(f0,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#66B3FF")
                flash_massa.place(relx=0.5,y=690,anchor=CENTER)
                flash_massa.configure(text_color="#04C34D")
                flash_message("Updated Successfully",flash_massa)
                treeview(std_code)
            save_b=CTkButton(f8,command=save,text="Save",height=35,width=100,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            save_b.place(x=480,y=490)

        def std5():
            std_hide_hover()
            std_5.configure(fg_color="#888888")
            treeview("5th std")
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("5th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        def std6():
            std_hide_hover()
            std_6.configure(fg_color="#888888")
            treeview("6th std")
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("6th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        def std7():
            std_hide_hover()
            std_7.configure(fg_color="#888888")
            treeview("7th std")
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("7th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        def std8():
            std_hide_hover()
            std_8.configure(fg_color="#888888")
            treeview("8th std")
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("8th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        def std9():
            std_hide_hover()
            std_9.configure(fg_color="#888888")
            treeview("9th std")
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("9th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        def std10():
            std_hide_hover()
            std_10.configure(fg_color="#888888")  
            treeview("10th std") 
            #edit button
            photo1=CTkImage(Image.open("pencil.png"),size=(50,50))
            edit_b=CTkButton(f0,command=lambda: edit_tt("10th std"),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
            edit_b.place(x=920,y=100)

        std_5=CTkButton(f3,hover_color="#D9D9D0",command=std5,text="5th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_5.place(relx=0.5,y=45,anchor=CENTER)
        animate_text(std_5,20)
        std_6=CTkButton(f3,hover_color="#D9D9D0",command=std6,text="6th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_6.place(relx=0.5,y=120,anchor=CENTER)
        animate_text(std_6,20)
        std_7=CTkButton(f3,hover_color="#D9D9D0",command=std7,text="7th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_7.place(relx=0.5,y=195,anchor=CENTER)
        animate_text(std_7,20)
        std_8=CTkButton(f3,hover_color="#D9D9D0",command=std8,text="8th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_8.place(relx=0.5,y=270,anchor=CENTER)
        animate_text(std_8,20)
        std_9=CTkButton(f3,hover_color="#D9D9D0",command=std9,text="9th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_9.place(relx=0.5,y=345,anchor=CENTER)
        animate_text(std_9,20)
        std_10=CTkButton(f3,hover_color="#D9D9D0",command=std10,text="10th Standard",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        std_10.place(relx=0.5,y=420,anchor=CENTER)
        animate_text(std_10,20)
        std_5.invoke()


    #----------------------------------------------------------------Complain Frame----------------------------------------------------------------



    #complain_frame
    def complain_frame():
        global buttons_complain
        date_time_display()
        def show_complains(btn,depart):
            for i in buttons_complain:
                if btn==i:
                    btn.configure(fg_color="#888888")
                else:
                    i.configure(fg_color="#33CCFF")
            open_f=CTkFrame(f0,width=870,height=560,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
            open_f.place(x=325,y=100)
            animate_frame(open_f)
            all=CTkLabel(open_f,text=depart+" - Related Complains",height=45,width=470,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
            all.place(x=20,y=20)
            scroll_f=CTkScrollableFrame(open_f,width=810,corner_radius=20,fg_color="#B6E5D8",height=430)
            scroll_f.place(relx=0.5,rely=0.55,anchor=CENTER)
            cursor=db.cursor()
            cursor.execute("Select complain_id,subject from complain where `to`=%s and depart=%s",("Admin",depart))
            data=cursor.fetchall()
            def comp_desc(id):
                sol_lb=CTkLabel(open_f,text="Solution",height=45,width=170,corner_radius=20,text_color="black",fg_color="#e4d1d1",font=CTkFont("Helvetica",20))
                sol_lb.place(x=650,y=20)
                all=CTkLabel(open_f,text=" ",height=45,width=470,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                all.place(x=20,y=20)
                #back button
                photo1=CTkImage(Image.open("back.png"),size=(40,40))
                edit_b=CTkButton(open_f,command=lambda param=depart,new_b=btn: show_complains(new_b, param),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
                edit_b.place(x=20,y=18)

                sol_f=CTkFrame(open_f,width=860,height=470,fg_color="#FFE5B4",border_width=0,corner_radius=12)
                sol_f.place(relx=0.5,rely=0.55,anchor=CENTER)
                cursor=db.cursor()
                cursor.execute("Select stud_code,description,solution,hide from complain where complain_id=%s",(id))
                result=cursor.fetchall()
                stud_id=result[0]['stud_code']
                desc=result[0]['description']
                solu=result[0]['solution']
                anon=result[0]['hide']
                id_lb=CTkLabel(sol_f,text="Complain id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                id_lb.place(x=15,y=10)
                id_et=CTkEntry(sol_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
                id_et.place(x=155,y=10)
                stid_lb=CTkLabel(sol_f,text="Student id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                stid_lb.place(x=415,y=10)
                stid_et=CTkEntry(sol_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
                stid_et.place(x=555,y=10)
                desc_t=CTkTextbox(sol_f,font=CTkFont("Helvetica",20),width=824,height=150,fg_color="#ffff99",border_width=3,corner_radius=12,border_color="black")
                desc_t.place(x=15,y=50)
                sol_t=CTkTextbox(sol_f,font=CTkFont("Helvetica",20),width=824,height=200,fg_color="#e6f7ff",border_width=3,corner_radius=12,border_color="black")
                sol_t.place(x=15,y=220)
                #inserting values
                id_et.insert(0,id)
                stid_et.insert(0,stud_id)
                desc_t.insert('0.0',desc)
                desc_t.configure(state="disabled")
                id_et.configure(state="disabled")
                stid_et.configure(state="disabled")
                if anon==1:
                    stid_et.destroy()
                    stid_lb1=CTkLabel(sol_f,text="Anonymous",height=25,width=70,corner_radius=10,text_color="#04C34D",fg_color="#FFE5B4",font=CTkFont("Helvetica",18))
                    stid_lb1.place(x=550,y=10)
                    
                if solu==None:
                    sol_t.insert('0.0',"Pending")
                else:
                    sol_t.insert('0.0',solu)
                def save_sol():
                    flash_massa=CTkLabel(sol_f,text_color="#04C34D",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
                    flash_massa.place(relx=0.5,rely=0.95,anchor=CENTER)
                    solution=sol_t.get("1.0", "end-1c")
                    if len(solution)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Try Again",flash_massa)
                    else:
                        solution=sol_t.get("1.0", "end-1c")
                        cursor=db.cursor()
                        cursor.execute('update complain set solution=%s where complain_id=%s',(solution,id))
                        db.commit()
                        flash_massa.configure(text_color="#04C34D")
                        flash_message("Saved Successfully",flash_massa)
                save_b=CTkButton(sol_f,text="Save",command=save_sol,height=35,width=100,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                save_b.place(relx=0.9,rely=0.95,anchor=CENTER)
            r=0
            y_pad=0
            for i in range(len(data)):
                id=data[i]['complain_id']
                id=str(id)
                sub=data[i]['subject']
                new_b=CTkButton(scroll_f,hover_color="#D9D9D0",command=lambda param=id: comp_desc(param),text=id+"       "+sub,height=45,width=790,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                animate_text(new_b,20)
                r+=1
        

        l2=CTkLabel(f0,text="Complain Section",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        new_f=CTkScrollableFrame(f0,width=240,height=535,fg_color="white",border_width=2,corner_radius=12,border_color="black")
        new_f.place(x=40,y=100)
        animate_frame(new_f)
        cursor=db.cursor()
        cursor.execute('select distinct(depart) from complain where `to`=%s',("Admin"))
        depart=cursor.fetchall()
        departments=[]
        for i in range(len(depart)):
            new=depart[i]
            departments.append(new['depart'])
        buttons_complain=[]
        r=0
        y_pad=0
        for i in departments:
            new_b=CTkButton(new_f,hover_color="#D9D9D0",text=i,height=45,width=200,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
            new_b.configure(command=lambda param=i,new_b=new_b: show_complains(new_b, param))
            animate_text(new_b,20)
            buttons_complain.append(new_b)
            r+=1
        buttons_complain[0].invoke()




    #----------------------------------------------------------------Admin's Main Window----------------------------------------------------------------


    set_appearance_mode("light")
    set_default_color_theme("blue")
    admin_win=CTk()
    admin_win.title("Admin home page")
    screen_width = admin_win.winfo_screenwidth()
    screen_height= admin_win.winfo_screenheight()
    admin_win_width = screen_width
    admin_win_height = screen_height
    admin_win.geometry(f"{admin_win_width}x{admin_win_height}")

    admin_win.geometry("+0+0")
    # admin_win.maxsize(width=1400,height=750)
    # admin_win.minsize(width=1400,height=750)
    admin_win.attributes('-fullscreen',True)
    admin_win.iconbitmap("logo_icon.ico")

    frame=CTkFrame(admin_win,width=1900,height=1000,fg_color="#66B3FF")
    frame.pack()

    #Home frame
    f0=CTkFrame(frame,width=1200,height=700,fg_color="#66B3FF")
    f0.place(x=140,y=20)

    #Dashboard
    f1=CTkFrame(frame,width=100,height=655,fg_color="white",border_width=3,corner_radius=15,border_color="black")
    f1.place(x=50,y=30)

    #logo
    photo=CTkImage(Image.open("logo.png"),size=(60,60))
    l1=CTkLabel(f1,image=photo,text=" ")
    l1.place(x=18,y=40)

    #home indicator
    home_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    home_indicate.place(x=7,y=150)

    #student indicator
    student_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    student_indicate.place(x=7,y=250)

    #teacher indicator
    teacher_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    teacher_indicate.place(x=7,y=350)

    #timetable indicator
    timetable_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    timetable_indicate.place(x=7,y=450)

    #complain indicator
    complain_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    complain_indicate.place(x=7,y=550)

    #to initialize the admin_win
    indicate(home_indicate,home_frame)

    #home button
    photo1=CTkImage(Image.open("home.png"),size=(50,50))
    b1=CTkButton(f1,command=lambda: indicate(home_indicate,home_frame),image=photo1,text="",hover_color="#white",cursor="hand2",width=15,height=40,fg_color="white")
    b1.place(x=17,y=150)


    #student button
    photo2=CTkImage(Image.open("college.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(student_indicate,student_frame),image=photo2,text=" ",hover_color="#white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=250)


    #teacher button 
    photo3=CTkImage(Image.open("class.png"),size=(50,50))
    b3=CTkButton(f1,command=lambda: indicate(teacher_indicate,teacher_frame),image=photo3,text=" ",hover_color="#white",cursor="hand2",width=15,height=40,fg_color="white")
    b3.place(x=15,y=350)


    #timetable button 
    photo4=CTkImage(Image.open("timetable.png"),size=(50,50))
    b4=CTkButton(f1,command=lambda: indicate(timetable_indicate,teacher_assign_frame),image=photo4,text=" ",hover_color="#white",cursor="hand2",width=15,height=40,fg_color="white")
    b4.place(x=15,y=450)


    #complain button
    photo4=CTkImage(Image.open("report.png"),size=(50,50))
    b5=CTkButton(f1,command=lambda: indicate(complain_indicate,complain_frame),image=photo4,text=" ",hover_color="#white",cursor="hand2",width=15,height=40,fg_color="white")
    b5.place(x=15,y=550)

    admin_win.mainloop()







#-----------------------------------------STUDENT_PAGE'S CODE STARTS HERE-----------------------------------------------------------





def student_page(student_id):

    #----------------------------------------------------------------Additional Data Access----------------------------------------------------------------

    #data access from database
    cursor=db.cursor()
    query1=f"Select * from student where stud_id={student_id}"
    cursor.execute(query1)
    s_data=cursor.fetchall()
    data1=s_data[0]
    su_id=data1.get("stud_id")
    Name=data1.get("name")
    Dob=data1.get("dob")
    Std=data1.get("std_code")
    Address=data1.get("address")
    mobile=data1.get("phone_no")
    gender=data1.get("gen")

    cursor = db.cursor()
    cursor.execute("select count(*) from student where std_code=%s and stud_id<%s", (Std, student_id))
    s_data = cursor.fetchall()
    roll_index = s_data[0]["count(*)"]  



    #----------------------------------------------------------------Local Function----------------------------------------------------------------


        
    #datetime
    def date_time_display():
        
        def ct_time():
            now = datetime.now()
            ct_string = now.strftime("%H:%M:%S")
            return ct_string

        def ct_change():
            ct_string = ct_time()
            time_lb.configure(text=ct_string)
            f0.after(1000, ct_change)  #update every 1 second
        #logout_frame
        def logout_frame():
            delete_frames()
            date_time_display()
            def destroy_window():
                student_win.destroy()
                login_page()
            #asking to logout
            log_lb=CTkLabel(f0,text="Do you want to logout ?",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
            log_lb.place(relx=0.4,y=200,anchor=CENTER)
            log_bu=CTkButton(f0,text="Yes",height=45,command=destroy_window,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            log_bu.place(relx=0.4,y=300,anchor=CENTER)

        today = datetime.today()
        t_date= today.strftime("%B %d, %Y")
        #date and time 
        d_f=CTkFrame(f0,width=350,height=50,border_color="black",border_width=3,fg_color="white",corner_radius=40)
        d_f.place(x=730,y=5)
        time_lb=CTkLabel(d_f,width=110,height=30,text="",font=CTkFont("Helvetica",19),fg_color="white",corner_radius=40,text_color="black")
        time_lb.place(relx=0.8,rely=0.5,anchor=CENTER)
        date_lb=CTkLabel(d_f,text=t_date,width=150,height=30,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="white")
        date_lb.place(relx=0.3,rely=0.5,anchor=CENTER)
        ct_change()

        photo1=CTkImage(Image.open("logout1.png"),size=(50,50))
        edit_b=CTkButton(f0,image=photo1,command=logout_frame,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
        edit_b.place(x=1100,y=0)


    #to delete frames
    def delete_frames():
        global flag_bot 
        if flag_bot==False:
            reverse_animation(frame1,30)
            flag_bot=True
        f0.update()
        
        for f in f0.winfo_children():
            f.destroy()
        f0.update()

    #to indicate
    def indicate(lb,frame):
        hide_indicators()
        lb.configure(fg_color="#0066ff")
        delete_frames()
        frame()

    #hide indicators
    def hide_indicators():
        home_indicate.configure(fg_color="white")
        search_indicate.configure(fg_color="white")
        schedule_indicate.configure(fg_color="white")
        complain_indicate.configure(fg_color="white")
        attendance_indicate.configure(fg_color="white")


    def display_connection_status():
        global first_enter,internet_bot
        def connect_check():
            IPaddress = socket.gethostbyname(socket.gethostname())
            if IPaddress == "127.0.0.1":
                return False
            else:
                return True 
        network_stat = connect_check()
        def dis():
            connect_check_lb.configure(fg_color="#66B3FF")
        def update_status():
            connect_display()
            connect_check_lb.after(3000, update_status)
        def connect_display():
            nonlocal network_stat
            network_Checked = connect_check()
            
            if network_Checked == True and network_stat == False:
                connect_check_lb.configure(fg_color="green")
                animate_bot_frame(connect_check_lb,5)
                flash_message("Internet Connected",connect_check_lb)
                connect_check_lb.after(3000,dis)
                network_stat = True
            if network_Checked == False and network_stat == True:
                connect_check_lb.configure(fg_color="red")
                animate_bot_frame(connect_check_lb,5)
                flash_message("No internet",connect_check_lb)
                connect_check_lb.after(3000,dis)
                network_stat = False

        connect_check_lb=CTkLabel(f0,text="",height=45,width=150,corner_radius=20,text_color="white",font=CTkFont("Helvetica",20,"bold"))
        connect_check_lb.place(relx=0.4,rely=0.95,anchor=CENTER)
        if first_enter:   
            if network_stat == True:
                connect_check_lb.configure(fg_color="green")
                animate_bot_frame(connect_check_lb,5)
                flash_message("Internet Connected",connect_check_lb)
                connect_check_lb.after(3000,dis)
            else:
                connect_check_lb.configure(fg_color="red")
                animate_bot_frame(connect_check_lb,5)
                flash_message("No internet",connect_check_lb)
                connect_check_lb.after(3000,dis)
            first_enter=False

        connect_check_lb.after(1000, update_status)



    #--------------------------------------------home frames-----------------------------------------------------------------------------------



    #home_frame
    def home_frame(): 
        #date and time 
        date_time_display()
        l2=CTkLabel(f0,text=("Welcome "+ Name),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=0)
        #animate_text(l2,20)
        
        f0.after(1000,display_connection_status)
        
        #identiy card
        f7=CTkFrame(f0,width=550,height=338,fg_color="#ffffe6",border_width=3,corner_radius=12,border_color="black")
        f7.place(x=330,y=150,anchor="n")
        photo5=CTkImage(Image.open("id_back.png"),size=(538,160))
        l3=CTkLabel(f7,image=photo5,text="")
        l3.place(x=6,y=5)
        photo6=CTkImage(Image.open("background.jpeg"),size=(538,18))
        l3=CTkLabel(f7,image=photo6,text="")
        l3.place(x=6,y=333,anchor="sw")

        photo6=CTkImage(Image.open("student_male.png"),size=(150,150))
        if gender=='M':
            pass
        else:
            photo6=CTkImage(Image.open("student_female.png"),size=(150,150))

        l4=CTkLabel(f7,image=photo6,text=" ")
        l4.place(x=315,y=90)

        photo7=CTkImage(Image.open("logo.png"),size=(70,70))
        l5=CTkLabel(f7,image=photo7,text=" ",bg_color='#6FD0FE')
        l5.place(x=60,y=15)

        l6=CTkLabel(f7,text="IDENTITY CARD",bg_color='#6FD0FE',font=("Helvetica",25),text_color="black")
        l6.place(x=170,y=33)

        id_lb=CTkLabel(f7,text=("STU.ID :  " + str(su_id)),width=100,height=30,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        id_lb.place(x=15,y=113)

        name_lb=CTkLabel(f7,text=("NAME :   " + Name),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        name_lb.place(x=15,y=140)

        dob_lb=CTkLabel(f7,text=("DOB    :  " + str(Dob)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        dob_lb.place(x=15,y=175)

        std_lb=CTkLabel(f7,text=("STD    :  "+ Std),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        std_lb.place(x=15,y=210)

        mobile_lb=CTkLabel(f7,text=("MOBILE NO. :  " + str(mobile)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        mobile_lb.place(x=15,y=245)

        address_lb=CTkLabel(f7,text=("ADDRESS :  " + Address),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6",wraplength=500)
        address_lb.place(x=15,y=280)
        messages = [{'role': 'system', 'content': f'behave like a bot named as Edubot owned by Edutrack ; which is develoved by manoj,dhruv and ankit ; sycs students at vivekanand education society ;dont use EduBot: text in your responce; you will be use by a student namely {Name},whose student id is {su_id} , he or shes address is {Address},mobile no. is {mobile} and studies in {Std}'}]
        def EduBot():
            
            global row_count,frame1 

            openai.api_key = '<Openai token key>'
            def AI_Responce(user_message):
                messages.append({"role": "user", "content": user_message})
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=100
                )
                reply = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": reply})
                return reply

            row_count =0

            

            def print_message(text1):
                global row_count
                chat1 = CTkLabel(chat_frame, text=text1, height=20, wraplength=300, corner_radius=12, text_color="black", fg_color="#ccffe6", font=CTkFont("Helvetica", 15))
                chat1.grid(column=1,row=row_count,sticky="e", pady=(13,10), padx=(5, 10),ipady=7)

                photo1=CTkImage(Image.open("student_male.png"),size=(30,30))
                if gender=='M':
                    pass
                else:
                    photo1=CTkImage(Image.open("student_female.png"),size=(30,30))
                l1=CTkLabel(chat_frame,image=photo1,text=" ")
                l1.grid(column=2,row=row_count,sticky="e")

                row_count+=1
                student_win.update()
                bot_reply =AI_Responce(text1)
                print_reply(bot_reply,(2,2))

            def print_reply(text1,valu):
                global chat2,row_count
                time.sleep(0)
                photo2=CTkImage(Image.open("chat_bot.png"),size=(30,30))
                l2 = CTkButton(chat_frame, image=photo2, command=lambda text2=text1: speak(text2),text="", width=0.1, hover_color="#e0e0e7", cursor="hand2", fg_color="#ECF2FF")
                l2.grid(column=0, row=row_count, sticky="w", padx=(5, 10))

                chat2 = CTkLabel(chat_frame, text=text1, height=30, wraplength=300, corner_radius=12, text_color="black", fg_color="#AEE2FF", font=CTkFont("Helvetica", 15))
                chat2.grid(column=1,row=row_count,sticky="w", padx=(0, 5),pady=valu,ipady=7)
                animate_text(chat2, 25)
                
                row_count+=1

            

            def send():
                request = entry1.get()
                entry1.delete(0, tk.END)
                if len(request) != 0:
                    print_message(request)

            frame1 = CTkFrame(f0, width=550, height=590,fg_color="#FFD0D0",corner_radius=30)
            frame1.place(x=330,y=100,anchor="n")
            animate_bot_frame(frame1,10)
            
            lab2=CTkLabel(frame1,text="bot",width=80,height=80,fg_color="#FFD0D0",corner_radius=80,font=CTkFont("Helvetica", 12))
            lab2.place(relx=0.5,rely=0,anchor=CENTER)
            photo1=CTkImage(Image.open("chat_bot.png"),size=(45,45))
            Bot_But=CTkButton(lab2,image=photo1,command=Bot_Button,text="",width=50,hover_color="#FFD0D0",cursor="hand2",fg_color="#FFD0D0",corner_radius=400)
            Bot_But.place(relx=0.5,rely=0.5,anchor=CENTER)

            chat_frame = CTkScrollableFrame(frame1, height=500, width=500,fg_color="#ECF2FF",scrollbar_button_color="#ECF2FF")
            chat_frame.place(relx=0.5, rely=0.47, anchor=CENTER)

            chat_frame.columnconfigure(0, weight=1)
            chat_frame.columnconfigure(1, weight=15)
            chat_frame.columnconfigure(2, weight=1)
            entry1 = CTkEntry(frame1,placeholder_text="Say hello to EduBot",width=450,height=35)
            entry1.place(relx=0.02, rely=0.95,anchor="w")
            send_button = CTkButton(frame1, text="Send",width=70,height=35, command=send)
            send_button.place(relx=0.98, rely=0.95, anchor="e")
            student_win.bind('<Return>', lambda event=None: send_button.invoke())
            
            frame1.update()
            frist=AI_Responce("greet us in 10 words;my name is"+Name)
            print_reply(frist,(20,0))

        def Bot_Button():
            global flag_bot 
            if flag_bot==True:
                EduBot()
                flag_bot=False
            else:
                reverse_animation(frame1,30)
                flag_bot=True
        
        lab2=CTkLabel(f0,text="bot",width=80,height=80,fg_color="#FFD0D0",corner_radius=80,font=CTkFont("Helvetica", 12))
        lab2.place(x=330,y=100,anchor=CENTER)
        photo1=CTkImage(Image.open("chat_bot.png"),size=(45,45))
        Bot_But=CTkButton(lab2,image=photo1,command=Bot_Button,text="",width=50,hover_color="#FFD0D0",cursor="hand2",fg_color="#FFD0D0",corner_radius=400)
        Bot_But.place(relx=0.5,rely=0.5,anchor=CENTER)

        
        #attendance_grap
        def attendance_graph(num_rows):
            def hide(btn):
                btn_day.configure(fg_color="#33CCFF")
                btn_week.configure(fg_color="#33CCFF")
                btn_month.configure(fg_color="#33CCFF")
                btn.configure(fg_color="#888888")
            if num_rows == 0:
                hide(btn_day)
            elif num_rows == 6:
                hide(btn_week)
            elif num_rows == 29:
                hide(btn_month)

            f8 = CTkFrame(f0, width=550, height=480, fg_color="#ccb3ff", border_width=3, corner_radius=12, border_color="black")
            f8.place(x=630, y=100)
            #animate_frame(f8)
            cursor.execute("select * from attendance WHERE attendance.attendance_date >= DATE(NOW()) - INTERVAL %s DAY and stand_code=%s order by attendance_date;", (num_rows,Std,))
            b_data = cursor.fetchall()

            subject_lec = [0, 0, 0, 0, 0, 0, 0]
            tot_attendance = [0, 0, 0, 0, 0, 0, 0]

            for i in range(len(b_data)):
                data1 = b_data[i]
                sub = data1["sub_code"]
                attendance_list = list(data1["attend_sheet"])

                if sub == 1:
                    subject_lec[0] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[0] += 1
                elif sub == 2:
                    subject_lec[1] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[1] += 1
                elif sub == 3:
                    subject_lec[2] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[2] += 1
                elif sub == 4:
                    subject_lec[3] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[3] += 1
                elif sub == 5:
                    subject_lec[4] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[4] += 1
                elif sub == 6:
                    subject_lec[5] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[5] += 1
                elif sub == 7:
                    subject_lec[6] += 1
                    if attendance_list[roll_index] == "1":
                        tot_attendance[6] += 1


            tot_subject = ['Eng', 'Hin', 'Mar', 'Math', 'Sci', 'Social', 'Pt']
            x = np.arange(len(tot_subject))
            fig = plt.figure(figsize=(5.40, 4.7))
            fig.set_facecolor("#e0e0eb")

            sns.lineplot(x=x, y=subject_lec, marker='o', label="Total_Lecture")
            sns.lineplot(x=x, y=tot_attendance, marker='o', label="Lecture_Attended")

            plt.xlabel('SUBJECT', fontstyle='italic', fontsize=12)
            plt.ylabel('ATTENDANCE(%)', fontstyle='italic', fontsize=12)
            plt.title('YOUR ATTENDANCE', fontweight='bold', fontsize=20)
            plt.xticks(x, tot_subject)
            plt.ylim([0, max(subject_lec) + 2])
            plt.legend()
            canvas = FigureCanvasTkAgg(fig, master=f8)
            canvas.draw()
            canvas.get_tk_widget().place(x=5, y=5)


        btn_day = CTkButton(f0, text=" Day ", command=lambda: attendance_graph(0), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
        btn_day.place(x=760, y=620,anchor=CENTER)

        btn_week = CTkButton(f0, text=" Week ", command=lambda: attendance_graph(6), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
        btn_week.place(x=925, y=620,anchor=CENTER)

        btn_month = CTkButton(f0, text=" Month ", command=lambda: attendance_graph(29), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
        btn_month.place(x=1080, y=620,anchor=CENTER)
        btn_day.invoke()

        #quotes
        def quotes():
            try:
                # list of quotes
                quotes = [
                    "Be the change you wish to see in the world.",
                    "The only way to do great work is to love what you do.",
                    "Not all who wander are lost.",
                    "Believe you can and you're halfway there.",
                    "If you want to go fast, go alone. If you want to go far, go together.",
                    "It does not matter how slowly you go as long as you do not stop.",
                    "Life is what happens when you're busy making other plans.",
                    "It is during our darkest moments that we must focus to see the light.",
                    "You must be the change you wish to see in the world.",
                    "The future belongs to those who believe in the beauty of their dreams.",
                    "The best way to predict the future is to invent it."]

                # randomly select a quote from the list
                selected_quote = random.choice(quotes)

                f6=CTkFrame(f0,width=550,height=100,fg_color="#ffe6cc",border_width=3,corner_radius=12,border_color="black")
                f6.place(x=55,y=520)
                quotes_lb=CTkLabel(f6,text="Thought of the day!",width=50,height=20,font=("bold",30),text_color="black",bg_color="transparent")
                quotes_lb.place(relx=0.5,rely=0.3,anchor=CENTER)
                label=CTkLabel(f6,text=selected_quote ,width=40,height=20,font=("bold",18),text_color="black",bg_color="transparent",wraplength=510)
                label.place(relx=0.5,rely=0.7,anchor=CENTER)
                animate_text(label,25)
                quotes_wall=CTkImage(Image.open("qoutes_bg.png"),size=(70,50))
                quotes_wlb=CTkLabel(f6,text="",image=quotes_wall,width=50,height=20,bg_color="transparent")
                quotes_wlb.place(relx=0.1,rely=0.3,anchor=CENTER)
            except Exception as e:
                pass
        quotes()



    #------------------------------------------------------search_frame-----------------------------------------------------------------------------------------------

    def search_frame():
        #date and time 
        date_time_display()
        #talk
        def talk():
            try:  
                speak(summary)
            except:
                speak("There is nothing to read, please search something first")

        def search():
            global result,summary
            try:
                result = input.get()
                if result=="":
                    raise speak("Please search something in search Bar !")
                summary = wikipedia.summary(result, sentences=3)
                text.configure(state="normal")
                text.delete('1.0',END) 
                text.insert('1.0',summary)
                text.configure(state="disable")
            except wikipedia.exceptions.WikipediaException:
                speak(f"Could not request results for ,{result}")

        
        def listen():
            input.delete(0, END)
            input.configure(placeholder_text=" Search with StudentHub")
            messagebox.showinfo("Status", "Tap to Speak...")
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
                    
            try:
                result = recognizer.recognize_google(audio)
                input.delete(0, END)
                input.insert(0, result)
                speak("You said, " + result)
                f0.update()
                search()
            except wikipedia.exceptions.WikipediaException:
                speak(f"Could not request results for ,{result}")
            except:
                speak("Could not understand audio")
            


        
        #student section
        l2=CTkLabel(f0,text="Search with StudentHub",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=0)
        #animate_text(l2,20)
        input = CTkEntry(f0, width = 350,height=40,font=("halvetica",20),fg_color="#ffe6cc",text_color="black",border_width=3,corner_radius=12,border_color="black",placeholder_text=" Search with StudentHub",placeholder_text_color="grey")
        input.place(x=100,y=90)

        text = CTkTextbox(f0,border_width=3,border_color="black",width=1040,corner_radius=15,height=531,font=("halvetica",20),fg_color="#ffe6cc")
        text.place(relx=0.44,rely=0.58,anchor=CENTER)
        text.configure(state="disable")
        animate_frame(text)

        photo7=CTkImage(Image.open("search_logo.png"),size=(30,30))
        button5 = CTkButton(f0,image=photo7,text='',width=50,height=40,fg_color="#e0e0eb")
        button5.configure(command=search)                                                 
        button5.place(x=455,y=90)

        photo8 = CTkImage(Image.open("mic.png"), size=(30, 30))
        button5 = CTkButton(f0, image=photo8, text='', width=50, height=40, fg_color="#e0e0eb",command=listen)
        button5.place(x=515, y=90)

        photo8 = CTkImage(Image.open("speaker.png"), size=(30, 30))
        button5 = CTkButton(f0, image=photo8, text='', width=50, height=40, fg_color="#e0e0eb",command=talk)
        button5.place(x=575, y=90)

        

    #-------------------------------------------------------------academic_frame-----------------------------------------------------------------------


    #Academic_frame
    def academic_frame():
        global taught_bt,grade_bt

        #Treeview
        def treeview():
            f91=CTkFrame(f0,width=900,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            f91.place(x=300,y=100)
            animate_frame(f91)
            cursor=db.cursor()
            cursor.execute("SELECT teacher.teacher_id,teacher.name,teacher.quali ,teacher.phone_no,subject.sub_name FROM teacher, class , teach_class , subject where teacher_id=teacher_code and sub_id=sub_code and std_id=std_code and std_id=%s",(Std))
            data=cursor.fetchall()
            teach_id=[]
            teach_name=[]
            teach_quali=[]
            subject_name=[]

            for i in data:
                    teacher_id=i.get("teacher_id")
                    teacher_name=i.get("name")
                    teacher_quali=i.get("quali")
                    sub_name=i.get("sub_name")
                    
                    teach_id.append(teacher_id)
                    teach_name.append(teacher_name)
                    teach_quali.append(teacher_quali)
                    subject_name.append(sub_name)

            stu_table=ttk.Treeview(f91,columns=("teach_id","teach_name","teach_quali","subject_name"),show="headings")
            style=ttk.Style(f91)
        
            style.theme_use("clam")
            style.configure("Treeview",rowheight=50,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("teach_id",text="Teacher_id")
            stu_table.heading("teach_name",text="Teacher_name")
            stu_table.heading("teach_quali",text="Teacher_quali")
            stu_table.heading("subject_name",text="Subject")

            stu_table.column("teach_id",width=170,anchor=CENTER)
            stu_table.column("teach_name",width=250,anchor=CENTER)
            stu_table.column("teach_quali",width=200,anchor=CENTER)
            stu_table.column("subject_name",width=250,anchor=CENTER)

            for i in range(len(teach_name)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(teach_id[i],teach_name[i],teach_quali[i],subject_name[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)

        def taught_by():
            grade_bt.configure(fg_color="#33CCFF")
            notes_bt.configure(fg_color="#33CCFF")
            taught_bt.configure(fg_color="#888888")
            treeview()
        def grade_frame():         
            def show_grade_stu(exam_type,own_bt,ot_bt):
                global capture_button
                f8=CTkFrame(tabview,width=860,height=480,fg_color="white",bg_color="white",corner_radius=12)
                f8.place(x=20,y=65)
                animate_frame(f8)
                flash_massa=CTkLabel(f8,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20))
                flash_massa.place(x=310,y=440)
                own_bt.configure(fg_color="#888888")
                ot_bt.configure(fg_color= "#33CCFF")
                if exam_type=="Mid Term":
                    unit_state="Unit I"
                elif exam_type=="Final Term":
                    unit_state="Unit II"
                cursor=db.cursor()
                cursor.execute("select obt_mks from grade where exam_type=%s and stud_code=%s order by(sub_code)",(exam_type,student_id))
                data=cursor.fetchall()
                if len(data)!=7:
                    capture_button = CTkButton(tabview, text="",width=60,height=40,corner_radius=20,fg_color= "white",font=CTkFont("Helvetica",20))
                    capture_button.place(relx=0.97, y=13,anchor="ne")
                    f8.configure(fg_color="white")
                    flash_massa.configure(text_color="red")
                    flash_message("Grades to be announced",flash_massa)
                else:
                    photo6=CTkImage(Image.open("download.png"),size=(30,30))
                    capture_button = CTkButton(tabview, text="",image=photo6,width=30,corner_radius=20,hover_color="#e0e0eb",fg_color= "white",font=CTkFont("Helvetica",20), command= lambda :capture_and_save(f8))
                    capture_button.place(relx=0.97, y=13,anchor="ne")
                    f8.configure(fg_color="#87CEFA")
                    sub_lb=CTkLabel(f8,text="Subject",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                    sub_lb.place(x=20,y=20)
                    unit_lb=CTkLabel(f8,text=unit_state+" /20",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                    unit_lb.place(x=190,y=20)
                    inter_lb=CTkLabel(f8,text="Internal /20",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                    inter_lb.place(x=360,y=20)
                    exter_lb=CTkLabel(f8,text="External /80",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                    exter_lb.place(x=530,y=20)
                    gr_lb=CTkLabel(f8,text="Grades",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                    gr_lb.place(x=700,y=20)

                    cursor=db.cursor()
                    cursor.execute("select sub_name from subject order by (sub_id)")
                    data=cursor.fetchall()
                    new_y=90
                    flag_fail=True
                    #for subjects
                    for i in range(len(data)):
                        subj_lb=CTkLabel(f8,text=data[i]['sub_name'],height=45,width=150,corner_radius=20,text_color="black",font=CTkFont("Helvetica",20))
                        subj_lb.place(x=90,y=new_y,anchor=CENTER)
                        new_y+=50

                    #for unit_marks
                    cursor=db.cursor()
                    cursor.execute("select obt_mks from grade where stud_code=%s and exam_type=%s order by (sub_code)",(student_id,unit_state))
                    data=cursor.fetchall()
                    new_y=90
                    for i in range(len(data)):
                        mk_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                        mk_e.place(x=260,y=new_y,anchor=CENTER)
                        mk_e.insert(0,data[i]['obt_mks'])
                        if data[i]['obt_mks']<7:
                            mk_e.configure(fg_color="#ffcccc")
                            flag_fail=False
                        mk_e.configure(state="disabled")
                        new_y+=50

                    #for inter_exter_marks
                    cursor=db.cursor()
                    cursor.execute("select internal_mk,external_mk,obt_mks,obt_grd from grade where stud_code=%s and exam_type=%s order by (sub_code)",(student_id,exam_type))
                    term_data=cursor.fetchall()
                    new_y=90
                    for i in range(len(term_data)):
                        int_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                        int_e.place(x=435,y=new_y,anchor=CENTER)
                        ext_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                        ext_e.place(x=600,y=new_y,anchor=CENTER)
                        gr_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                        gr_e.place(x=770,y=new_y,anchor=CENTER)
                        int_e.insert(0,term_data[i]['internal_mk'])
                        ext_e.insert(0,term_data[i]['external_mk'])
                        gr_e.insert(0,term_data[i]['obt_grd'])
                        if term_data[i]['external_mk']<28:
                            ext_e.configure(fg_color="#ffcccc")
                            flag_fail=False
                        int_e.configure(state="disabled")
                        ext_e.configure(state="disabled")
                        gr_e.configure(state="disabled")
                        new_y+=50
                    tot_mk=0
                    status="Pass"
                    for i in range(len(term_data)):
                        tot_mk+=data[i]['obt_mks']+term_data[i]['obt_mks']
                    percent=round(((tot_mk/840)*100),2)
                    per_lb=CTkLabel(f8,text="Percentage",height=45,width=150,corner_radius=20,text_color="black",fg_color="#E6E6FA",font=CTkFont("Helvetica",20))
                    per_lb.place(x=20,y=420)
                    per_e=CTkEntry(f8,justify="center",fg_color="#A2C4E0",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    per_e.insert(0,str(percent)+" %")
                    per_e.configure(state="disabled")
                    per_e.place(x=180,y=423)
                    st_lb=CTkLabel(f8,text="Status",height=45,width=150,corner_radius=20,text_color="black",fg_color="#E6E6FA",font=CTkFont("Helvetica",20))
                    st_lb.place(x=530,y=420)
                    st_e=CTkEntry(f8,justify="center",fg_color="#A2C4E0",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    st_e.place(x=690,y=423)
                    if flag_fail==False:
                        st_e.configure(text_color="red")
                        status="Fail"
                    else:
                        st_e.configure(text_color="green")
                    st_e.insert(0,status)
                    st_e.configure(state="disabled")

            def capture_and_save(tabview):
                screenshot = pyautogui.screenshot(region=(tabview.winfo_rootx(), tabview.winfo_rooty(), tabview.winfo_width(), tabview.winfo_height()))
                screenshot_path = "frame_screenshot.png"
                screenshot.save(screenshot_path)
                screenshots.append(screenshot_path)
                update_pdf()

            def update_pdf():
                pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                if pdf_path:
                    c = canvas.Canvas(pdf_path, pagesize=letter)

                    # Add school logo before the school name
                    logo_path = "pdf_logo.png"  # Update with your logo file path
                    logo_width, logo_height = 100, 100  # Adjust the logo size as needed


                    school_name = "EDU TRACK"
                    sid="Student Id : "+str(su_id)
                    sb_date="DOB : "+str(Dob)
                    exam="Mid Term"
                    s_stan=" STD : "+str(Std)
                    sname="Name : "+Name
                    c.setFont("Helvetica-Bold", 30)
                    
                    for img_path in screenshots:
                        img = Image.open(img_path)
                        img_width, img_height = img.size
                        page_width, page_height = letter

                        # Calculate the scale factor to fit the image within the PDF page
                        scale = min(page_width / img_width, page_height / img_height)

                        scaled_width = img_width * scale
                        scaled_height = img_height * scale
    
                        x = (page_width - scaled_width) / 2
                        y = (page_height - scaled_height) / 2

                        # Draw logo
                        c.drawImage(logo_path, 150, page_height - 100, width=logo_width, height=logo_height)
                        
                        # Draw school name
                        c.drawString(250, page_height - 70, school_name)
                        def drawdetails():
                                c.setFont("Helvetica-Bold", 15)
                                c.drawString(120, page_height - 150, sid)
                                c.drawString(350, page_height - 150, s_stan)
                                c.drawString(120, page_height - 180, sname)
                                c.drawString(350, page_height - 180, sb_date)
                                c.setFillColor(colors.red)
                                c.setFont("Times-Italic", 19)
                                c.drawString(260, page_height - 110, exam)
                        drawdetails()
                        
                        # Draw screenshot
                        c.drawImage(img_path, x, y, width=scaled_width, height=scaled_height)
                        c.showPage()
                        break

                    c.save()


            grade_bt.configure(fg_color="#888888")
            notes_bt.configure(fg_color="#33CCFF")
            taught_bt.configure(fg_color="#33CCFF") 
            tabview = CTkFrame(f0,width=900,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            tabview.place(x=300,y=100)
            animate_frame(tabview)
            screenshots = []  
            mid_term=CTkButton(tabview,hover_color="#D9D9D0",text="Mid Term",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            mid_term.configure(command=lambda :show_grade_stu("Mid Term",mid_term,final_term))
            mid_term.place(x=280,y=10) 
            final_term=CTkButton(tabview,hover_color="#D9D9D0",text="Final Term",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            final_term.configure(command=lambda :show_grade_stu("Final Term",final_term,mid_term))
            final_term.place(x=460,y=10)
            mid_term.invoke() 

        def notes_frame():
            grade_bt.configure(fg_color="#33CCFF")
            taught_bt.configure(fg_color="#33CCFF")
            notes_bt.configure(fg_color="#888888")
            tabview = CTkFrame(f0,width=900,height=560,fg_color="#ffcc99",border_width=3,corner_radius=12,border_color="black")
            tabview.place(x=300,y=100)
            animate_frame(tabview)
            sub_f = CTkFrame(tabview,width=240,height=515,fg_color="#b3d9ff",border_width=3,corner_radius=12,border_color="black")
            sub_f.place(x=30,y=20)
            #fetching classes
            cursor=db.cursor()
            cursor.execute("select sub_name from subject")
            data=cursor.fetchall()
            s_nm=[]
            for i in data:
                    std_name=i.get("sub_name")
                    s_nm.append(std_name)
            def show_pdf(id):
                    cursor=db.cursor()
                    cursor.execute('select path from notes where note_id=%s',(id))
                    data=cursor.fetchone()
                    path = data["path"]
                    edge_path="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
                    webbrowser.get('edge').open(path)
                    
            def show_notes(btn,sub_name):
                for i in buttons:
                    if btn==i:
                        btn.configure(fg_color="#888888")
                    else:
                        i.configure(fg_color="#33CCFF")
                scroll_f=CTkScrollableFrame(tabview,width=540,corner_radius=20,border_color='black',border_width=2,fg_color="#ccffcc",height=470)
                scroll_f.place(relx=0.65,rely=0.5,anchor=CENTER)
                cursor=db.cursor()
                cursor.execute("Select note_id,notes.title from notes where notes.sub_code=(select sub_id from subject where sub_name=%s) and std_code=%s",(sub_name,Std))
                data=cursor.fetchall()
                r=0
                y_pad=0
                icon=CTkImage(Image.open("notebook.png"),size=(40,40))
                for i in range(len(data)):
                    note_tl=data[i]['title']
                    note_id=data[i]['note_id']
                    new_b=CTkButton(scroll_f,hover_color="#D9D9D0",command=lambda id=note_id: show_pdf(id),text=note_tl,height=55,width=390,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20),compound='right')
                    new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                    animate_text(new_b,25)
                    new_l=CTkButton(new_b,image=icon,text='',height=40,command=lambda id=note_id: show_pdf(id),hover_color="#D9D9D0",width=40,text_color="black",fg_color="#33CCFF")
                    new_l.place(x=70,rely=0.5,anchor="e")
                    r+=1
            r=45
            buttons=[]
            for i in s_nm:
                new_b=CTkButton(sub_f,hover_color="#D9D9D0",text=i,height=45,width=180,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
                new_b.place(x=120,y=r,anchor=CENTER)
                animate_text(new_b,25)
                new_b.configure(command=lambda btn=new_b,param=i: show_notes(btn,param))
                buttons.append(new_b)
                r+=70
            buttons[0].invoke()
        date_time_display()
        l2=CTkLabel(f0,text="Academic information",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        animate_frame(f3)
        taught_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: taught_by(),text="Taught by",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        taught_bt.place(x=120,y=45,anchor=CENTER)
        animate_text(taught_bt,25)
        grade_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: grade_frame(),text="Grades",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        grade_bt.place(x=120,y=110,anchor=CENTER)
        animate_text(grade_bt,25)
        notes_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: notes_frame(),text="Notes",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        notes_bt.place(x=120,y=175,anchor=CENTER)
        animate_text(notes_bt,25)
        taught_bt.invoke()


    #-------------------------------------------------------------complain_frame----------------------------------------------------------------


    #Complain_frame
    def complain_frame():
        date_time_display()
        l2=CTkLabel(f0,text=("Complain Section"),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=55,y=20)
        #animate_text(l2,20)

        new_f=CTkFrame(f0,width=675,height=585,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
        new_f.place(x=55,y=100)
        animate_frame(new_f)
        sub_lb=CTkLabel(new_f,text="Subject",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        sub_lb.place(x=20,y=20)
        sub_et=CTkEntry(new_f,height=40,width=450,border_width=3,corner_radius=30,font=("Roboto",18))
        sub_et.place(x=200,y=22)

        #list with all teacher's names
        cursor=db.cursor()
        cursor.execute("Select name from teacher")
        data=cursor.fetchall()
        teach_name=[]
        for i in data:
            teacher_name=i.get("name")
            teach_name.append(teacher_name)
        
        to_lb=CTkLabel(new_f,text="To",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        to_lb.place(x=20,y=90)
        to_op=CTkOptionMenu(new_f,height=30,width=190,values=["Admin","All Teachers"]+teach_name)
        to_op.place(x=210,y=95)
        dept_lb=CTkLabel(new_f,text="Department",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        dept_lb.place(x=20,y=155)
        dept_op=CTkOptionMenu(new_f,height=30,width=190,values=["Faculty","Administration","Campus Security","IT Services","Library","Food Services","Maintenance","Student Affairs","Logistics","Sports & Activities","Others"])
        dept_op.place(x=210,y=160)
        desc_lb=CTkLabel(new_f,text="Description",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
        desc_lb.place(x=20,y=220)
        anon_switch=CTkSwitch(new_f,switch_height=30,switch_width=60,text="Hide",font=CTkFont("Helvetica",20))
        anon_switch.place(x=550,y=240)
        desc_et=CTkTextbox(new_f,corner_radius=10,height=250,width=635,border_width=2,font=("Roboto",18))
        desc_et.place(x=20,y=280)
        id_lb=CTkLabel(new_f,text="Complain id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
        id_lb.place(x=15,y=540)
        id_et=CTkEntry(new_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
        id_et.place(x=155,y=540)
        cursor=db.cursor()
        cursor.execute("select max(complain_id)+1 from complain")
        data=cursor.fetchall()
        auto=data[0]
        auto_id=auto.get("max(complain_id)+1")
        id_et.insert(0,auto_id)
        id_et.configure(state="disabled")
        def send():
            flash_massa=CTkLabel(new_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
            flash_massa.place(relx=0.5,y=540)
            c_id=id_et.get()
            sub=sub_et.get()
            to=to_op.get()
            depart=dept_op.get()
            anon=anon_switch.get()
            desc=desc_et.get("1.0", "end-1c")
            flag_check=True
            if len(sub)==0 or len(desc)==0:
                flag_check=False
                flash_massa.configure(text_color="red")
                flash_message("Try Again",flash_massa)
            if flag_check==True:
                cursor=db.cursor()
                cursor.execute("insert into complain (complain_id, stud_code, subject, `to`, depart, description,hide) values(%s,%s,%s,%s,%s,%s,%s)",(c_id,student_id,sub,to,depart,desc,anon))
                db.commit()
                all_complains()
                #wipping the textbox
                id_et.configure(state="normal")
                id_et.delete(0,END)
                id_et.insert(0,auto_id+1)
                id_et.configure(state="disabled")
                sub_et.delete(0,END)
                desc_et.delete(1.0,END)
                flash_massa.configure(text_color="green")
                flash_message("Sent",flash_massa)
                
        send_b=CTkButton(new_f,command=send,text="Send",height=35,width=100,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        send_b.place(x=540,y=540)

        #All complains
        new_f=CTkFrame(f0,width=455,height=585,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
        new_f.place(x=750,y=100)
        animate_frame(new_f)
        def solution(comp_id):
            cursor=db.cursor()
            cursor.execute("Select description,solution from complain where complain_id=%s",(comp_id))
            data=cursor.fetchall()
            desc=data[0]["description"]
            sol=data[0]["solution"]
            desc_f=CTkFrame(new_f,width=422,height=490,fg_color="#FFE5B4",border_width=0,corner_radius=12)
            desc_f.place(relx=0.5,rely=0.55,anchor=CENTER)
            sol_t=CTkTextbox(desc_f,font=CTkFont("Helvetica",20),width=400,height=230,fg_color="#e6f7ff",border_width=3,corner_radius=12,border_color="black")
            sol_t.place(x=10,y=250)
            desc_t=CTkTextbox(desc_f,font=CTkFont("Helvetica",20),width=400,height=230,fg_color="#ffff99",border_width=3,corner_radius=12,border_color="black")
            desc_t.place(x=10,y=10)
            if sol==None:
                desc_t.delete('0.0','end')
                desc_t.insert('0.0',desc)
                desc_t.configure(state="disabled")
                sol_t.delete('0.0','end')
                sol_t.insert('0.0',"Action in Process")
                sol_t.configure(state="disabled")
            else: 
                desc_t.delete('0.0','end')
                desc_t.insert('0.0',desc)
                desc_t.configure(state="disabled")
                sol_t.delete('0.0','end')
                sol_t.insert('0.0',sol)
                sol_t.configure(state="disabled")
            all=CTkLabel(new_f,text=" ",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#FFE5B4",font=CTkFont("Helvetica",20))
            all.place(x=20,y=20)
            #back button
            photo1=CTkImage(Image.open("back.png"),size=(40,40))
            edit_b=CTkButton(new_f,command=all_complains,image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
            edit_b.place(x=20,y=18)
            sol_lb=CTkLabel(new_f,text="Solution",height=45,width=170,corner_radius=20,text_color="black",fg_color="#e4d1d1",font=CTkFont("Helvetica",20))
            sol_lb.place(x=250,y=20)  
        def all_complains():
            all=CTkLabel(new_f,text="All Complains",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            all.place(x=20,y=20) 
            sol_lb=CTkLabel(new_f,text=" ",height=45,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
            sol_lb.place(x=250,y=20)
            scroll_f=CTkScrollableFrame(new_f,width=380,corner_radius=20,fg_color="#B6E5D8",height=450)
            scroll_f.place(relx=0.5,rely=0.55,anchor=CENTER)
            cursor=db.cursor()
            cursor.execute("Select complain_id,depart from complain where stud_code=%s",(student_id))
            data=cursor.fetchall()
            r=0
            y_pad=0
            for i in range(len(data)):
                sub=data[i]['complain_id']
                sub=str(sub)
                dep=data[i]['depart']
                new_b=CTkButton(scroll_f,command=lambda param=sub: solution(param),hover_color="#D9D9D0",text=sub+"     "+dep,height=45,width=330,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                r+=1
        all_complains()



    #------------------------------------------------------Attendance Frame---------------------------------------------------------
    def attendance_frame():
        date_time_display()
        l2=CTkLabel(f0,text=("Detailed Attendance"),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=55,y=20)
        new_f=CTkFrame(f0,width=1040,height=531,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
        new_f.place(relx=0.44,rely=0.58,anchor=CENTER)
        animate_frame(new_f)
        cursor.execute("SELECT attend_sheet, sub_code FROM attendance WHERE stand_code = %s ORDER BY attendance_date DESC", (Std,))
        b_data = cursor.fetchall()

        subject_lec = [0, 0, 0, 0, 0, 0, 0]
        tot_attendance = [0, 0, 0, 0, 0, 0, 0]

        for i in range(len(b_data)):
            data1 = b_data[i]
            sub = data1["sub_code"]
            attendance_list = list(data1["attend_sheet"])
        

            if sub == 1:
                subject_lec[0] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[0] += 1
            elif sub == 2:
                subject_lec[1] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[1] += 1
            elif sub == 3:
                subject_lec[2] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[2] += 1
            elif sub == 4:
                subject_lec[3] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[3] += 1
            elif sub == 5:
                subject_lec[4] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[4] += 1
            elif sub == 6:
                subject_lec[5] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[5] += 1
            elif sub == 7:
                subject_lec[6] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[6] += 1

        
        
        tot_subject = ['  English', '   Hindi', ' Marathi', '   Maths', ' Science', 'Social sci', 'phy.training']

        def arrange_graphs_horizontally():
            num_graphs = len(tot_subject)
            cols = 4
            row_height = 220
            col_width = 200
            x_padding = 20
            y_padding = 20
            x_space = 60  
            y_space = 40  

            for i in range(num_graphs):
                row = i // cols
                col = i % cols
                attend_per = int((tot_attendance[i] / subject_lec[i]) * 100)
                x_offset = (col * (col_width + x_space)) + x_padding
                y_offset = (row * (row_height + y_space)) + y_padding
                if i>3:
                    frame_subjects = CTkFrame(new_f, height=row_height, width=col_width, corner_radius=15, fg_color="white", bg_color="#FFE5B4")
                    frame_subjects.place(x=x_offset +130, y=y_offset)
                    animate_bot_frame(frame_subjects,15)
                else:
                    frame_subjects = CTkFrame(new_f, height=row_height, width=col_width, corner_radius=15, fg_color="white", bg_color="#FFE5B4")
                    frame_subjects.place(x=x_offset, y=y_offset)
                    animate_bot_frame(frame_subjects,15)
                if attend_per > 75:
                    progress_color = "green"
                elif attend_per > 50 and attend_per <= 75:
                    progress_color = "yellow"
                else:
                    progress_color = "red"

                if attend_per >25 and attend_per <50:
                    end=100
                elif attend_per <=50:
                    end=180
                elif attend_per<=60 and attend_per >50:
                    end=200
                elif attend_per<=75 and attend_per >60:
                    end=300
                elif attend_per<=80 and attend_per >75:
                    end=400
                elif attend_per<=90 and attend_per >80:
                    end=500
                elif attend_per>90 and attend_per<95:
                    end=750

                
                if attend_per == 100:
                    knob = ScrollKnob(master=frame_subjects, text="", steps=10, radius=110, bar_color="#fffcfc",outer_color="green", 
                                    outer_length=10,border_width=30, start_angle=90, inner_width=0,outer_width=20,text_font="calibri 1", 
                                    text_color="black", fg="#fffcfc", end=100, progress_color="white", start=0)
                else:
                    knob = ScrollKnob(master=frame_subjects, text="", steps=10, radius=160, bar_color="#fffcfc",outer_color="#fffcfc", 
                                    outer_length=10,border_width=30, start_angle=90, inner_width=0, outer_width=2,text_font="calibri 1", 
                                    text_color="black", fg="#fffcfc", end=end, progress_color=progress_color, start=0)
                knob.place(relx=0.5,rely=0.4, anchor="center") 

                knob.set(attend_per)
                sub_label = CTkLabel(knob, text=("  "+str(attend_per) + "%" + "\n" + str(tot_subject[i])))
                sub_label.place(relx=0.5,rely=0.5,anchor=CENTER)  
                sep_label=CTkLabel(frame_subjects, text="- - - - - - - - - - - - - - -",font=("Arial Bold", 15))
                sep_label.place(x=40, y=row_height - 65)
                hello_label = CTkLabel(frame_subjects, text=(" Total    : " + str(subject_lec[i]) + " " + "\n Present : " + str(tot_attendance[i]) + " "),font=("Arial Bold", 15), text_color="black")
                hello_label.place(x=45, y=row_height - 40)  


        arrange_graphs_horizontally()

    #------------------------------------------------------student window code ----------------------------------------------------------------------

    #student window
    def main_window():
        global f0,student_id,home_indicate,search_indicate,schedule_indicate,complain_indicate,student_win,attendance_indicate

        set_appearance_mode("light")
        set_default_color_theme("blue")
        student_win=CTk()
        
        student_win.title("Student home page")
        screen_width = student_win.winfo_screenwidth()
        screen_height= student_win.winfo_screenheight()
        student_win_width = screen_width
        student_win_height = screen_height
        student_win.geometry(f"{student_win_width}x{student_win_height}")
        student_win.geometry("+0+0")
        # student_win.maxsize(width=1400,height=750)
        # student_win.minsize(width=1400,height=750) 
        student_win.attributes('-fullscreen',True)
        student_win.iconbitmap("logo_icon.ico")
        
        frame=CTkFrame(student_win,width=1900,height=1000,fg_color="#66B3FF")
        frame.pack()
        #Home frame
        f0=CTkFrame(frame,width=1400,height=700,fg_color="#66B3FF")
        f0.place(x=140,y=30)
        #Dashboard
        f1=CTkFrame(frame,width=100,height=680,fg_color="white",border_width=3,corner_radius=15,border_color="black")
        f1.place(x=50,y=35)
        #logo
        photo=CTkImage(Image.open("logo.png"),size=(60,60))
        l1=CTkLabel(f1,image=photo,text=" ")
        l1.place(x=18,y=40)

        #home indicator
        home_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
        home_indicate.place(x=7,y=150)
        
        search_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
        search_indicate.place(x=7,y=250)
        
        schedule_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
        schedule_indicate.place(x=7,y=350)

        complain_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
        complain_indicate.place(x=7,y=450)
        
        attendance_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
        attendance_indicate.place(x=7,y=550)
        
        #to initialize the student_win
        indicate(home_indicate,home_frame)
        #home button
        photo1=CTkImage(Image.open("home.png"),size=(50,50))
        b1=CTkButton(f1,command=lambda: indicate(home_indicate,home_frame),image=photo1,text="",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
        b1.place(x=17,y=150)
        #search button
        photo2=CTkImage(Image.open("search_logo.png"),size=(50,50))
        b2=CTkButton(f1,command=lambda: indicate(search_indicate,search_frame),image=photo2,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
        b2.place(x=15,y=250)
        ##taught_by button 
        photo3=CTkImage(Image.open("Schedule.png"),size=(50,50))
        b2=CTkButton(f1,command=lambda: indicate(schedule_indicate,academic_frame),image=photo3,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
        b2.place(x=15,y=350)
        #Complain button
        photo5=CTkImage(Image.open("report.png"),size=(50,50))
        b2=CTkButton(f1,command=lambda: indicate(complain_indicate,complain_frame),image=photo5,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
        b2.place(x=15,y=450)
        #Atttendance button
        photo5=CTkImage(Image.open("detailed_attend.png"),size=(50,50))
        b2=CTkButton(f1,command=lambda: indicate(attendance_indicate,attendance_frame),image=photo5,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
        b2.place(x=15,y=550)
        student_win.mainloop()

    main_window()















#-----------------------------------------TEACHER_PAGE'S CODE STARTS HERE-----------------------------------------------------------







def teacher_page(teacher_id):


    #datetime
    def date_time_display():
        
        def ct_time():
            now = datetime.now()
            ct_string = now.strftime("%H:%M:%S")
            return ct_string

        def ct_change():
            ct_string = ct_time()
            time_lb.configure(text=ct_string)
            f0.after(1000, ct_change)  # update every 1 second
        #logout_frame
        def logout_frame():
            delete_frames()
            date_time_display()
            def destroy_window():
                teacher_win.destroy()
                login_page()
            log_lb=CTkLabel(f0,text="Do you want to logout ?",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
            log_lb.place(relx=0.5,y=200,anchor=CENTER)
            log_bu=CTkButton(f0,text="Yes",height=45,command=destroy_window,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            log_bu.place(relx=0.5,y=300,anchor=CENTER)

        today = datetime.today()
        t_date= today.strftime("%B %d, %Y")
        #date and time 
        d_f=CTkFrame(f0,width=350,height=50,border_color="black",border_width=3,fg_color="white",corner_radius=40)
        d_f.place(x=750,y=5)
        time_lb=CTkLabel(d_f,width=110,height=30,text="",font=CTkFont("Helvetica",19),fg_color="white",corner_radius=40,text_color="black")
        time_lb.place(relx=0.8,rely=0.5,anchor=CENTER)
        date_lb=CTkLabel(d_f,text=t_date,width=150,height=30,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="white")
        date_lb.place(relx=0.3,rely=0.5,anchor=CENTER)
        ct_change()

        photo1=CTkImage(Image.open("logout1.png"),size=(50,50))
        edit_b=CTkButton(f0,image=photo1,command=logout_frame,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
        edit_b.place(x=1120,y=0)


    #hide indicators
    def hide_indicators():
        home_indicate.configure(fg_color="white")
        student_indicate.configure(fg_color="white")
        teacher_indicate.configure(fg_color="white")
        grade_indicate.configure(fg_color="white")
        complain_indicate.configure(fg_color="white")

    #to delete frames
    def delete_frames():
        for f in f0.winfo_children():
            f.destroy()

    def id_card(teacher_id):
        global Name
        cursor=db.cursor()
        query1=f"Select * from teacher where teacher_id={teacher_id}"
        cursor.execute(query1)
        data0=cursor.fetchall()
        data1=data0[0]
            
        tr_id=data1["teacher_id"]
        Name=data1["name"]
        Dob=data1["dob"]
        quali=data1["quali"]
        Address=data1["address"]
        mobile=data1["phone_no"]
        gender=data1["gen"]
        
        l2=CTkLabel(f0,text=("Welcome "+ Name),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=55,y=20)
        
        f7=CTkFrame(f0,width=548,height=338,fg_color="#ffffe6",border_width=3,corner_radius=12,border_color="black")
        f7.place(x=580,y=100)
        photo5=CTkImage(Image.open("id_back.png"),size=(538,160))
        l3=CTkLabel(f7,image=photo5,text="")
        l3.place(x=5,y=5)
        

        photo6=CTkImage(Image.open("male_teach.png"),size=(150,150))
        if gender=="M":
            pass
        else:
            photo6=CTkImage(Image.open("female_teach.png"),size=(150,150))
        l4=CTkLabel(f7,image=photo6,text=" ",fg_color="transparent")
        l4.place(x=315,y=115)

        photo7=CTkImage(Image.open("logo.png"),size=(70,70))
        l5=CTkLabel(f7,image=photo7,text=" ",bg_color='#6FD0FE')
        l5.place(x=60,y=15)
        l6=CTkLabel(f7,text="IDENTITY CARD",bg_color='#6FD0FE',font=("Helvetica",25),text_color="black")
        l6.place(x=170,y=33)
        id_lb=CTkLabel(f7,text=("TEACHER.ID :  " + str(tr_id)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        id_lb.place(x=15,y=115)
        name_lb=CTkLabel(f7,text=("NAME :   " + Name),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        name_lb.place(x=15,y=150)
        dob_lb=CTkLabel(f7,text=("DOB    :  " + str(Dob)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        dob_lb.place(x=15,y=185)
        std_lb=CTkLabel(f7,text=("Quali  : "+ quali),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        std_lb.place(x=15,y=220)
        mobile_lb=CTkLabel(f7,text=("MOBILE NO. :  " + str(mobile)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
        mobile_lb.place(x=15,y=255)
        address_lb=CTkLabel(f7,text=("ADDRESS :  " + Address),width=100,height=35,corner_radius=8,font=("Helvetica",15),wraplength=500,text_color="black",bg_color="#ffffe6")
        address_lb.place(x=15,y=290)


    #----------------------------------------------------------------home_frame----------------------------------------------------------------



    #home_frame
    def home_frame():
        global board1
        date_time_display()
        id_card(teacher_id)

        cursor = db.cursor()
        grade_levels = ["5th std", "6th std", "7th std", "8th std", "9th std", "10th std"]
        positions = [(40, 98), (40, 230), (40, 360), (300, 98), (300, 230), (300, 360)]
        fg_colors = ["#FFFFCC", "#C7FAC7", "#FFFFCC", "#C7FAC7", "#FFFFCC", "#C7FAC7"]

        for i, grade_level in enumerate(grade_levels):
            cursor.execute("SELECT COUNT(*) AS count FROM student WHERE std_code = %s", (grade_level,))
            data = cursor.fetchone()
            total_students = data["count"]
            frame = CTkFrame(f0, width=240, height=110, fg_color=fg_colors[i], border_width=3, corner_radius=12, border_color="black")
            frame.place(x=positions[i][0], y=positions[i][1])
            label_count = CTkLabel(frame, text=total_students, font=CTkFont(family="Helvetica", weight="bold", size=50), text_color="black")
            label_count.place(x=20, y=10)
            label_grade = CTkLabel(frame, text=grade_level, font=CTkFont(family="Helvetica", weight="bold", size=25), text_color="black")
            label_grade.place(x=20, y=75)

        #quotes
        def quotes():
            try:
                # list of quotes
                quotes = [
                    "Be the change you wish to see in the world.",
                    "The only way to do great work is to love what you do.",
                    "Not all who wander are lost.",
                    "Believe you can and you're halfway there.",
                    "If you want to go fast, go alone. If you want to go far, go together.",
                    "It does not matter how slowly you go as long as you do not stop.",
                    "Life is what happens when you're busy making other plans.",
                    "It is during our darkest moments that we must focus to see the light.",
                    "You must be the change you wish to see in the world.",
                    "The future belongs to those who believe in the beauty of their dreams.",
                    "The best way to predict the future is to invent it."]

                # randomly select a quote from the list
                selected_quote = random.choice(quotes)

                f6=CTkFrame(f0,width=1140,height=100,fg_color="#ffe6cc",border_width=3,corner_radius=12,border_color="black")
                f6.place(x=40,y=520)
                quotes_lb=CTkLabel(f6,text="Thought of the day!",width=50,height=20,font=("bold",30),text_color="black",bg_color="transparent")
                quotes_lb.place(relx=0.5,rely=0.3,anchor=CENTER)
                label=CTkLabel(f6,text=selected_quote ,width=40,height=20,font=("bold",18),text_color="black",bg_color="transparent")
                label.place(relx=0.5,rely=0.7,anchor=CENTER)
                animate_text(label,25)
                quotes_wall=CTkImage(Image.open("qoutes_bg.png"),size=(70,50))
                quotes_wlb=CTkLabel(f6,text="",image=quotes_wall,width=50,height=20,bg_color="transparent")
                quotes_wlb.place(relx=0.05,rely=0.3,anchor=CENTER)
            except Exception as e:
                pass
        quotes()




    #------------------------------------------------------------------Attendance Frame----------------------------------------------------------------





    #attendance_frame
    def attendance_frame():
        #date_time_display()
        l2 = CTkLabel(f0, text="Student Attendance", font=CTkFont(family="Helvetica", weight="bold", size=50),text_color="black")
        l2.place(x=40, y=30)
        #animate_text(l2,25)
        def attend_view(btn, val):
            for i in buttons_attendance:
                if btn == i:
                    btn.configure(fg_color="#888888")
                else:
                    i.configure(fg_color="#33CCFF")
            
            cursor = db.cursor()
            cursor.execute("select sub_code from teach_class where teacher_code=%s and std_code=%s", (teacher_id,val))
            sub = cursor.fetchall()
            sub1=sub[0]["sub_code"]
            
            cursor.execute("select sub_name from subject where sub_id=%s", (sub1))
            data3 = cursor.fetchall()
            sub_name=data3[0]["sub_name"]

            open_f = CTkFrame(f0, width=900, height=560, fg_color="#FFE5B4", border_width=3, corner_radius=12,border_color="black")
            open_f.place(x=295, y=100)
            animate_frame(open_f)
            sr_lbl = CTkLabel(open_f,text="Sr.",height=45,width=40,corner_radius=20,text_color="black", fg_color="#ccffe6", font=CTkFont("Helvetica", 18))
            sr_lbl.place(x=30, y=13)
            id_lbl = CTkLabel(open_f,text="ID",height=45,width=100,corner_radius=20,text_color="black",fg_color="#ccffe6", font=CTkFont("Helvetica", 18))
            id_lbl.place(x=125, y=13)
            nm_lbl = CTkLabel(open_f,text="Name",height=45,width=150,corner_radius=20,text_color="black",fg_color="#ccffe6", font=CTkFont("Helvetica", 18))
            nm_lbl.place(x=270, y=13)
            pre_lbl = CTkLabel(open_f,text="Previous",height=45,width=150,corner_radius=20,text_color="black",fg_color="#ccffe6", font=CTkFont("Helvetica", 18))
            pre_lbl.place(x=450, y=13)
            today = datetime.today()
            to_date= today.strftime("%d/%m/%Y")
            tod_lbl = CTkLabel(open_f, text=f"Today ({to_date})", height=45, width=200, corner_radius=20, text_color="black",fg_color="#ccffe6", font=CTkFont("Helvetica", 18))
            tod_lbl.place(x=650, y=13)
            animate_text(tod_lbl,25)
            flash_massa=CTkLabel(open_f,text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
            flash_massa.place(relx=0.5, rely=0.95,anchor=CENTER)

            scroll_f = CTkScrollableFrame(open_f, width=820, height=400, corner_radius=20,border_width=2,border_color="#3941B8", fg_color="#B6E5D8")
            scroll_f.place(x=20, y=65)

            subject=CTkLabel(open_f,text=sub_name,height=35,width=100,corner_radius=20,text_color="black",fg_color="#ccffe6", font=CTkFont("Helvetica", 17))
            subject.place(x=30, rely=0.92)
            

            def insert_att(list1,p,flag):
                cursor = db.cursor()
                cursor.execute("select attend_sheet,attendance_date from attendance where stand_code=%s and sub_code=%s order by attendance_date;", (val,sub1))
                data = cursor.fetchall()
                if len(data) == 0:
                    pass
                else:
                    data_sheet = data[-p]["attend_sheet"]
                    for i in range(len(data_sheet)):
                        status=data_sheet[i]
                        list1[i].delete(0,END)
                        if status=="0":
                            list1[i].configure(fg_color="#ffcccc")
                            list1[i].configure(state="normal")
                            list1[i].insert(0,status )
                        else:
                            list1[i].configure(state="normal",fg_color="#c8f7ea")
                            list1[i].insert(0,status )
                        if flag=="dis":
                            list1[i].configure(state="disabled")

            def selected_insertion():
                cursor = db.cursor()
                cursor.execute("select attendance_date from attendance where stand_code=%s and sub_code=%s order by attendance_date;", (val,sub1))
                data = cursor.fetchall()
                if len(data) == 0:
                    flash_massa.configure(text_color="red")
                    flash_message("Attendance not taken",flash_massa)
                else:
                    data_date=str(data[-1]["attendance_date"])
                    today = datetime.today()
                    to_date = today.strftime("%Y-%m-%d")
                    
                    if data_date == to_date:
                        insert_att(att_mark_list,1,"nr")
                        if len(data)>1:
                            insert_att(pre_mark_list,2,"dis")
                    else:
                        insert_att(pre_mark_list,1,"dis")            

            #submit attendance_sheet
            def submit_sheet():

                sv_mks.configure(text="Saved",fg_color="#00ace6",hover_color="#0fb9f2")
                result = ""
                check_val=""
                for entry in att_mark_list:
                    value = entry.get()
                    if value == "":
                        check_val+=""
                        result += "0" 
                    else:
                        check_val+="@"
                        result += value[0]
                if len(check_val) == 0:
                    flash_massa.configure(text_color="red")
                    flash_message("Attendance not entered",flash_massa)
                else:
                    for entry in att_mark_list:
                        value = entry.get()
                        if value == "":
                            entry.configure(fg_color="#ffcccc")
                            entry.insert(0,"0")
                    today = datetime.today()
                    to_date = today.strftime("%Y-%m-%d")
                    try:
                        cursor = db.cursor()
                        cursor.execute("insert into attendance VALUES(%s, %s, %s, %s)",(val,sub1, to_date, result))
                        db.commit()
                        flash_massa.configure(text_color="green")
                        flash_message("Attendance Recorded",flash_massa)
                    except pymysql.err.OperationalError:
                        cursor.execute("UPDATE attendance SET attend_sheet = %s WHERE stand_code = %s AND sub_code = %s AND attendance_date = %s",(result,val,sub1, to_date ))
                        db.commit()
                        insert_att(att_mark_list,1,"nr")
                        flash_massa.configure(text_color="green")
                        flash_message("Attendance Updated",flash_massa)
                """except pymysql.err.OperationalError:
                flash_massa.configure(text_color="red")
                flash_message("Attendance already taken",flash_massa) """           

            

            
            cursor = db.cursor()
            cursor.execute("select stud_id,name from student where std_code=%s", (val,))
            data = cursor.fetchall()
            id_list = []
            name_list = []
            for i in data:
                id_list.append(i['stud_id'])
                name_list.append(i['name'])

            r = 1
            pre_mark_list = []
            att_mark_list = []
            for i in range(len(data)):
                sr_nm = CTkLabel(scroll_f, text=i+1, height=45, width=40, text_color="black", fg_color="#B6E5D8",font=CTkFont("Helvetica", 19))
                sr_nm.grid(row=r, column=0, padx=10 , pady= 5)
                s_id = CTkLabel(scroll_f, text=id_list[i], height=45, width=40, text_color="black", fg_color="#B6E5D8",font=CTkFont("Helvetica", 19))
                s_id.grid(row=r, column=1,padx=45, pady= 5)
                s_nm = CTkLabel(scroll_f, text=name_list[i], height=45, width=40, text_color="black", fg_color="#B6E5D8",font=CTkFont("Helvetica", 19))
                s_nm.grid(row=r, column=2,padx=40, pady= 5)
                pre_mark = CTkEntry(scroll_f, height=45,state="disabled",width=65,justify="center", text_color="black", fg_color="#C5C5C5",font=CTkFont("Helvetica", 19))
                pre_mark.grid(row=r, column=3,padx=(40,65), pady= 5)
                pre_mark_list.append(pre_mark)
                att_mark = CTkEntry(scroll_f, height=45, width=65,justify="center", text_color="black", fg_color="#c8f7ea",font=CTkFont("Helvetica", 19))
                att_mark.grid(row=r, column=4,padx=(75,0), pady= 5)
                att_mark_list.append(att_mark)

                r += 1
            
            sv_mks = CTkButton(open_f,command=submit_sheet, hover_color="#D9D9D0", text="Save", height=40, width=170, border_width=2,corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF",font=CTkFont("Helvetica", 20))
            sv_mks.place(x=705, y=513)
            selected_insertion()
            

        f3 = CTkFrame(f0, width=240, height=560, fg_color="white", border_width=3, corner_radius=12, border_color="black")
        f3.place(x=40, y=100)
        animate_frame(f3)
        cursor = db.cursor()
        cursor.execute("SELECT class.std_id FROM teacher, class, teach_class, subject WHERE teacher.teacher_id = teach_class.teacher_code AND class.std_id = teach_class.std_code AND subject.sub_id = teach_class.sub_code AND teacher.teacher_id = %s ORDER BY CAST(class.std_id AS UNSIGNED) ASC", (teacher_id,))
        data = cursor.fetchall()
        std_id_list = []
        for i in data: 
            std_name = i.get("std_id")
            std_id_list.append(std_name)
        
        buttons_attendance = []
        r = 45
        for i in std_id_list:
            new_b = CTkButton(f3, hover_color="#D9D9D0", text=i, height=45, width=170, border_width=2, corner_radius=20,border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
            new_b.place(relx=0.5, y=r,anchor=CENTER)
            animate_text(new_b,40)
            new_b.configure(command=lambda btn=new_b, param=i: attend_view(btn, param))
            buttons_attendance.append(new_b)
            r += 70
        buttons_attendance[0].invoke()
        teacher_win.mainloop()





    #----------------------------------------------------------------Teacher Workspace Frame------------------------------------------------










    #teacher_workspace_frame
    def timetable_frame():
        date_time_display()
        def treeview():
            f9=CTkFrame(f0,width=900,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
            f9.place(x=300,y=100)
            cursor=db.cursor()
            cursor.execute("select class.std_id,subject.sub_name from teacher,class,teach_class,subject where teacher.teacher_id=teach_class.teacher_code and class.std_id=teach_class.std_code and subject.sub_id=teach_class.sub_code and teacher.teacher_id=%s",(teacher_id))
            data=cursor.fetchall()
            std_i=[]
            subject_name=[]

            for i in data:
                    teacher_name=i.get("std_id")
                    sub_name=i.get("sub_name")
                    std_i.append(teacher_name)
                    subject_name.append(sub_name)

            stu_table=ttk.Treeview(f9,columns=("std_i","subject_name"),show="headings",height=10)
            style=ttk.Style(f9)
            
            style.theme_use("clam")
            style.configure("Treeview",rowheight=49,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
            style.configure("Treeview.Heading",font=("Roboto"))
            stu_table.heading("std_i",text="Standard")
            stu_table.heading("subject_name",text="Subject")
            stu_table.column("std_i",width=430,anchor=CENTER)
            stu_table.column("subject_name",width=430,anchor=CENTER)

            for i in range(len(std_i)-1,-1,-1):
                    stu_table.insert(parent="",index=0,values=(std_i[i],subject_name[i]))
            stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)   
        #teacher_section
        l2=CTkLabel(f0,text="Teacher's Workspace",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)

        #frame for buttons 
        def hide_hover():
            tt_lb.configure(fg_color="#33CCFF")
            stand_6.configure(fg_color="#33CCFF")
            notes_bt.configure(fg_color="#33CCFF")
        
        #view details
        def time_tb():
            hide_hover()
            tt_lb.configure(fg_color="#888888")
            f8=CTkFrame(f0,width=900,height=560,fg_color="#ffffb3",border_width=3,corner_radius=12,border_color="black")
            f8.place(x=300,y=100)
            time_lb=CTkLabel(f8,fg_color="#FFD586",height=50,width=150,corner_radius=20,text="Teachers Assignments",font=CTkFont(family="Helvetica",size=30),text_color="black")
            time_lb.place(x=300,y=20)

            #subject and class
            cursor=db.cursor()
            subject=['Eng','Hin','Mara','Math','Sci','SS','PT']
            cursor.execute('select std_id from class')
            data=cursor.fetchall()
            newy=160
            newx=130
            for i in range(len(subject)):
                time_lb=CTkLabel(f8,fg_color="#FFD586",height=40,width=80,corner_radius=20,text=subject[i],font=CTkFont(family="Helvetica",size=20),text_color="black")
                time_lb.place(x=60,y=newy,anchor=CENTER)
                newy+=60
            for i in range(len(data)):
                time_lb=CTkLabel(f8,fg_color="#FFD586",height=40,width=80,corner_radius=20,text=data[i]['std_id'],font=CTkFont(family="Helvetica",size=20),text_color="black")
                time_lb.place(x=newx,y=80)
                newx+=125

            #fetching teachers_names
            newx=180
            for i in range(len(data)):
                cursor=db.cursor()
                cursor.execute("select teacher.name from teacher,teach_class where teach_class.std_code=%s and teach_class.teacher_code=teacher.teacher_id",(data[i]['std_id']))
                new_d=cursor.fetchall()
                teacher_list=[]
                for j in new_d:
                    teacher_list.append(j['name'])
                newy=160
                for i in range(len(teacher_list)):
                    time_lb=CTkLabel(f8,fg_color="#ffffb3",wraplength=80,height=45,corner_radius=20,text=teacher_list[i],font=CTkFont(family="Helvetica",size=16),text_color="black")
                    time_lb.place(x=newx,y=newy,anchor=CENTER)
                    newy+=60
                newx+=125

        tt_lb=CTkButton(f3,hover_color="#D9D9D0",command=time_tb,text="Mapping",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        tt_lb.place(x=120,y=45,anchor=CENTER)

        def my_sub():
            hide_hover()
            stand_6.configure(fg_color="#888888")
            treeview()
        stand_6=CTkButton(f3,hover_color="#D9D9D0",command=my_sub,text="My Classes",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        stand_6.place(x=120,y=120,anchor=CENTER)

        def notes():
            global select_lb
            hide_hover()
            notes_bt.configure(fg_color="#888888")
            new_f=CTkFrame(f0,fg_color='#ffcc99',width=900,height=560,border_width=3,corner_radius=12,border_color="black")
            new_f.place(x=300,y=100)
            flash_massa=CTkLabel(new_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#ffcc99")
            flash_massa.place(x=270,y=520,anchor=CENTER)
            title=CTkLabel(new_f,text="Title",height=45,width=130,corner_radius=20,text_color="black",fg_color= "#b3d9ff",font=CTkFont("Helvetica",20))
            title.place(x=20,y=20)
            tl_et=CTkTextbox(new_f,height=10,width=400,corner_radius=15,border_width=2,font=("Roboto",18))
            tl_et.place(x=20,y=70) 
            title=CTkLabel(new_f,text="Standard",height=45,width=130,corner_radius=20,text_color="black",fg_color= "#b3d9ff",font=CTkFont("Helvetica",20))
            title.place(x=20,y=163)
            all=CTkLabel(new_f,text="All Notes",height=45,width=200,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            all.place(x=570,y=20) 
            scroll_f=CTkScrollableFrame(new_f,width=400,corner_radius=20,border_color='black',border_width=2,fg_color="#ccffcc",height=430)
            scroll_f.place(relx=0.73,rely=0.55,anchor=CENTER)
            note=[]
            def show_pdf(n_id):
                cursor=db.cursor()
                cursor.execute("Select path from notes where note_id=%s",(n_id))
                data=cursor.fetchone()
                path = data["path"]
                edge_path="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
                webbrowser.get('edge').open(path)
            def all_notes():
                cursor=db.cursor()
                cursor.execute("Select notes.note_id,subject.sub_name,notes.title from subject join notes on subject.sub_id=notes.sub_code where notes.teacher_code=%s order by subject.sub_id",(teacher_id))
                data=cursor.fetchall()
                r=0
                y_pad=0
                for i in range(len(data)):
                    note_id=data[i]['note_id']
                    sub=data[i]['sub_name']
                    dep=data[i]['title']
                    new_b=CTkButton(scroll_f,hover_color="#D9D9D0",command=lambda param=note_id: show_pdf(param),text=sub+"     "+dep,height=45,width=330,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                    new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                    r+=1
            all_notes()

            #fetching classes
            cursor=db.cursor()
            cursor.execute("select class.std_id from teacher,class,teach_class,subject where teacher.teacher_id=teach_class.teacher_code and class.std_id=teach_class.std_code and subject.sub_id=teach_class.sub_code and teacher.teacher_id=%s",(teacher_id))
            data=cursor.fetchall()
            std_i=[]

            for i in data:
                std_name=i.get("std_id")
                std_i.append(std_name)
            std_op=CTkOptionMenu(new_f,width=160,height=30,values=std_i)
            std_op.place(x=243,y=185,anchor=CENTER)
            ch_fi=CTkLabel(new_f,text="Choose File",height=45,width=130,corner_radius=20,text_color="black",fg_color= "#b3d9ff",font=CTkFont("Helvetica",20))
            ch_fi.place(x=20,y=280)
            #browse files
            def browse():
                global path
                file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
                path=''
                if file_path:
                    select_lb.configure(text='file selected',text_color='green')
                    path=file_path
            
            #adding note
            def add_n():
                global path
                flag_check=True
                std_code=std_op.get()
                cursor=db.cursor()
                cursor.execute("select subject.sub_id from teacher,class,teach_class,subject where teacher.teacher_id=teach_class.teacher_code and class.std_id=teach_class.std_code and subject.sub_id=teach_class.sub_code and teacher.teacher_id=%s and class.std_id=%s",(teacher_id,std_code))
                data=cursor.fetchall()
                subject_name=[]
                for i in data:
                    sub_name=i.get("sub_id")
                    subject_name.append(sub_name)
                sub_code=subject_name[0]
                title=tl_et.get("1.0", "end-1c")
                lb_text=select_lb.cget("text")
                if len(title)==0:
                    flag_check=False
                elif lb_text=="No File Selected":
                    flag_check=False
                if flag_check==True:
                    cursor=db.cursor()
                    cursor.execute("insert into notes(std_code,teacher_code,sub_code,title,path) values(%s,%s,%s,%s,%s)",(std_code,teacher_id,sub_code,title,path))
                    db.commit()
                    all_notes()
                    flash_massa.configure(text_color='green')
                    flash_message('Note Added',flash_massa)
                    time.sleep(3)
                    notes()
                    
                else:
                    flash_massa.configure(text_color='red')
                    flash_message('Try Again',flash_massa)
                    
            file_lb=CTkButton(new_f,text="Browse",command= lambda :browse(),height=35,width=160,corner_radius=20,border_color='black',border_width=2,text_color="black",fg_color= "#cceeff",font=CTkFont("Helvetica",20))
            file_lb.place(x=170,y=285) 
            select_lb=CTkLabel(new_f,text="No File Selected",height=35,width=160,corner_radius=20,text_color="red",fg_color="#ffcc99",font=CTkFont("Helvetica",20))
            select_lb.place(x=163,y=330)
            add_note=CTkButton(new_f,command= lambda: add_n(),hover_color="#D9D9D0",text="Add Note",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            add_note.place(x=100,y=520,anchor=CENTER)
        notes_bt=CTkButton(f3,hover_color="#D9D9D0",command=notes,text="Notes",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        notes_bt.place(x=120,y=195,anchor=CENTER) 



    #to indicate
    def indicate(lb,frame):
        hide_indicators()
        lb.configure(fg_color="#0066ff")
        delete_frames()
        frame()





















    #----------------------------------------------------------------Grade Frames----------------------------------------------------------------






    #grade_frame
    def grade_frame():
        date_time_display()
        #asking to grade
        l2=CTkLabel(f0,text="Student Grades",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f3.place(x=40,y=100)
        animate_frame(f3)

        def get_grade(obt_mks):
            if obt_mks>=105:
                obt_grd="A+"
            elif obt_mks>=90:
                obt_grd="A"
            elif obt_mks>=75:
                obt_grd="B+"
            elif obt_mks>=60:
                obt_grd="B"
            elif obt_mks>=45:
                obt_grd="C"
            else:
                obt_grd="E"
            return obt_grd
        
        def passing_eligibility(mks_obt,mks):
            for i in mks_obt:
                if int(i.get())<mks:
                    i.configure(fg_color="#ffcccc")
                else:
                    i.configure(fg_color="#c8f7ea")

        def add_grades(btn,std):
            global flash_message,flash_massa
            for i in buttons_grade:
                if i==btn:
                    btn.configure(fg_color="#888888")
                else:
                    i.configure(fg_color="#33CCFF")
            open_f=CTkFrame(f0,width=900,height=560,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
            open_f.place(x=295,y=100)
            animate_frame(open_f)
            flash_massa=CTkLabel(open_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
            flash_massa.place(relx=0.5,rely=0.95,anchor=CENTER)
            all=CTkLabel(open_f,text="Examination Type",height=45,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
            all.place(x=20,y=20)
            option_exam=CTkOptionMenu(open_f,width=200,height=40,values=["Unit I","Mid Term","Unit II","Final Term"])
            option_exam.place(x=230,y=23)
            def show_student_list():
                e_t=option_exam.get()
                cursor=db.cursor()
                cursor.execute("select sub_code from teach_class where teacher_code=%s and std_code=%s",(teacher_id,std))
                subject=cursor.fetchall()
                sub_code=subject[0]['sub_code']
                if e_t=="Unit I" or e_t=="Unit II":#unit exam section
                    flash_massa=CTkLabel(open_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
                    flash_massa.place(relx=0.5,rely=0.95,anchor=CENTER)
                    
                    all=CTkLabel(open_f,text="",height=45,width=500,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                    all.place(x=20,y=20)
                    #back button
                    photo1=CTkImage(Image.open("back.png"),size=(40,40))
                    edit_b=CTkButton(open_f,command=lambda param=std,btn1=btn: add_grades(btn1,param),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
                    edit_b.place(x=20,y=20)
                    scroll_f=CTkScrollableFrame(open_f,border_width=2,border_color="#3941B8",width=820,corner_radius=20,fg_color="#B6E5D8",height=330)
                    scroll_f.place(x=20,y=123)
                    s_id=CTkLabel(open_f,text="id",height=45,width=140,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                    s_id.place(x=40,y=70)
                    s_nm=CTkLabel(open_f,text="Name",height=45,width=200,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                    s_nm.place(x=200,y=70)
                    ob_mk=CTkLabel(open_f,text="Marks Obtained",height=45,width=200,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                    ob_mk.place(x=430,y=70)
                    ob_gr=CTkLabel(open_f,text="Out of",height=45,width=200,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                    ob_gr.place(x=650,y=70)
                    ex_ty=CTkLabel(open_f,text=e_t,height=45,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                    ex_ty.place(relx=0.5,y=35,anchor=CENTER)
                    cursor=db.cursor()
                    cursor.execute("select * from grade where stud_code=(select stud_id from student where std_code=%s limit 1) and sub_code=%s and exam_type=%s",(std,sub_code,e_t))
                    count=cursor.fetchall()
                    if len(count)!=0:
                        et_mks.destroy()
                        cursor=db.cursor()
                        cursor.execute("select student.stud_id,student.name,grade.obt_mks from student,grade where student.stud_id=grade.stud_code and grade.exam_type=%s and grade.sub_code=%s and student.stud_id in (select stud_id from student where std_code=%s)",(e_t,sub_code,std))
                        data=cursor.fetchall()
                        id=[]
                        nms=[]
                        obt_mk=[]

                        for i in data:
                            id.append(i['stud_id'])
                            nms.append(i['name'])
                            obt_mk.append(i['obt_mks'])
                        r=1
                        for i in range(len(data)):
                            s_id=CTkLabel(scroll_f,text=id[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                            s_id.grid(row=r, column=1,padx=50, pady= 5)
                            s_nm=CTkLabel(scroll_f,text=nms[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                            s_nm.grid(row=r, column=2,padx=30, pady= 5)
                            r+=1
                        #marks_obtained
                        mks_obt=[]
                        r=1
                        for i in range(len(data)):
                            mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                            mk_e.grid(row=r,column=3,padx=80, pady=5)
                            mk_e.insert(0,obt_mk[i])
                            mks_obt.append(mk_e)
                            r+=1
                        
                        #checking passing eligibility
                        passing_eligibility(mks_obt,7)
                        #out_of
                        out_of=[]
                        r=1
                        for i in range(len(data)):
                            mk_e=CTkEntry(scroll_f,justify="center",height=40,fg_color="#c8f7ea",width=80,text_color="black",font=CTkFont("Helvetica",18))
                            mk_e.grid(row=r,column=4,padx=50, pady=5)
                            out_of.append(mk_e)
                            r+=1
                        for i in out_of:
                            i.insert(0,"20")
                            i.configure(state="disabled")
                
                        def update_unit_marks():
                            flag_len=True
                            flag_check=True
                            wrong_list=[]
                            for i in mks_obt:                    
                                if len(i.get())==0:
                                    flag_len=False
                                elif i.get().isalpha() or int(i.get())>20 or int(i.get())<0:
                                    flag_check=False
                                    wrong_list.append(i)         
                            if flag_check==False:
                                for i in wrong_list:
                                    i.delete(0,END)
                                flash_massa.configure(text_color="red")
                                flash_message("Invalid marks entered",flash_massa)
                            elif flag_len==False:
                                flash_massa.configure(text_color="red")
                                flash_message("Marks not entered",flash_massa)
                            elif flag_check==True and flag_len==True:
                                passing_eligibility(mks_obt,7)
                                for i in range(len(id)):
                                    stud_code=id[i]
                                    obt_mks=int(mks_obt[i].get())
                                    cursor=db.cursor()
                                    cursor.execute("update grade set obt_mks=%s where exam_type=%s and sub_code=%s and stud_code=%s",(obt_mks,e_t,sub_code,stud_code))
                                    db.commit()
                                flash_massa.configure(text_color="green")
                                flash_message("Updated Successfully",flash_massa)

                        sv_mks=CTkButton(open_f,command=update_unit_marks,hover_color="#D9D9D0",text="Update",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                        sv_mks.place(x=700,y=505)

                    else:
                        et_mks.destroy()
                        cursor=db.cursor()
                        cursor.execute("select stud_id,name from student where std_code=%s",(std))
                        data=cursor.fetchall()
                        id=[]
                        nms=[]
                        for i in data:
                            id.append(i['stud_id'])
                            nms.append(i['name'])
                        r=1
                        for i in range(len(data)):
                            s_id=CTkLabel(scroll_f,text=id[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                            s_id.grid(row=r, column=1,padx=50, pady= 5)
                            s_nm=CTkLabel(scroll_f,text=nms[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                            s_nm.grid(row=r, column=2,padx=30, pady= 5)
                            r+=1
                        #marks_obtained
                        mks_obt=[]
                        r=1
                        for i in range(len(data)):
                            mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                            mk_e.grid(row=r,column=3,padx=80, pady=5)
                            mks_obt.append(mk_e)
                            r+=1
                        #out_of
                        out_of=[]
                        r=1
                        for i in range(len(data)):
                            mk_e=CTkEntry(scroll_f,justify="center",height=40,fg_color="#c8f7ea",width=80,text_color="black",font=CTkFont("Helvetica",18))
                            mk_e.grid(row=r,column=4,padx=50, pady=5)
                            out_of.append(mk_e)
                            r+=1
                        for i in out_of:
                            i.insert(0,"20")
                            i.configure(state="disabled")
                        teacher_win.update()
                        flag_re=True
                        def save_unit_mks():
                            nonlocal flag_re
                            flag_len=True
                            flag_check=True
                            wrong_list=[]
                            for i in mks_obt:                    
                                if len(i.get())==0:
                                    flag_len=False
                                elif i.get().isalpha() or int(i.get())>20 or int(i.get())<0:
                                    flag_check=False
                                    wrong_list.append(i)         
                            if flag_check==False:
                                for i in wrong_list:
                                    i.delete(0,END)
                                flash_massa.configure(text_color="red")
                                flash_message("Invalid marks entered",flash_massa)
                            elif flag_len==False:
                                flash_massa.configure(text_color="red")
                                flash_message("Marks not entered",flash_massa)
                            elif flag_check==True and flag_len==True:
                                if flag_re:
                                    passing_eligibility(mks_obt,7)
                                    for i in range(len(id)):
                                        stud_code=id[i]
                                        obt_mks=int(mks_obt[i].get())
                                        cursor=db.cursor()
                                        cursor.execute("insert into grade(stud_code, sub_code, obt_mks, exam_type) values(%s,%s,%s,%s)",(stud_code,sub_code,obt_mks,e_t))
                                        db.commit()
                                    flash_massa.configure(text_color="green")
                                    flash_message("Saved Successfully",flash_massa)
                                    flag_re=False
                                else:
                                    flash_massa.configure(text_color="red")
                                    flash_message("Marks Already Entered",flash_massa)

                        sv_mks=CTkButton(open_f,command=save_unit_mks,hover_color="#D9D9D0",text="Save",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                        sv_mks.place(x=700,y=505)
                
                else:#term exam section
                    flash_massa=CTkLabel(open_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
                    flash_massa.place(relx=0.5,rely=0.95,anchor=CENTER)

                    if e_t=="Mid Term":
                        unit_state="Unit I"
                    else:
                        unit_state="Unit II"
                    cursor=db.cursor()
                    cursor.execute("select * from grade where stud_code=(select stud_id from student where std_code=%s limit 1) and sub_code=%s and exam_type=%s",(std,sub_code,unit_state))
                    count=cursor.fetchall()
                    if len(count)==0:
                        flash_massa.configure(text_color="red")
                        flash_message(unit_state+ " marks not entered",flash_massa)
                    else:
                        et_mks.destroy()
                        all=CTkLabel(open_f,text="",height=45,width=500,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                        all.place(x=20,y=20)
                        #back button
                        photo1=CTkImage(Image.open("back.png"),size=(40,40))
                        edit_b=CTkButton(open_f,command=lambda param=std,btn1=btn: add_grades(btn1,param),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
                        edit_b.place(x=20,y=20)
                        scroll_f=CTkScrollableFrame(open_f,border_width=2,border_color="#3941B8",width=820,corner_radius=20,fg_color="#B6E5D8",height=330)
                        scroll_f.place(x=20,y=123)
                        s_id=CTkLabel(open_f,text="id",height=45,width=40,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                        s_id.place(x=30,y=70)
                        s_nm=CTkLabel(open_f,text="Name",height=45,width=200,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                        s_nm.place(x=100,y=70)
                        in_mk=CTkLabel(open_f,text="Internal Marks/20",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",18))
                        in_mk.place(x=320,y=70)
                        ex_mk=CTkLabel(open_f,text="External Marks/80",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",18))
                        ex_mk.place(x=510,y=70)
                        ob_gr=CTkLabel(open_f,text="Grades/120",height=45,width=150,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",18))
                        ob_gr.place(x=700,y=70)
                        ex_ty=CTkLabel(open_f,text=e_t,height=45,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
                        ex_ty.place(relx=0.5,y=35,anchor=CENTER)

                        #checking for update marks for term
                        cursor=db.cursor()
                        cursor.execute("select * from grade where exam_type=%s and sub_code=%s and stud_code in (select stud_id from student where std_code=%s)",(e_t,sub_code,std))
                        count=cursor.fetchall()
                        if len(count)!=0:
                            cursor=db.cursor()
                            cursor.execute("select student.stud_id,student.name,grade.internal_mk,grade.external_mk,grade.obt_grd from student,grade where student.stud_id=grade.stud_code and grade.exam_type=%s and grade.sub_code=%s and stud_id in (select stud_id from student where std_code=%s)",(e_t,sub_code,std))
                            data=cursor.fetchall()
                            id=[]
                            nms=[]
                            inter_mr=[]
                            exter_mr=[]
                            obt_grd=[]
                        
                            for i in data:
                                id.append(i['stud_id'])
                                nms.append(i['name'])
                                inter_mr.append(i['internal_mk'])
                                exter_mr.append(i['external_mk'])
                                obt_grd.append(i['obt_grd'])
                            r=1
                            for i in range(len(data)):
                                s_id=CTkLabel(scroll_f,text=id[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                                s_id.grid(row=r,column=0,pady=10)
                                s_nm=CTkLabel(scroll_f,text=nms[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                                s_nm.grid(row=r,column=1,pady=10,padx=40)
                                r+=1

                            #internal_marks
                            internal_mk=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=3,padx=30, pady=5)
                                mk_e.insert(0,inter_mr[i])
                                internal_mk.append(mk_e)
                                r+=1

                            #external_marks
                            external_mk=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=4,padx=80, pady=5)
                                mk_e.insert(0,exter_mr[i])
                                external_mk.append(mk_e)
                                r+=1
                            #obtained_grade
                            ob_gr=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=5,padx=20, pady=5)
                                mk_e.insert(0,obt_grd[i])
                                ob_gr.append(mk_e)
                                r+=1
                            #checking passing eligibility
                            passing_eligibility(external_mk,28)
                            #calculate_grades
                            def cal_grades():
                                global unit1_mk
                                cursor=db.cursor()
                                cursor.execute('Select grade.obt_mks from grade join student on grade.stud_code = student.stud_id where grade.sub_code =%s and student.std_code = %s and grade.exam_type = %s',(sub_code,std,unit_state))
                                data=cursor.fetchall()
                                unit1_mk=[]
                                for i in range(len(data)):
                                    new=data[i]['obt_mks']
                                    unit1_mk.append(new)
                                flag_len=True
                                flag_check=True
                                wrong_list=[]
                                for i in internal_mk:                    
                                    if len(i.get())==0:
                                        flag_len=False
                                    elif i.get().isalpha() or int(i.get())>20 or int(i.get())<0:
                                        flag_check=False
                                        wrong_list.append(i) 
                                for i in external_mk:                    
                                    if len(i.get())==0:
                                        flag_len=False
                                    elif i.get().isalpha() or int(i.get())>80 or int(i.get())<0:
                                        flag_check=False
                                        wrong_list.append(i)       
                                if flag_check==False:
                                    for i in wrong_list:
                                        i.delete(0,END)
                                    flash_massa.configure(text_color="red")
                                    flash_message("Invalid marks entered",flash_massa)
                                elif flag_len==False:
                                    flash_massa.configure(text_color="red")
                                    flash_message("Marks not entered",flash_massa)
                                elif flag_check==True and flag_len==True:
                                        passing_eligibility(external_mk,28)
                                        for i in range(len(id)):
                                            obt_mks=int(internal_mk[i].get())+int(external_mk[i].get())+int(unit1_mk[i])
                                            obt_grd=get_grade(obt_mks)
                                            ob_gr[i].configure(state="normal")
                                            ob_gr[i].delete(0,END)
                                            ob_gr[i].insert(0,obt_grd)
                                            ob_gr[i].configure(state="disabled")
                                            
                            def update_term_mks():
                                for i in range(len(id)):
                                    stud_code=id[i]
                                    obt_mk1=int(internal_mk[i].get())+int(external_mk[i].get())
                                    obt_mks=int(internal_mk[i].get())+int(external_mk[i].get())+int(unit1_mk[i])
                                    inter_mk=internal_mk[i].get()
                                    exter_mk=external_mk[i].get()
                                    obt_grd=get_grade(obt_mks)
                                    cursor=db.cursor()
                                    cursor.execute("update grade set obt_mks=%s,obt_grd=%s,internal_mk=%s,external_mk=%s where stud_code=%s and sub_code=%s and exam_type=%s",(obt_mk1,obt_grd,inter_mk,exter_mk,stud_code,sub_code,e_t))
                                    db.commit()
                                flash_massa.configure(text_color="green")
                                flash_message("Updated Successfully",flash_massa)    

                            cal_mks=CTkButton(open_f,command=cal_grades,hover_color="#D9D9D0",text="Calculate Grades",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                            cal_mks.place(x=20,y=505)
                            sv_mks=CTkButton(open_f,command=update_term_mks,hover_color="#D9D9D0",text="Update",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                            sv_mks.place(x=700,y=505)

                        else:
                            cursor=db.cursor()
                            cursor.execute("select stud_id,name from student where std_code=%s",(std))
                            data=cursor.fetchall()
                            id=[]
                            nms=[]
                            for i in data:
                                id.append(i['stud_id'])
                                nms.append(i['name'])
                            r=1
                            for i in range(len(data)):
                                s_id=CTkLabel(scroll_f,text=id[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                                s_id.grid(row=r,column=0,pady=10)
                                s_nm=CTkLabel(scroll_f,text=nms[i],height=45,width=40,text_color="black",fg_color="#B6E5D8",font=CTkFont("Helvetica",20))
                                s_nm.grid(row=r,column=1,pady=10,padx=40)
                                r+=1
                            
                            #internal_marks
                            internal_mk=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=3,padx=30, pady=5)
                                internal_mk.append(mk_e)
                                r+=1

                            #external_marks
                            external_mk=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=4,padx=80, pady=5)
                                external_mk.append(mk_e)
                                r+=1

                            #grade_obtained
                            grade_ob=[]
                            r=1
                            for i in range(len(data)):
                                mk_e=CTkEntry(scroll_f,state="disabled",justify="center",fg_color="#c8f7ea",height=40,width=80,text_color="black",font=CTkFont("Helvetica",18))
                                mk_e.grid(row=r,column=5,padx=20, pady=5)
                                grade_ob.append(mk_e)
                                r+=1

                            #calculate_grades
                            def cal_grades():
                                global unit1_mk
                                cursor=db.cursor()
                                cursor.execute('Select grade.obt_mks from grade join student on grade.stud_code = student.stud_id where grade.sub_code =%s and student.std_code = %s and grade.exam_type = %s',(sub_code,std,unit_state))
                                data=cursor.fetchall()
                                unit1_mk=[]
                                for i in range(len(data)):
                                    new=data[i]['obt_mks']
                                    unit1_mk.append(new)
                                flag_len=True
                                flag_check=True
                                wrong_list=[]
                                for i in internal_mk:                    
                                    if len(i.get())==0:
                                        flag_len=False
                                    elif i.get().isalpha() or int(i.get())>20 or int(i.get())<0:
                                        flag_check=False
                                        wrong_list.append(i) 
                                for i in external_mk:                    
                                    if len(i.get())==0:
                                        flag_len=False
                                    elif i.get().isalpha() or int(i.get())>80 or int(i.get())<0:
                                        flag_check=False
                                        wrong_list.append(i)       
                                if flag_check==False:
                                    for i in wrong_list:
                                        i.delete(0,END)
                                    flash_massa.configure(text_color="red")
                                    flash_message("Invalid marks entered",flash_massa)
                                elif flag_len==False:
                                    flash_massa.configure(text_color="red")
                                    flash_message("Marks not entered",flash_massa)
                                elif flag_check==True and flag_len==True:
                                    passing_eligibility(external_mk,28)
                                    for i in range(len(id)):
                                        obt_mks=int(internal_mk[i].get())+int(external_mk[i].get())+int(unit1_mk[i])
                                        obt_grd=get_grade(obt_mks)
                                        grade_ob[i].configure(state="normal")
                                        grade_ob[i].delete(0,END)
                                        grade_ob[i].insert(0,obt_grd)
                                        grade_ob[i].configure(state="disabled")

                            flag_1=True            
                            def save_term_mks():
                                nonlocal flag_1
                                if flag_1:
                                    for i in range(len(id)):
                                        stud_code=id[i]
                                        obt_mk1=int(internal_mk[i].get())+int(external_mk[i].get())
                                        obt_mks=int(internal_mk[i].get())+int(external_mk[i].get())+int(unit1_mk[i])
                                        inter_mk=internal_mk[i].get()
                                        exter_mk=external_mk[i].get()
                                        obt_grd=get_grade(obt_mks)
                                        cursor=db.cursor()
                                        cursor.execute("insert into grade(stud_code, sub_code, obt_mks, obt_grd, exam_type, internal_mk, external_mk) values(%s,%s,%s,%s,%s,%s,%s)",(stud_code,sub_code,obt_mk1,obt_grd,e_t,inter_mk,exter_mk))
                                        db.commit()
                                    flash_massa.configure(text_color="green")
                                    flash_message("Saved Successfully",flash_massa)
                                    flag_1=False  

                                else:
                                    flash_massa.configure(text_color="red")
                                    flash_message("Marks Already Entered",flash_massa)

                                    

                            cal_mks=CTkButton(open_f,command=cal_grades,hover_color="#D9D9D0",text="Calculate Grades",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                            cal_mks.place(x=20,y=505)
                            time.sleep(2)
                            sv_mks=CTkButton(open_f,command=save_term_mks,hover_color="#D9D9D0",text="Save",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                            sv_mks.place(x=700,y=505) 

            et_mks=CTkButton(open_f,command=show_student_list,hover_color="#D9D9D0",text="Enter marks",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            et_mks.place(x=700,y=20)

        cursor=db.cursor()
        cursor.execute("SELECT class.std_id FROM teacher, class, teach_class, subject WHERE teacher.teacher_id = teach_class.teacher_code AND class.std_id = teach_class.std_code AND subject.sub_id = teach_class.sub_code AND teacher.teacher_id = %s ORDER BY CAST(class.std_id AS UNSIGNED) ASC", (teacher_id,))
        data=cursor.fetchall()
        std_i=[]

        for i in data:
                teacher_name=i.get("std_id")
                std_i.append(teacher_name)
        buttons_grade=[]
        r=45
        
        for i in std_i:
            new_b=CTkButton(f3,hover_color="#D9D9D0",text=i,height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            new_b.place(relx=0.5, y=r,anchor=CENTER)
            new_b.configure(command=lambda new=new_b,std=i: add_grades(new,std))
            animate_text(new_b,40)
            buttons_grade.append(new_b)
            r+=70
        buttons_grade[0].invoke()
        



    #------------------------------------------------------------COMPLAIN FRAME------------------------------------------------------------------------------------------------------------------------------------




    def complain_frame():
        global buttons_complain
        date_time_display()
        def show_complains(btn,depart):
            for i in buttons_complain:
                if btn==i:
                    btn.configure(fg_color="#888888")
                else:
                    i.configure(fg_color="#33CCFF")
            open_f=CTkFrame(f0,width=880,height=560,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
            open_f.place(x=325,y=100)
            animate_frame(open_f,)
            all=CTkLabel(open_f,text=depart+" - Related Complains",height=45,width=470,corner_radius=20,text_color="black",fg_color= "#ccffe6",font=CTkFont("Helvetica",20))
            all.place(x=20,y=20)
            scroll_f=CTkScrollableFrame(open_f,width=810,corner_radius=20,fg_color="#B6E5D8",height=430)
            scroll_f.place(relx=0.5,rely=0.55,anchor=CENTER)
            cursor=db.cursor()
            cursor.execute("select complain_id,subject from complain where (`to`=%s or `to`= %s) and depart=%s",("All Teachers",Name,depart))
            data=cursor.fetchall()
            def comp_desc(id):
                sol_lb=CTkLabel(open_f,text="Solution",height=45,width=170,corner_radius=20,text_color="black",fg_color="#e4d1d1",font=CTkFont("Helvetica",20))
                sol_lb.place(x=650,y=20)
                all=CTkLabel(open_f,text=" ",height=45,width=470,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                all.place(x=20,y=20)
                #back button
                photo1=CTkImage(Image.open("back.png"),size=(40,40))
                edit_b=CTkButton(open_f,command=lambda param=depart,new_b=btn: show_complains(new_b, param),image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
                edit_b.place(x=20,y=18)
                sol_f=CTkFrame(open_f,width=860,height=470,fg_color="#FFE5B4",border_width=0,corner_radius=12)
                sol_f.place(relx=0.5,rely=0.55,anchor=CENTER)
                cursor=db.cursor()
                cursor.execute("Select stud_code,description,solution,hide from complain where complain_id=%s",(id))
                result=cursor.fetchall()
                stud_id=result[0]['stud_code']
                desc=result[0]['description']
                solu=result[0]['solution']
                anon=result[0]['hide']
                id_lb=CTkLabel(sol_f,text="Complain id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                id_lb.place(x=15,y=10)
                id_et=CTkEntry(sol_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
                id_et.place(x=155,y=10)
                stid_lb=CTkLabel(sol_f,text="Student id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
                stid_lb.place(x=415,y=10)
                stid_et=CTkEntry(sol_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
                stid_et.place(x=555,y=10)
                desc_t=CTkTextbox(sol_f,font=CTkFont("Helvetica",20),width=824,height=150,fg_color="#ffff99",border_width=3,corner_radius=12,border_color="black")
                desc_t.place(x=15,y=50)
                sol_t=CTkTextbox(sol_f,font=CTkFont("Helvetica",20),width=824,height=200,fg_color="#e6f7ff",border_width=3,corner_radius=12,border_color="black")
                sol_t.place(x=15,y=220)
                #inserting values
                id_et.insert(0,id)
                stid_et.insert(0,stud_id)
                desc_t.insert('0.0',desc)
                desc_t.configure(state="disabled")
                id_et.configure(state="disabled")
                stid_et.configure(state="disabled")
                if anon==1:
                    stid_et.destroy()
                    stid_lb1=CTkLabel(sol_f,text="Anonymous",height=25,width=70,corner_radius=10,text_color="#04C34D",fg_color="#FFE5B4",font=CTkFont("Helvetica",18))
                    stid_lb1.place(x=550,y=10)
                if solu==None:
                    sol_t.insert('0.0',"Pending")
                else:
                    sol_t.insert('0.0',solu)
                def save_sol():
                    flash_massa=CTkLabel(sol_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
                    flash_massa.place(relx=0.5,rely=0.95,anchor=CENTER)
                    solution=sol_t.get("1.0", "end-1c")
                    if len(solution)==0:
                        flash_massa.configure(text_color="red")
                        flash_message("Try Again",flash_massa)
                    else:
                        solution=sol_t.get("1.0", "end-1c")
                        cursor=db.cursor()
                        cursor.execute('update complain set solution=%s where complain_id=%s',(solution,id))
                        db.commit()
                        flash_massa.configure(text_color="green")
                        flash_message("Saved Successfully",flash_massa)
                save_b=CTkButton(sol_f,text="Save",command=save_sol,height=35,width=100,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                save_b.place(relx=0.9,rely=0.95,anchor=CENTER)
            r=0
            y_pad=0
            for i in range(len(data)):
                id=data[i]['complain_id']
                id=str(id)
                sub=data[i]['subject']
                new_b=CTkButton(scroll_f,hover_color="#D9D9D0",command=lambda param=id: comp_desc(param),text=id+"       "+sub,height=45,width=790,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
                new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                r+=1


        l2=CTkLabel(f0,text="Complain Section",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        l2.place(x=40,y=30)
        new_f=CTkScrollableFrame(f0,width=240,height=535,fg_color="white",border_width=2,corner_radius=12,border_color="black")
        new_f.place(x=40,y=100)
        cursor=db.cursor()
        cursor.execute('select distinct depart from complain where `to`=%s or `to`=%s',("All Teachers",Name))
        depart=cursor.fetchall()
        departments=[]
        for i in range(len(depart)):
            new=depart[i]
            departments.append(new['depart'])
            
        r=0
        y_pad=0
        buttons_complain=[]
        for i in departments:
            new_b=CTkButton(new_f,hover_color="#D9D9D0",text=i,height=45,width=200,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
            new_b.configure(command=lambda param=i,new_b=new_b: show_complains(new_b, param))
            buttons_complain.append(new_b)
            animate_text(new_b,40)
            r+=1
        buttons_complain[0].invoke()

        



    #----------------------------------------------------------------Teachers Main Window----------------------------------------------------------------



    set_appearance_mode("light")
    set_default_color_theme("blue")
    teacher_win=CTk()
    teacher_win.title("Teacher home page")
    teacher_win.geometry(f"{teacher_win.winfo_screenwidth()}x{teacher_win.winfo_screenwidth()}")
    teacher_win.geometry("+0+0")
    # teacher_win.maxsize(width=1400,height=750)
    # teacher_win.minsize(width=1400,height=750)
    teacher_win.attributes('-fullscreen',True)
    teacher_win.iconbitmap("logo_icon.ico")

    frame=CTkFrame(teacher_win,width=1920,height=1080,fg_color="#66B3FF")
    frame.pack()
    #Home frame
    f0=CTkFrame(frame,width=1200,height=700,fg_color="#66B3FF")
    f0.place(x=140,y=30)
    #Dashboard
    f1=CTkFrame(frame,width=100,height=680,fg_color="white",border_width=3,corner_radius=15,border_color="black")
    f1.place(x=50,y=30)

    #logo
    photo=CTkImage(Image.open("logo.png"),size=(60,60))
    l1=CTkLabel(f1,image=photo,text=" ")
    l1.place(x=18,y=40)

    #home indicator
    home_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    home_indicate.place(x=7,y=150)
    #student indicator
    student_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    student_indicate.place(x=7,y=250)
    #teacher indicator
    teacher_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    teacher_indicate.place(x=7,y=350)

    #grade indicator
    grade_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    grade_indicate.place(x=7,y=450)

    #grade indicator
    complain_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    complain_indicate.place(x=7,y=550)

    #to initialize the teacher_win
    indicate(home_indicate,home_frame)

    #home button
    photo1=CTkImage(Image.open("home.png"),size=(50,50))
    b1=CTkButton(f1,command=lambda: indicate(home_indicate,home_frame),image=photo1,text="",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b1.place(x=17,y=150)

    #student button
    photo2=CTkImage(Image.open("attendence.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(student_indicate,attendance_frame),image=photo2,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=250)

    #teacher button 
    photo3=CTkImage(Image.open("time_table.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(teacher_indicate,timetable_frame),image=photo3,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=350)

    #grade button 
    photo4=CTkImage(Image.open("grade.png"),size=(50,50))
    b3=CTkButton(f1,command=lambda: indicate(grade_indicate,grade_frame),image=photo4,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b3.place(x=15,y=450)

    #complain button
    photo5=CTkImage(Image.open("report.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(complain_indicate,complain_frame),image=photo5,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=550)
    teacher_win.mainloop()






#---------------------------------------SPLASH SCREEN CODE START HERE---------------------------------------------------------------- 



def splash_loading():

    def count_to_100(progressbar):
        delay = 80
        def loop():
            progress_per = progressbar.get()
            percent = str(progress_per*100)
            load_lab_right.configure(text=f"{percent[:2]}/100")
            splash_root.update()
            if progress_per <= 100:
                
                splash_root.after(delay, loop)
        loop()
    current_text_index = 0
    def update_text():
        nonlocal current_text_index
        try:
            load_lab.configure(text=texts[current_text_index])
            animate_text(load_lab, 20)
            load_lab.update()
            current_text_index = (current_text_index + 1)
            load_lab.after(3000, update_text)
        except:
            time.sleep(1)
            splash_root.destroy()
            login_page()

    texts = [
        "Connecting database...",
        "Loading data...",
        "Preparing user interface...",
        "Finalizing setup...",
        "Almost there..."]
    
    set_appearance_mode("dark")
    set_default_color_theme("blue")
    splash_root = CTk()
    splash_root_width = 540
    splash_root_height = 250
    splash_root.title("WELCOME!")
    splash_root.overrideredirect(True)
    splash_root.geometry(f"{splash_root_width}x{splash_root_height}")
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    x = int((screen_width - splash_root_width) / 2)
    y = int((screen_height - splash_root_height) / 2)
    splash_root.geometry(f"+{x}+{y-50}")

    main_frame=CTkFrame(splash_root,bg_color="#1C1F26",fg_color="#1C1F26",corner_radius=12,width = splash_root_width,height=splash_root_height)
    main_frame.place(x=0,y=0)
    
    photo1 = CTkImage(Image.open("splash_img.jpg"),size=(splash_root_width-100,splash_root_height))
    label1 = CTkLabel(main_frame,image=photo1, text="")
    label1.place(relx=0.5,rely=0.5,anchor=CENTER)    

    
    load_lab=CTkLabel(main_frame,text="Starting Up...",height=7,font=CTkFont(family="Helvetica",weight="bold",size=12),bg_color="#1C1F26",text_color="white")
    load_lab.place(relx=0.02,rely=0.925,anchor="w")
    animate_text(load_lab, 20)

    load_lab_right=CTkLabel(main_frame,text="0/100",height=7,font=CTkFont(family="Helvetica",weight="bold",size=12),bg_color="#1C1F26",text_color="white")
    load_lab_right.place(relx=0.99,rely=0.93,anchor="e")

    progressbar_1 = CTkProgressBar(main_frame,bg_color="#1C1F26",fg_color="#1C1F26",height=4,mode="determinate",width=splash_root_width-20,determinate_speed=0.55)
    progressbar_1.place(relx=0.5,rely=0.975,anchor="center")
    progressbar_1.start()
    
    label1.after(1000, update_text)
    count_to_100(progressbar_1)

    splash_root.mainloop()


if __name__ == "__main__":
    splash_loading()