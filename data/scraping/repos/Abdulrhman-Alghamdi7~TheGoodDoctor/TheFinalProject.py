import customtkinter
from customtkinter import *
import openai
import webbrowser

# خاصة  بالذكاء الاصطناعي اعدادات
openai.api_key = ""
customtkinter.set_appearance_mode("light")
root = customtkinter.CTk(fg_color="#cccccc")
root.geometry("350x550")
root.title('الطبيب الطيب')
user = dict()
# نهايه
Khalid = {
    'Full name':'خالد علي',
    'MH':'https://drive.google.com/file/d/1pxydJ01obcgw-5pnzDYNurebWdBfE95L/view?usp=sharing',
    'Age':21,
    'Gender' : 'male',
    'chronic diseases':'الضغط، السكر',
    'current medications':'لايوجد',
    'allergy':'لايوجد',
    'Genetic Disease':'لايوجد'

}

Ahmed = {
    'Full name':'احمد سعيد',
    'MH':'https://drive.google.com/file/d/1VVq9_-XvDr8AV8Y2G1Ze4IW84Izw5tEp/view?usp=sharing',
    'Age':19,
    'Gender' : 'male',
    'chronic diseases':'الضغط، القولون العصبي',
    'current medications':'اسبرين',
    'allergy':'لايوجد',
    'Genetic Disease':'لايوجد'
}

Maha = {
    'Full name':'مها سالم',
    'MH':'https://drive.google.com/file/d/1eVAKYpoyXrKqr_Z2O0Qc6PsAW3LuDjK7/view?usp=sharing',
    'Age':18,
    'Gender' : 'female',
    'chronic diseases':'الربو، السكر',
    'current medications':'لايوجد',
    'allergy':'القمح',
    'Genetic Disease':'السكر'
}
#داله الذكاء الخاصه ب الامراض
def askgpt():
    global resultt

    details = f'على ماذا يمكن ان تدل هذه الاعراض : الضغط الانقباضي : {user_data["Diastolic"]}, الضغط الانبساطي للمريض: {user_data["Systolic Pressure"]}, درجة الحرارة للمريض: {user_data["Heat"]}, و {symptoms.get()}, علما بان العمر للمريض هو {user_data["Age"]}, و مؤشر كتلة الجسم للمريض هو {user_data["BMI"]}, الامراض المزمنة لدى المريض:{user_data["chronic diseases"]},الادوية التي يستخدمها المريض حاليا : {user_data["current medications"]}, الامراض الوراثية لدى المريض: {user_data["Genetic Disease"]}, الحساسية التي يعاني منها المريض : {user_data["allergy"]}, واخيرا اذكر بعض النصائح في مثل هذه الحالة واذكر ما قد يجب فعله و اذكر مستوى الخطر مرتفع او منخفض او متوسط'

    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=details,
        temperature=0.9,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )
    resultt = response['choices'][0]['text']
    result.insert(END, resultt)
    frame4.pack_forget()
    frame5.pack(pady=20, padx=60, fill="both", expand=True)
#---
#داله الذكاء الخاصه ب العادات

def askgbt1():
    global resultt2

    details1 = f'ارجو منك مساعدتي واعطائي خطوات مفيدة وسهلة الفهم في : {h_entry.get()}'
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=details1,
        temperature=0.9,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
    )
    resultt = response['choices'][0]['text']
    result1.insert(END, resultt)
    frame4.pack_forget()
    frame6.pack_forget()
    frame7.pack(pady=20, padx=60, fill="both", expand=True)
#
#الفريم2
def save_2():
    global name, id, user
    try:
        id = int(entry2.get())
        id = str(id)
    except ValueError:
        verror = customtkinter.CTkLabel(frame2, text="الرجاء ادخال المعلومات صحيحة وكاملة",
                                        font=("inherit", 10, "bold"), text_color='red')
        verror.pack(pady=12, padx=10)

    if len(id) != 10:
        verror = customtkinter.CTkLabel(frame2, text="الرجاء ادخال الهوية الوطنية\nاو الاقامة بصورة صحيحة",
                                        font=("inherit", 10, "bold"), text_color='red')
        verror.pack(pady=12, padx=10)
    else:
        if id == '1122334455':
            user = Khalid
        elif id == '1212343455':
            user = Ahmed
        elif id == '1234512345':
            user = Maha
        else:
            verror = customtkinter.CTkLabel(frame2, text="حساب غير مسجل",
                                        font=("inherit", 10, "bold"), text_color='red')
            verror.pack(pady=12, padx=10)
            return
        frame2.pack_forget()
        frame3.pack(pady=20, padx=60, fill="both", expand=True)
        frame4.pack_forget()

#فريم3
def save_3():
    global height_value, weight_value, blood_pressure_value, user_data, sugerLevel_value
    try:

        height_value = int(height_entry.get())
        if str(height_value)[1] != "":
            height_value = float(height_value) / 100
        weight_value = int(weight_entry.get())
        blood_pressure_value = str(blood_pressure_entry.get())
        if blood_pressure_value == '0':
            diastolic = None
            systolic_pressure = None
        else:
            if '/' in blood_pressure_value:
                blood_pressure_value = (''.join(blood_pressure_value.split(' '))).split('/')
                if len(blood_pressure_value) == 2:
                    diastolic = int(blood_pressure_value[0])
                    systolic_pressure = int(blood_pressure_value[1])
                else:
                    verror = customtkinter.CTkLabel(frame3, text="الرجاء ادخال ضغط الدم بشكل صحيح",
                                                    font=("inherit", 13, "bold"), text_color='red')

                    verror.pack(pady=12, padx=10)
            else:
                verror = customtkinter.CTkLabel(frame3, text="الرجاء ادخال ضغط الدم بشكل صحيح",
                                                font=("inherit", 13, "bold"), text_color='red')
                verror.pack(pady=12, padx=10)

        bmi_value = int((float(weight_value) / float(height_value) ** 2))
        heat_value = int(heat_entry.get())
        if heat_value == 0:
            heat_value = None
        else:
            if heat_value > 42:
                heat_value = (heat_value - 32) / 1.8
        sugerLevel_value = int(sugerLevel_entry.get())
        if sugerLevel_value == 0:
            sugerLevel_value = None
    except ValueError:
        verror = customtkinter.CTkLabel(frame3, text="الرجاء كتابة ارقام فقط", font=("inherit", 13, "bold"),
                                        text_color='red')
        aerror = customtkinter.CTkLabel(frame3, text="'في حالة عدم وجود اي من\n القيم الاختيارية اكتب '0'",
                                        font=("inherit", 13, "bold"), text_color='red')
        aerror.pack(pady=12, padx=10)
        verror.pack(pady=12, padx=10)
    else:
        # Store the data in a dictionary
        user_data = {
            'ID': id,
            'Name': user['Full name'],
            "Age": user['Age'],
            'Gender' : user['Gender'],
            'MH':user['MH'],
            "Height": height_value,
            "Weight": weight_value,
            "Diastolic": diastolic,
            'Systolic Pressure': systolic_pressure,
            'Suger Level':sugerLevel_value,
            "BMI": bmi_value,
            'Heat': heat_value,
            'chronic diseases':user['chronic diseases'],
            'current medications':user['current medications'],
            'allergy':user['allergy'],
            'Genetic Disease' : user['Genetic Disease']
        }
        frame3.pack_forget()
        frame4.pack(pady=20, padx=60, fill="both", expand=True)

#فريم4 6 7
def save_4():

    frame2.pack_forget()
    frame4.pack_forget()
    frame6.pack()
def back():
    frame6.pack_forget()
    frame2.pack()
def back_1():
     frame7.pack_forget()
     frame2.pack()
def back_2():
    frame5.pack_forget()
    frame2.pack()

def back_3():
    frame3.pack_forget()
    frame2.pack()
def back_4():
    frame4.pack_forget()
    frame2.pack()

def book():
    frame5.pack_forget()
    frame8.pack()

def bback():
    frame8.pack_forget()
    frame2.pack()

def show_website():
    webbrowser.open_new(user_data['MH'])

def show_location():
    webbrowser.open_new('https://www.google.com/maps/place/%D9%85%D8%B1%D9%83%D8%B2+%D8%A7%D9%84%D8%B1%D8%B9%D8%A7%D9%8A%D8%A9+%D8%A7%D9%84%D8%B5%D8%AD%D9%8A%D8%A9+%D8%A7%D9%84%D8%A3%D9%88%D9%84%D9%8A%D8%A9+%D9%82%D8%B1%D8%B7%D8%A8%D8%A9%7C+primary+health+care+gurtubah%E2%80%AD/@24.800389,46.729333,17z/data=!4m7!3m6!1s0x3e2efd304852a6d9:0x38218c69e1530ad!4b1!8m2!3d24.800389!4d46.729333!16s%2Fg%2F11r0wqv1hd?hl=en-SA&entry=ttu')


frame2 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame3 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame4 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame5 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame6 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame7 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')
frame8 = customtkinter.CTkFrame(master=root,fg_color='#cccccc')

#فريم2 ينقل ل3
label1 = customtkinter.CTkLabel(master=frame2, text="حياك الله\nفي عيادتنا الذكية\nالف سلامة عليك \n\nتسجيل الدخول\nالي العيادة الذكية", font=("inherit", 18),height=200,width=200)
label1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame2, placeholder_text="الهوية الوطنية او الاقامة")
entry2.pack(pady=12, padx=10)


button1 = customtkinter.CTkButton(fg_color='#095d7e',master=frame2, text="تفضل", command=save_2)
button1.pack(pady=12, padx=10)

#فريم 3 ينقل ل4
heading = customtkinter.CTkLabel(frame3, text="الرجاء تعبئة البيانات", font=("inherit", 23, "bold"))
heading.pack(pady=5, padx=10)



height_entry = customtkinter.CTkEntry(master=frame3, placeholder_text="الطول",font=("inherit", 15, "bold"))
height_entry.pack(pady=12, padx=10)

weight_entry = customtkinter.CTkEntry(master=frame3, placeholder_text="الوزن",font=("inherit", 15, "bold"))
weight_entry.pack(pady=12, padx=10)

bplable = customtkinter.CTkLabel(frame3, text="الانبساطي / الانقباضي 'من اليسار الى اليمين'",
                                 font=("inherit", 11, "bold"))
bplable.pack()
#
blood_pressure_entry = customtkinter.CTkEntry(master=frame3, placeholder_text="ضغط الدم (اختياري)",font=("inherit", 15, "bold"))
blood_pressure_entry.pack(pady=12, padx=10)

heat_entry = customtkinter.CTkEntry(master=frame3, placeholder_text="درجة الحرارة (اختياري)",font=("inherit", 15, "bold"))
heat_entry.pack(pady=12, padx=10)

sugerLevel_entry = customtkinter.CTkEntry(master=frame3, placeholder_text="قياس سكر الدم (اختياري)",font=("inherit", 15, "bold"))
sugerLevel_entry.pack(pady=12, padx=10)

button2 = customtkinter.CTkButton(fg_color='#095d7e',master=frame3, text="اكمل", command=save_3)
button2.pack(pady=12)


button8 = customtkinter.CTkButton(fg_color='#095d7e',master=frame3, text="الرجوع", command=back_3)
button8.pack(pady=30)

orlable = customtkinter.CTkLabel(master=frame2, text="او",font=("inherit", 23, "bold"))
orlable.pack(pady=20, padx=12)

button3=customtkinter.CTkButton(fg_color='#095d7e',master=frame2, text="اطلب مساعدة اخرى", command=save_4,font=("inherit", 13))
button3.pack(pady=22)
label666 = customtkinter.CTkLabel(master=frame6, text="فيما اساعدك",font=("inherit", 23, "bold"))
label666.pack(pady=20, padx=12)

frame2.pack(pady=20, padx=12)


label_frame6 = customtkinter.CTkLabel(master=frame6, text="ترك عادة معينة\nنظام صحي\nخسارة الوزن\nوغيرها",font=("inherit", 18))
label_frame6.pack(pady=10, padx=10)


h_entry = customtkinter.CTkEntry(master=frame6)
h_entry.pack(pady=10, padx=10)
button5 = customtkinter.CTkButton(fg_color='#095d7e',master=frame6, text="اكمل", command=askgbt1)  # , command=save_4)
button5.pack(pady=12)

button6=customtkinter.CTkButton(fg_color='#095d7e',master=frame6, text="الرجوع", command=back)
button6.pack(pady=25)
my_frame1 = customtkinter.CTkFrame(master=frame7)

result1 = customtkinter.CTkTextbox(master=frame7, font=("inherit", 14), height=300, width=200)
result1.pack(pady=10, padx=10)
button7 = customtkinter.CTkButton(fg_color='#095d7e',master=frame7, text="الرجوع", command=back_1)
button7.pack()
heading.pack(pady=5, padx=10)

ll = customtkinter.CTkLabel(master=frame7,text='هذه النتيجة ناتجة عن ذكاء اصطناعي\nولايمكن الاعتماد عليها بصورة كبيرة',font=("inherit", 14),text_color='red')
ll.pack(pady=10,padx=10)
ll = customtkinter.CTkLabel(master=frame5,text='هذه النتيجة ناتجة عن ذكاء اصطناعي\nولايمكن الاعتماد عليها بصورة كبيرة',font=("inherit", 14),text_color='red')
ll.pack(pady=10,padx=10)


donelabel = customtkinter.CTkLabel(master=frame8,text='لقد تم حجز موعد\nنرجو منك التوجه الى :\nمركز الرعاية الصحية الأولية قرطبة',font=("inherit", 14, "bold"),height=420)
donelabel.pack(pady=10,padx=10)

lbutton = customtkinter.CTkButton(fg_color='#095d7e',master=frame8,text='موقع المركز الصحي',command=show_location)
lbutton.pack(pady=10, padx=10)

bbutton = customtkinter.CTkButton(fg_color='#095d7e',master=frame8,text = 'الصفحة الرئيسية',command=bback)
bbutton.pack(pady=10,padx=10)


heading = customtkinter.CTkLabel(master=frame4, text="اكتب جميع الاعراض \nالتي تعاني منها بدقة",
                                 font=("inherit", 23, "bold"))
heading.pack(pady=10, padx=10)

symptoms = customtkinter.CTkEntry(master=frame4)
symptoms.pack(pady=10, padx=10)


#الخاص chat gpt
button4 = customtkinter.CTkButton(fg_color='#095d7e',master=frame4, text="اكمل", command=askgpt)  # , command=save_4)
button4.pack(pady=12)
button9= customtkinter.CTkButton(fg_color='#095d7e',master=frame4, text="الرجوع", command=back_4)
button9.pack()

button10=customtkinter.CTkButton(fg_color='#095d7e',master=frame4, text="اعرض الملف الطبي",command=show_website)
button10.pack(pady=79)

my_frame = customtkinter.CTkFrame(master=frame5)




result = customtkinter.CTkTextbox(master=frame5, font=("inherit", 14), height=300, width=900)
result.pack(pady=10, padx=10)
button7 = customtkinter.CTkButton(fg_color='#095d7e',master=frame5, text="الرجوع", command=back_2)
button7.pack()

dlabel = customtkinter.CTkLabel(master=frame5,text='هل تحتاج الى حجز موعد طارئ ؟',font=("inherit", 14))
dlabel.pack(pady=11,padx=10)

dbutton = customtkinter.CTkButton(master=frame5,text='احجز موعد طارئ',command=book,fg_color='red')
dbutton.pack(pady=11,padx=10)

root.mainloop()
