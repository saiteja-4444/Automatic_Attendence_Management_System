import csv
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, session, flash
import numpy as np
import mysql.connector
import cv2, os
import pandas as pd
from PIL import Image
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import pickle
import calendar
import time
from datetime import datetime
from threading import Thread


app = Flask(__name__)
app.config['SECRET_KEY'] = 'attendance system'
mydb = mysql.connector.connect(host="localhost", user="root", passwd="", port= 3306, database="smart_attendance")
cursor = mydb.cursor()
sender_address = 'appcloud887@gmail.com'
sender_pass = 'uihywuzqiutvfofo'
detection_threshold = 40
trining_image_count = 3000
trainer_thread = None

def send_mail(to,subject,content):
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = to
    message['Subject'] = subject
    message.attach(MIMEText(content, 'plain'))
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, to, text)
    session.quit()
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/adminhome')
def adminhome():
    return render_template('adminhome.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['uname']
        password = request.form['password']
        if email == 'admin' and password == 'admin':
            flash("Welcome Admin", "success")
            return render_template('adminhome.html')
        else:
            flash("Invalid Credentials Please Try Again", "warning")
            return render_template('admin.html')
    return render_template('admin.html')


@app.route("/addback", methods=['POST', 'GET'])
def addback():
    if request.method == 'POST':
        Id = request.form['rno']
        name = request.form['name']
        email = request.form['email']
        pno = request.form['pno']
        cam = cv2.VideoCapture(0)
        harcascadePath = "Haarcascade/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        sql = f"select id from students where roll_no = {Id}"
        cursor.execute(sql)
        
        if cursor.fetchone() is not None:
            flash("Roll already exists", "danger")
            return render_template("updatedata.html")

        else:
            while (True):
                _, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage/ " + name + "." + Id + '.' + str(
                        sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # display the frame
                else:
                    cv2.imshow('frame', img)
                    # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                    # break if the sample number is morethan 100
                elif sampleNum > trining_image_count:
                    break
        
        cam.release()
        cv2.destroyAllWindows()
        global trainer_thread
        training_thread = Thread(target=train_data,name='train_thread')
        trainer_thread = training_thread
        training_thread.start()
        # res = "Roll Number : " + Id + " Name : " + name
        sql = "insert into students(name,email,phone,roll_no) values(%s,%s,%s,%s)"
        row = (name,email, pno,Id)
        cursor.execute(sql, row)
        mydb.commit()
        flash("Captured images successfully!!", "success")
        return render_template("updatedata.html")
    return render_template("updatedata.html")

def train_data():
    le = LabelEncoder()
    faces, Ids = getImagesAndLabels("TrainingImage")
    Ids = le.fit_transform(Ids)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Ids))
    recognizer.save(r"Trained_Model\Trainner.yml")

@app.route('/trainback')
def trainback():
    le = LabelEncoder()
    faces, Ids = getImagesAndLabels("TrainingImage")
    Ids = le.fit_transform(Ids)
    output = open('label_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Ids))
    recognizer.save(r"Trained_Model\Trainner.yml")

    flash("Model Trained Successfully", "success")
    return render_template('adminhome.html')

@app.route('/train-start')
def train_start():
    global trainer_thread
    training_thread = Thread(target=train_data,name='train_thread')
    trainer_thread = training_thread
    training_thread.start()
    return render_template('index.html')

@app.route('/train-status')
def train_status():
    return f"{trainer_thread.name} is {'Running ' if trainer_thread.is_alive() else 'Completed'}"

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        if imagePath==r"TrainingImage\Thumbs.db":
            continue
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = str(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

@app.route('/view_students')
def view_students():
    df = pd.read_sql_query('select * from students',mydb)
    

    return render_template('view_students.html', col_name=df.columns, row_val=list(df.values.tolist()))

@app.route('/admin')
def admin():
    return render_template('admin.html')


def findDay(date):
    day, m1, year = (int(i) for i in date.split('-'))
    dayNumber = calendar.weekday(year, m1, day)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    return (days[dayNumber])

@app.route('/viewreport', methods=['POST', 'GET'])
def viewreport():
    if request.method == "POST":
        opt = request.form.get('opt')
        rno = request.form.get('rno')

        # Using parameterized queries to prevent SQL injection
        if opt == "day":
            sql = "SELECT * FROM attendance WHERE date1 = %s"
            msg = 'Day Attendance Report'
        else:
            sql = "SELECT * FROM attendance WHERE m1 = %s"
            msg = 'Month Attendance Report'
        
        cursor.execute(sql, (rno,))  # Execute the query with rno as a parameter
        data = cursor.fetchall()

        # If data exists for the given input, render viewstudentreport.html; else, view_reports.html
        if data:
            return render_template('viewstudentreport.html', data=data, msg=msg, a=rno)
        else:
            return render_template('view_reports.html', msg="No data found for the given input.", a=rno)

    # GET request or initial page load without POST data
    return render_template('view_reports.html')

@app.route('/prediction')
def prediction():
    print(trainer_thread.name)
    if trainer_thread.is_alive():
        flash("Training is not done yet please wait", "warning")
        return render_template('index.html')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(r"Trained_Model\Trainner.yml")
    harcascadePath = r"Haarcascade\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pkl_file = open('label_encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()
    data = pd.read_sql_query("select * from students",mydb)
    all_roll_nos = str(data.roll_no.values)
    # all_emails = str(data.email.values)
    # all_names = str(data.name.values)
    
    while True:
        if cv2.waitKey(1) == ord('q'):
            break
        
        _, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            s_name = 'None'
            
            
            if conf > detection_threshold:
                # we are detecting the person
                print(id,'detected')
                pass
            
            cv2.putText(im, f'{str(conf)} - {s_name}', (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('im', im)
        
    cam.release()
    cv2.destroyAllWindows()
    flash("Attendance taken", "success")
    return render_template('index.html')


@app.route('/viewdata', methods=['POST', 'GET'])
def viewdata():
    # Default logic or value for 'rno' if not a POST request
    rno = None

    if request.method == 'POST':
        rno = request.form['rno']
        
    if not rno:
        # If 'rno' is not set, either render a default page or prompt for input
        return render_template('viewdata.html') 
    
    # Check if 'rno' was provided and set
    sql = "select id from students WHERE roll_no=%s"
    cursor.execute(sql, (rno,))  # 'rno' is passed as a tuple
    data = cursor.fetchall()
    session['rno'] = rno
    # Decide which template to render based on whether data was found
    if data:  # If data for 'rno' exists
        return render_template('viewalldata.html', data=data)
    else:  # No data found for 'rno'
        return render_template('viewdata.html', data=data)
    

@app.route("/marksback", methods=['POST', 'GET'])
def marksback():
    if request.method == 'POST':
        rno = request.form['rno']
        name = request.form['name']
        semail = request.form['email']
        pno = request.form['pno']
        mid = request.form['mid']
        # sub=request.form['sub']
        # mrk = request.form['mrk']
        # marks1 = request.form['marks']
        sem = request.form['sem']
        year = request.form['year']
        dic = request.form
        data = {}
        m={}
        for key, value in dic.to_dict().items():
            if "member" in key:
                data[key] = value
            if "merks" in key:
                m[key]=value
        subjects = [(k, data[k]) for k in data]
        marks = [(k, m[k]) for k in m]
        x = [i[1] for i in subjects]
        y = [i[1] for i in marks]
        date1 = datetime.now().strftime('%d-%m-%Y')
        mcomp = year + sem + mid
        df = pd.read_csv("Student_Details/StudentDetails.csv")
        val = df.Roll_Number.values
        if rno not in str(val):
            flash("Roll not available", "danger")
            return render_template("addmarks.html")
        else:
            ss = "select count(*) from mid_results where rno='" + rno + "' and mcomp='" + mcomp + "'"
            z = pd.read_sql_query(ss, mydb)
            count = z.values[0][0]
            sub = []
            if count == 0:
                for i in range(len(x)):
                    sub_lists = str(x[i])
                    mrk_lists = str(y[i])
                    total_data=sub_lists+':'+mrk_lists
                    sq = "insert into mid_results(sname,semail,rno,pno,year,sem,section,subject,marks,d1,mcomp)values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    val = (name,semail, rno, pno, year, sem, mid, sub_lists, mrk_lists, date1, mcomp)
                    cursor.execute(sq, val)
                    mydb.commit()
                    mail_content ='Miss/Mstr '+name +' and Rollno '+ rno +' Your Midresults for subject :'+ sub_lists + ' and the marks are :'+ mrk_lists + '' 
                    sender_address = 'appcloud887@gmail.com'
                    sender_pass = 'uihywuzqiutvfofo'
                    receiver_address = "appcloud887@gmail.com"
                    message = MIMEMultipart()
                    message['From'] = sender_address
                    message['To'] = receiver_address
                    message['Subject'] = 'Design and implementation of smart Attendance system using advanced deep learning methods'
                    message.attach(MIMEText(mail_content, 'plain'))
                    session = smtplib.SMTP('smtp.gmail.com', 587)
                    session.starttls()
                    session.login(sender_address, sender_pass)
                    text = message.as_string()
                    session.sendmail(sender_address, receiver_address, text)
                    session.quit()

                flash("Data Added SuccessFully", "success")
                return render_template('addmarks.html')
    return render_template('addmarks.html')
#view marks details form
@app.route('/view_marks', methods = ['POST','GET'])
def view_marks():
    if request.method == 'POST':
        rno = request.form['rno']
        data_query = "select * from mid_results where rno='"+rno+"'"
        cursor.execute(data_query)
        data = cursor.fetchall()
        if len(data) == 0:
            flash(f"data not availalble on roll number {rno}","warning")
            return render_template('viewMarks.html')

        return render_template('viewMarks.html', data = data)
    data_query = "select * from mid_results"
    cursor.execute(data_query)
    data = cursor.fetchall()
    return render_template('viewMarks.html', data = data)

@app.route('/viewmarks', methods = ['POST','GET'])
def viewmarks():
    if request.method == 'POST':
        year = request.form['year']
        data_query = "select * from mid_results where year='"+year+"' and rno = '"+session['rno']+"'"
        cursor.execute(data_query)
        data = cursor.fetchall()
        if len(data) == 0:
            flash(f"data not availalble in the year of {year}","warning")
            return render_template('marks.html')

        return render_template('marks.html', data = data)
    data_query = "select * from mid_results where rno = '"+session['rno']+"'"
    cursor.execute(data_query)
    data = cursor.fetchall()
    return render_template('marks.html', data = data)
if __name__ == '__main__':
    app.run(debug=True)
