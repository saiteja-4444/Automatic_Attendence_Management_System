from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template,redirect,url_for, request, session, flash,jsonify
import numpy as np
import mysql.connector
import cv2, os
import pandas as pd
from PIL import Image
from datetime import datetime,timedelta
from mysql.connector import pooling
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pickle
import calendar
from threading import Thread
from flask_cors import CORS 
from datetime import datetime
import json






app = Flask(__name__)
app.config["SECRET_KEY"] = "attendance system"
app.config["TRAINED_MODEL_PATH"] = os.path.join(
    os.getcwd(), "Trained_Model", "Trainner.yml"
)
app.config["LABEL_ENCODER_PATH"] = os.path.join(os.getcwd(), "label_encoder.pkl")
app.config["HARCASCADE_PATH"] = os.path.join(
    os.getcwd(), "Haarcascade", "haarcascade_frontalface_default.xml"
)
app.config["TRAINING_IMAGES_PATH"] = os.path.join(os.getcwd(), "TrainingImage")
app.config["MAIL_TEMPLATE_PATH"] = os.path.join(os.getcwd(), "attendence-mail.html")


def get_db():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        port=3306,
        database="smart_attendance",
    )
    cursor = mydb.cursor(buffered=True)
    
    return mydb , cursor
    



sender_address = "cse.takeoff@gmail.com"
sender_pass = "digkagfgyxcjltup"
detection_threshold = 40
trining_image_count = 10
trainer_thread = None
CORS(app)


def send_mail(to, subject, content):
    message = MIMEMultipart()
    message["From"] = sender_address
    message["To"] = to
    message["Subject"] = subject
    message.attach(MIMEText(content, "html"))
    session = smtplib.SMTP("smtp.gmail.com", 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, to, text)
    session.quit()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/adminhome")
def adminhome():
    return render_template("adminhome.html")


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        email = request.form["uname"]
        password = request.form["password"]
        if email == "admin" and password == "admin":
            # flash("Welcome Admin", "success")
            return redirect(url_for('dash'))
        else:
            flash("Invalid Credentials Please Try Again", "warning")
            return render_template("admin.html")
    return render_template("admin.html")


@app.route("/addback", methods=["POST", "GET"])
def addback():
    if request.method == "POST":
        Id = request.form["rno"]
        name = request.form["name"]
        email = request.form["email"]
        pno = request.form["pno"]
        cam = cv2.VideoCapture(0)
        harcascadePath = app.config["HARCASCADE_PATH"]
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        db, cursor = get_db()
        sql = f"select id from students where roll_no = {Id}"
        cursor.execute(sql)

        if cursor.fetchone() is not None:
            flash("Roll already exists", "danger")
            return render_template("updatedata.html")

        else:
            while True:
                _, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for x, y, w, h in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                    file_name = f"{name}.{Id}.{sampleNum}.jpg"
                    cv2.imwrite(
                        os.path.join(app.config["TRAINING_IMAGES_PATH"], file_name),
                        gray[y : y + h, x : x + w],
                    )
                    # display the frame
                else:
                    cv2.imshow("frame", img)
                    # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
                    # break if the sample number is morethan 100
                elif sampleNum > trining_image_count:
                    break

        cam.release()
        cv2.destroyAllWindows()
        global trainer_thread
        training_thread = Thread(target=train_data, name="train_thread")
        trainer_thread = training_thread
        training_thread.start()
        # res = "Roll Number : " + Id + " Name : " + name
        sql = "insert into students(name,email,phone,roll_no) values(%s,%s,%s,%s)"
        row = (name, email, pno, Id)
        cursor.execute(sql, row)
        db.commit()
        flash("Captured images successfully!!", "success")
        return render_template("updatedata.html")
    return render_template("updatedata.html")


def train_data():
    le = LabelEncoder()
    faces, Ids = getImagesAndLabels(app.config["TRAINING_IMAGES_PATH"])
    Ids = le.fit_transform(Ids)
    with open(app.config["LABEL_ENCODER_PATH"], "wb") as output:
        pickle.dump(le, output)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Ids))
    recognizer.save(app.config["TRAINED_MODEL_PATH"])


@app.route("/trainback")
def trainback():
    le = LabelEncoder()
    faces, Ids = getImagesAndLabels(os.path.join(os.getcwd(), "TrainingImage"))
    Ids = le.fit_transform(Ids)
    output = open(app.config["LABEL_ENCODER_PATH"], "wb")
    pickle.dump(le, output)
    output.close()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(Ids))
    recognizer.save(app.config["TRAINED_MODEL_PATH"])

    flash("Model Trained Successfully", "success")
    return render_template("adminhome.html")


@app.route("/train-start")
def train_start():
    global trainer_thread
    training_thread = Thread(target=train_data, name="train_thread")
    trainer_thread = training_thread
    training_thread.start()
    return render_template("index.html")


@app.route("/train-status")
def train_status():
    return f"{trainer_thread.name} is {'Running ' if trainer_thread.is_alive() else 'Completed'}"


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        if imagePath.endswith(".jpg") is False:
            continue
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        Id = str(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


@app.route("/view_students")
def view_students():
    db ,cursor = get_db()
    df = pd.read_sql_query("select * from students", db)
    db.close()

    return render_template(
        "view_students.html", col_name=df.columns, row_val=list(df.values.tolist())
    )


@app.route("/admin")
def admin():
    return render_template("admin.html")


def findDay(date):
    day, m1, year = (int(i) for i in date.split("-"))
    dayNumber = calendar.weekday(year, m1, day)
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return days[dayNumber]


@app.route("/viewreport", methods=["POST", "GET"])
def viewreport():
    if request.method == "POST":
        opt = request.form.get("opt")
        rno = request.form.get("rno")

        # Using parameterized queries to prevent SQL injection
        if opt == "day":
            sql = "SELECT * FROM attendance WHERE date1 = %s"
            msg = "Day Attendance Report"
        else:
            sql = "SELECT * FROM attendance WHERE m1 = %s"
            msg = "Month Attendance Report"

        cursor.execute(sql, (rno,))  # Execute the query with rno as a parameter
        data = cursor.fetchall()

        # If data exists for the given input, render viewstudentreport.html; else, view_reports.html
        if data:
            return render_template("viewstudentreport.html", data=data, msg=msg, a=rno)
        else:
            return render_template(
                "view_reports.html", msg="No data found for the given input.", a=rno
            )

    # GET request or initial page load without POST data
    return render_template("view_reports.html")


@app.route("/prediction")
def prediction():

    if os.path.exists(app.config["TRAINED_MODEL_PATH"]) is False:
        flash("Model is not trained", "danger")
        return render_template("index.html")

    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read(app.config["TRAINED_MODEL_PATH"])
    harcascadePath = app.config["HARCASCADE_PATH"]
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX
    with open(app.config["LABEL_ENCODER_PATH"], "rb") as pkl_file:
        encoder = pickle.load(pkl_file)
    db , cursor = get_db()
    data = pd.read_sql_query("select * from students", db)
    current_frame_count = 0
    max_frame_detection_count = 10
    db.close()
    is_cam = True
    while True:
        if cv2.waitKey(1) == ord("q"):
            break

        ret, im = cam.read()
        

        if ret is False:
            print("Camera not found")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for x, y, w, h in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            id, conf = recognizer.predict(gray[y : y + h, x : x + w])
            s_name = "None"

            if conf <= detection_threshold:
                # we are detecting the person
                detected_roll_no = encoder.inverse_transform([id])
                detected_roll_no = detected_roll_no[0]
                student = data[data['roll_no'] == detected_roll_no]
                s_name = student['name'][0]
                s_email = student['email'][0]
                
                current_frame_count += 1
                
                if current_frame_count >= max_frame_detection_count:
                    
                    cam.release()
                    cv2.destroyAllWindows()
                    is_cam = False
                    db , cursor = get_db()
                    current_time = (datetime.now() + timedelta(hours=0,minutes=0)).time()
                    # current_time = datetime.now().time()
                    current_date = datetime.now().strftime('%d-%m-%Y')
                    current_month = datetime.now().strftime("%B")
                    
                    # Check if the student has already punched 4 times today
                    check_sql = "SELECT COUNT(*) FROM punches WHERE roll_no = %s AND date = %s"
                    cursor.execute(check_sql, (str(detected_roll_no), current_date))
                    punch_count = cursor.fetchone()[0]

                    if punch_count >= 4:
                        print("The student has already taken 4 punches today. No more punches allowed.")
                        flash(f"Roll number {detected_roll_no} has already taken 4 punches today. No more punches allowed.")
                        return render_template('index.html')
                    else:
                        # check the current time is less than morning 10am
                        # first punch (early in) - below or equal to 10am
                        # late punch (late in) - above 10am to below 12
                        # second punch (lunch out) - in between 12 t 12.10pm
                        # third punch (lunch in) - in between 12.45pm to 1pm 
                        # final punch (day out) - above or equal to 4pm or below 6pm
                        
                        # Check the current time and set the status based on the given conditions
                        if current_time.hour < 10:  # Before 10:00 AM
                            # Morning perfect in, so insert into the database and send mail
                            status = 'Early In'
                        elif 10 <= current_time.hour < 12:  # Between 10:00 AM and 11:59 AM
                            status = 'Late In'
                        elif current_time.hour == 12 and 0 <= current_time.minute <= 10:  # Between 12:00 PM and 12:10 PM
                            status = 'Lunch Out'
                        elif (current_time.hour == 12 and current_time.minute >= 45) or (current_time.hour == 13 and current_time.minute == 0):
                            # Between 12:45 PM and 1:00 PM
                            status = "Lunch In"
                        elif 16 <= current_time.hour < 18:  # Between 4:00 PM and 5:59 PM
                            status = 'Day Out'
                        else:
                            # Default case for time outside specified ranges
                            status = 'No Punch Applicable'
                        
                        
                        sql = "insert into punches(roll_no,timing,status,date,month) values(%s,%s,%s,%s,%s)"
                        val = (str(detected_roll_no), current_time.strftime('%H:%M:%S'),status,current_date,current_month)
                        cursor.execute(sql, val)
                        db.commit()
                        
                        with open(app.config["MAIL_TEMPLATE_PATH"],'r') as mail:
                            content = mail.read()
                            content = content.replace("{{name}}",s_name)
                            content = content.replace("{{date}}",current_date)
                            content = content.replace("{{time}}",current_time.strftime('%H:%M:%S'))
                            content = content.replace("{{status}}",status)
                            
                        send_mail(s_email,'Daily Attendence Report' , content)
                        # store the attendence in the database and run mail logic
                        db.close()
            if is_cam:
                cv2.putText(
                    im, s_name, (x, y + h), font, 1, (255, 255, 255), 2
                )   
                
        if is_cam:
            cv2.imshow("im", im)

    cam.release()
    cv2.destroyAllWindows()
    flash("Attendance taken", "success")
    return render_template("index.html")


@app.route("/viewdata", methods=["POST", "GET"])
def viewdata():
    # Default logic or value for 'rno' if not a POST request
    if request.method == "POST":
        rno = request.form["rno"]

        db, cursor = get_db()
        # Check if 'rno' was provided and set
        sql = "select id from students WHERE roll_no=%s"
        cursor.execute(sql, (rno,))  # 'rno' is passed as a tuple
        data = cursor.fetchall()
        session["rno"] = rno
        db.close()
        print("session number: ",session['rno'])
        # Decide which template to render based on whether data was found
        if data:  # If data for 'rno' exists
            return render_template("studentdash.html", data=data, rno = session['rno'])
        else:  # No data found for 'rno'
            return render_template("viewdata.html", data=data)
       
    return render_template("viewdata.html")


@app.route("/studentdash", methods=["POST", "GET"])
def student_dash():
    return render_template("studentdash.html", rno = session['rno'])



@app.route("/marksback", methods=['POST', 'GET'])
def marksback():
    db, cursor = get_db()
    if request.method == 'POST':
        rno = request.form['student_id']
        mid = request.form['mid']
        sem = request.form['sem']
        year = request.form['year']
        dic = request.form

        cursor.execute(f"select name,phone,email from students where roll_no = '{rno}'")

        student = cursor.fetchone()
        
        name = student[0]
        pno = student[1]
        semail = student[2]
        m_comp = year + sem + mid

        cursor.execute(f"select id from mid_results where mcomp = '{m_comp}'")

        existing_result = cursor.fetchone()

        if existing_result is not None:
            flash("Results for this year and sem and mid already added" , "danger")
            return redirect(url_for('marksback'))


        query = f'insert into mid_results (sname,semail,rno,pno,year,sem,section,subject,marks,d1,mcomp) values '

        subjects = []
        marks = []
        current_date = datetime.now().strftime('%d-%m-%Y')
       

        for key, value in dic.to_dict().items():
            if "subject" in key: # member is subject
                subjects.append(value)
            if "marks" in key:
                marks.append(value)
            

        for i in range(len(subjects)):
            sub = subjects[i]    
            mark = marks[i]
            query += f"('{name}','{semail}','{rno}', '{pno}' , '{year}','{sem}','{mid}','{sub}','{mark}','{current_date}','{m_comp}'),"

        query = query[:-1]

        cursor.execute(query)

        db.commit()


        table_content = pd.read_sql_query(f"select subject,marks from mid_results where rno = '{rno}' and mcomp = '{m_comp}'",db)

        with open(os.path.join(os.getcwd() , 'marks-mail.html')) as mail_file:
            template = mail_file.read()
            template = template.replace("{{name}}",name)
            template = template.replace("{{marks}}",table_content.to_html(classes='marks-table',index=False))


        send_mail(semail,'Mid Marks Report' , template)

        db.close()

        flash("Results added successfully","success")

        return redirect(url_for('marksback'))
    
    cursor.execute(f'select roll_no,name from students')

    students = cursor.fetchall()

    return render_template('addmarks.html',students=students)

# view marks details form
@app.route("/view_marks", methods=["POST", "GET"])
def view_marks():
    db, cursor = get_db()
    if request.method == "POST":
        rno = request.form["rno"]
        data_query = "select * from mid_results where rno='" + rno + "'"
        cursor.execute(data_query)
        data = cursor.fetchall()
        db.close()
        if len(data) == 0:
            flash(f"data not availalble on roll number {rno}", "warning")
            return render_template("viewMarks.html")

        return render_template("viewMarks.html", data=data)
    data_query = "select * from mid_results"
    cursor.execute(data_query)
    data = cursor.fetchall()
    db.close()
    return render_template("viewMarks.html", data=data)


@app.route("/viewmarks", methods=["POST", "GET"])
def viewmarks():
    if request.method == "POST":
        year = request.form["year"]
        data_query = (
            "select * from mid_results where year='"
            + year
            + "' and rno = '"
            + session["rno"]
            + "'"
        )
        cursor.execute(data_query)
        data = cursor.fetchall()
        if len(data) == 0:
            flash(f"data not availalble in the year of {year}", "warning")
            return render_template("marks.html")

        return render_template("marks.html", data=data)
    data_query = "select * from mid_results where rno = '" + session["rno"] + "'"
    cursor.execute(data_query)
    data = cursor.fetchall()
    return render_template("marks.html", data=data)

def fetch_punches():
    """Fetch punches data from the database."""
    db,cursor = get_db()
    
    # Query to get roll_no, date, and status
    query = """
        SELECT roll_no, DATE(timing) AS punch_date
        FROM punches
    """
    cursor.execute(query)
    punches = cursor.fetchall()
    db.close()
    
    # Manually convert to dictionaries
    keys = ['roll_no', 'punch_date']
    punches = [dict(zip(keys, row)) for row in punches]
    return punches

#admin dhash board
@app.route('/admin_dashboard')
def admin_dashboard():
    # Fetch punches from the database
    db , cursor = get_db()
    query = """
        SELECT name as D, roll_no as R FROM students
    """
    cursor.execute(query)
    punches = cursor.fetchall()
    print(punches)
    db.close()
    
    return render_template("admin_dashboard.html", students=punches)

@app.route('/dash',methods = ['POST','GET'])
def dash():
    if request.method == 'POST':
        db , cursor = get_db()

        roll = request.form['roll_no']
        query = "SELECT * FROM punches where roll_no = '"+roll+"'"
        cursor.execute(query)
        punches = cursor.fetchall()
        print(punches)
        db.close()


    return render_template('admin_dashboard.html')


@app.route('/studentsdata', methods=['GET', 'POST'])
def student():
    if request.method == 'GET':
        db , cursor = get_db()
        
        query = "SELECT name, roll_no FROM students"
        cursor.execute(query,)
        print("data error")
        students = cursor.fetchall()
        
        # Check if there are students and return the response in JSON format
        if students:
            student_list = [{"name": student[0], "roll_no": student[1]} for student in students]
            return jsonify(student_list), 200
        else:
            return jsonify({"message": "No students found"}), 404
        db.close()




@app.route('/students', methods=['GET'])
def studentdata():
    if request.method == 'GET':
        try:
            
            db , cursor = get_db()
            # Get the current date in `dd-mm-yyyy` format
            current_date = datetime.now().strftime("%d-%m-%Y")

            # Query 1: Total number of students
            cursor.execute("SELECT COUNT(*) FROM students")
            total_students = cursor.fetchone()[0]

            # Query 2: Total punches for the current date
            # cursor.execute("SELECT COUNT(*) FROM punches WHERE date =%s and status = Late In or status = Early In", (current_date,))
            # total_punches = cursor.fetchone()[0]
            cursor.execute( "SELECT COUNT(*) FROM punches WHERE date = %s AND (status = 'Late In' OR status = 'Early In')", (current_date,))
            total_punches = cursor.fetchone()[0]


            # Query 3: Last punch-in student name for the current date
            cursor.execute(
                """
                SELECT s.name
                FROM punches p
                JOIN students s ON s.roll_no = p.roll_no
                WHERE p.date = %s
                ORDER BY p.timing DESC
                LIMIT 1
                """,
                (current_date,)
            )
            last_punch_student = cursor.fetchone()
            last_punch_student = last_punch_student[0] if last_punch_student else "No punches today"

            # Construct the JSON response
            response = {
                "total_students": total_students,
                "total_punches_today": total_punches,
                "last_punch_student": last_punch_student,
                "date": current_date
            }

        except Exception as e:
            # In case of an error, return an error message
            response = {
                "error": "An error occurred while retrieving data.",
                "message": str(e)
            }
        db.close()
        
        return jsonify(response)
    


@app.route('/punches/<id>', methods=['GET'])
def studentpunches(id):
    if request.method == 'GET':
    
        try:
            db , cursor = get_db()
            
            # Query to fetch punches, grouped by DATE
            query = """
                SELECT 
                    `DATE` AS punch_date, 
                    GROUP_CONCAT(timing ORDER BY timing ASC SEPARATOR ', ') AS punch_timings
                FROM 
                    punches
                WHERE 
                    roll_no = %s
                GROUP BY 
                    `DATE`
                ORDER BY 
                    `DATE` ASC;
            """
            
            # Execute the query with the student roll_no (id from URL)
            cursor.execute(query, (id,))
            punches = cursor.fetchall()

            # Prepare data in the desired structure
            punch_data = []
            for row in punches:
                # Convert punch timings into a list
                timings = row[1].split(", ")
                punch_data.append({
                    "punch_date": row[0],
                    "punch_timings": timings
                })
            
            # Close the database connection
            cursor.close()
            db.close()

            # Return the structured response as JSON
            return jsonify({
                "status": "success",
                "data": punch_data
            }), 200
        except Exception as e:
            # In case of an error, return an error message
            response = {
                "error": "An error occurred while retrieving data.",
                "message": str(e)
            }
   
  




@app.route('/present/<id>', methods=['GET'])
def present(id):
    try: 
       db , cursor = get_db()
        
       query= """
       SELECT COUNT(DISTINCT date) 
        FROM punches
        WHERE roll_no =%s
        GROUP BY date
         HAVING COUNT(*) = 4;
       """
       cursor.execute(query, (id,))
       present = cursor.fetchone()[0] or 0  # If no results, use 
      
       
       query1 = f"""
        SELECT COUNT(DISTINCT date) 
        FROM punches
        WHERE roll_no = '{id}'
        GROUP BY date
       """
       cursor.execute(query1,)
       total_days = cursor.fetchone()[0] or 0  # If no results, use 0
       
       absent = total_days - present

       cursor.execute(f"select email from students where roll_no = '{id}'")
       student = cursor.fetchone()
       email = student[0]

       percentage = (present / 100) * 100

       mail_content = ''

       if percentage < 60:
           mail_content = 'Your child having condonation of 3000'
       elif percentage > 60 and percentage <= 70:
            mail_content = 'Your child having condonation of 2000'
       elif percentage > 70 and percentage <= 75:
            mail_content = 'Your child having condonation of 1500'
        
       if len(mail_content) > 10:
        #    pass
           send_mail(email,'Condonation',mail_content)
       
       db.close()
       
       
       return jsonify({
            'roll_no': id,
            'present_days': present,
            'absent_days': absent,
            'total_days': total_days,
            "percentage" : percentage or 0,
            'msg' : mail_content
        }) 
       
    except Exception as e:
        # In case of an error, return the error in JSON format
        return jsonify({'errorsssssss': str(e)}), 500


# dowload pdf
@app.route('/download/<id>', methods=['GET'])
def allpunches(id:str):
    if request.method == 'GET':
        try:
            db , cursor = get_db()
          
            query = '''
            SELECT s.roll_no, p.timing, p.status, p.date, p.month, s.email, s.name, s.phone 
            FROM punches p 
            JOIN students s ON s.roll_no = p.roll_no 
            WHERE p.roll_no =%s;
            '''
            
            # Execute the query with the provided roll_no (id from the URL)
            cursor.execute(query, (id,))
            
            # Fetch the results
            rows = cursor.fetchall()
            
            print(rows)

            # Convert the data to a list of dictionaries
            result = []
            for row in rows:
                result.append({
                    "roll_no": row[0],
                    "timing": row[1],
                    "status": row[2],
                    "date": row[3],
                    "month": row[4],
                    "email": row[5],
                    "name": row[6],
                    "phone": row[7]
                })
            
            # Convert the result to JSON
            json_result = json.dumps(result, indent=4)
            
           

            # Return the JSON response
            db.close()

            return jsonify(result)
        
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return json_result
            


@app.route("/studentdash/<id>", methods=['GET','POST'])
def viewdashboard(id):
    if request.method == 'GET':

        # Get query parameters for date and month
        date = request.args.get('date')  # e.g., ?date=23-11-2024
        month = request.args.get('month')  # e.g., ?month=November
        # Get the database connectiovin and cursor
        db, cursor = get_db()

        # Base query
        query = "SELECT * FROM punches WHERE roll_no = %s"
        params = [id,]

        # Add filters dynamically
        if date:
            query += " AND date = %s"
            params.append(date)

        if month:
            query += " AND month = %s"
            params.append(month)

        try:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
        except Exception as e:
            db.close()  # Ensure DB connection is closed on error
            return jsonify({"error": str(e)}), 500

        # Close the database connection after fetching the data
        db.close()

        # Prepare the result data
        result = []
        for row in rows:
            result.append({
                "roll_no": row[1],
                "timing": row[2],
                "month": row[5],
                "date": row[4],
                "status": row[3],
            })

        # Return the results as a JSON response
        return jsonify(result)




# http://localhost:3000/marks/2


@app.route('/marks/<id>', methods=['GET'])
def studentmarks(id):
    if request.method == 'GET':  # Corrected from 'request == 'GET''
        db, cursor = get_db()

        query = """SELECT * FROM mid_results WHERE rno = %s"""
        cursor.execute(query, (id,))
        data = cursor.fetchall()
        
        
        db.close()

        result = []

        for datas in data:
            result.append({
                "id": datas[0],
                "sname": datas[1],
                "semail": datas[2],
                "rno": datas[3],
                "pno": datas[4],
                "year": datas[5],
                "sem": datas[6],
                "section": datas[7],
                "subject": datas[8],
                "marks": datas[9],
                "dl": datas[10],
                "mcomp": datas[11]
            })

        # Return the result as JSON
        if result:
            return jsonify(result), 200  # Return the result with a 200 OK status
        else:
            return jsonify({"error": "No marks found for the given student"}), 404  # If no data found

# http://localhost:3000/profile/2


@app.route('/Sprofile')
def Sprofile():
    return render_template('profile.html')


# student profile
@app.route('/profile/<id>',methods=['GET'])
def profile(id):
    if request.method == 'GET':
            db ,cursor = get_db()
            query = """select * from students where roll_no =%s"""
            cursor.execute(query,(id,))
            data =cursor.fetchone()
            db.close()
            
            if data:
                return jsonify({
                    'roll_no': id,
                    'name': data[1],
                    'email': data[2],
                    'phone': data[3],
                }) 

            else:
                return jsonify({"error":"No student found with ths id"})


@app.get('/mail')
def mail_temp():
    db , _ = get_db()

    table_content = pd.read_sql_query(f"select subject,marks from mid_results where rno = '2'",db)

    with open(os.path.join(os.getcwd() , 'marks-mail.html')) as mail_file:
        template = mail_file.read()
        template = template.replace("{{name}}", "Vamsi K")
        template = template.replace("{{marks}}",table_content.to_html(classes='marks-table',index=False))


    # send_mail(,'Mid Marks Report' , template)

    db.close()

    return template

if __name__ == "__main__":
    app.run(port=int(os.getenv('FLASK_PORT' , '3000')), debug=True)