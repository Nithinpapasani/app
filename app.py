import base64
import csv
import os
import sqlite3
from datetime import datetime

import bcrypt
import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
from streamlit.components.v1 import html

# Initialize global variables at module level
known_face_encodings = []
known_face_names = []
known_user_encodings = []
known_user_names = []
user_roles = []

# Database setup
db_path = "faces.db"
image_storage_dir = "images"  # Directory to store uploaded images

# Create the image storage directory if it doesn't exist
os.makedirs(image_storage_dir, exist_ok=True)

# Initialize the database with role-based access control
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
   
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            age INTEGER NOT NULL,
            dob DATE NOT NULL,
            gender TEXT CHECK(gender IN ('Male', 'Female', 'Others')),
            phone VARCHAR(10) NOT NULL CHECK (LENGTH(phone) = 10),
            address TEXT,
            marital_status TEXT CHECK(marital_status IN ('Single', 'Married', 'Divorced', 'Widowed')),
            encoding BLOB NOT NULL,
            image BLOB NOT NULL,
            image_path TEXT,
            timestamp TEXT,
            added_by TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT,
            image BLOB NOT NULL,
            encoding BLOB NOT NULL,
            timestamp TEXT,
            role TEXT DEFAULT 'user' CHECK (role IN ('user', 'admin')),
            UNIQUE(username, role)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            recognized_name TEXT NOT NULL,
            confidence_score REAL,
            recognized_by TEXT 
        )
    """)
    
    # Create an admin user if none exists
    cursor.execute("SELECT * FROM users WHERE role = 'admin'")
    if not cursor.fetchone():
        # Create a default blank image
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Create a 100x100 black image
        _, buffer = cv2.imencode('.jpg', blank_image)
        image_blob = sqlite3.Binary(buffer.tobytes())
        
        # Create a default encoding (zeros)
        default_encoding = np.zeros(128, dtype=np.float64)
        
        hashed_password = bcrypt.hashpw("admin@123".encode("utf-8"), bcrypt.gensalt())
        cursor.execute("""
            INSERT INTO users (username, password, image, encoding, role, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("admin", hashed_password.decode("utf-8"), image_blob, default_encoding.tobytes(), "admin", datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

# Load known faces from the database
def load_known_faces():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    data = cursor.fetchall()
    conn.close()

    known_encodings, known_names = [], []
    for name, encoding in data:
        if encoding is not None:  # Skip None encodings
            known_encodings.append(np.frombuffer(encoding, dtype=np.float64))
            known_names.append(name)

    return known_encodings, known_names

# Load known user faces from the database
def load_known_user_faces():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT username, encoding, role FROM users WHERE encoding IS NOT NULL")
    data = cursor.fetchall()
    conn.close()

    known_user_encodings, known_user_names, roles = [], [], []
    for username, encoding, role in data:
        if encoding is not None:  # Skip rows where encoding is NULL
            known_user_encodings.append(np.frombuffer(encoding, dtype=np.float64))
            known_user_names.append(username)
            roles.append(role)

    return known_user_encodings, known_user_names, roles

# Check if user is admin
def is_admin(username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == "admin"

# Function to log recognition events with user tracking
def log_recognition(recognized_name, confidence_score, recognized_by):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO recognition_logs (timestamp, recognized_name, confidence_score, recognized_by)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), recognized_name, confidence_score, recognized_by))
    conn.commit()
    conn.close()

# Initialize the database and load known faces
init_db()
known_face_encodings, known_face_names = load_known_faces()
known_user_encodings, known_user_names, user_roles = load_known_user_faces()

# Face Detection functions
def detect_faces_haar(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_faces_dnn(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations

# Improved Face Recognition Login Function
def face_recognition_login():
    global known_user_encodings, known_user_names, user_roles
    
    st.header("🔐 Face Recognition Login")
    st.write("Use your face to log in. Click the button below to start the webcam and capture your face.")
    
    if st.button("Start Face Recognition Login"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("Failed to capture image from webcam!")
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using both methods
            faces_haar = detect_faces_haar(frame)
            for (x, y, w, h) in faces_haar:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
            face_locations = detect_faces_dnn(rgb_frame)
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
            st.image(frame, channels="BGR", caption="Detected Face (Green: Haar, Blue: DNN)", use_container_width=True)
            
            # Face recognition with improved matching
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                st.warning("No face encodings could be generated from the detected face!")
                return
            
            match_found = False
            for face_encoding in face_encodings:
                # Compare with known faces
                face_distances = face_recognition.face_distance(known_user_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                confidence = 1.0 - face_distances[best_match_index]
                
                # Only consider it a match if confidence is above threshold
                if confidence > 0.6:  # Adjusted threshold for better accuracy
                    username = known_user_names[best_match_index]
                    role = user_roles[best_match_index]
                    log_recognition(username, confidence, "system")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    match_found = True
                    st.success(f"Welcome {username}! You are now logged in as {role}.")
                    st.balloons()
                    break
            
            if not match_found:
                st.warning("No matching face found. Please register your face.")
                name = st.text_input("Enter your name to register your face:")
                if name:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM users WHERE username = ?", (name,))
                    if cursor.fetchone():
                        st.warning(f"Username '{name}' already exists. Please log in using your face.")
                    else:
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_blob = sqlite3.Binary(buffer.tobytes())
                        encoding = face_encodings[0]
                        
                        try:
                            # password is NULL here, which is now allowed
                            cursor.execute("""
                                INSERT INTO users (username, image, encoding, timestamp, role)
                                VALUES (?, ?, ?, ?, ?)
                            """, (name, image_blob, encoding.tobytes(), datetime.now().isoformat(), "user"))
                            conn.commit()
                            st.success(f"Face registered successfully! Welcome, {name}.")
                            st.session_state.logged_in = True
                            st.session_state.username = name
                            st.session_state.role = "user"
                            # Refresh known user faces
                            known_user_encodings, known_user_names, user_roles = load_known_user_faces()
                        except sqlite3.IntegrityError:
                            st.error("Username already exists! Please choose a different name.")
                        finally:
                            conn.close()

# Login and Sign-Up Functions with bcrypt
def login(username, password):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        stored_password, role = user
        if stored_password is None: # Cannot log in with password if none is set
            return False, None
        try:
            if bcrypt.checkpw(password.encode("utf-8"), stored_password.encode("utf-8")):
                return True, role
        except Exception as e:
            print(f"Error during password verification: {e}")
    return False, None

def sign_up(username, password, role="user"):
    global known_user_encodings, known_user_names, user_roles
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False, "Username already exists. Please choose another username."
        
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        
        st.write("Please look at the camera to capture your face for registration.")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "Failed to capture image from webcam!"
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return False, "No face detected. Please try again."
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            return False, "Failed to encode face. Please try again."
            
        _, buffer = cv2.imencode('.jpg', frame)
        image_blob = sqlite3.Binary(buffer.tobytes())
        
        cursor.execute("""
            INSERT INTO users (username, password, image, encoding, role, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, hashed_password.decode("utf-8"), image_blob, face_encodings[0].tobytes(), role, datetime.now().isoformat()))
        
        conn.commit()
        # Refresh known user faces
        known_user_encodings, known_user_names, user_roles = load_known_user_faces()
        return True, "Sign up successful! Please log in."
    except Exception as e:
        print(f"Error during sign-up: {e}")
        return False, f"An error occurred during sign-up: {e}"
    finally:
        conn.close()

def user_exists(username):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# Streamlit app
def main():
    # Initialize the database and load known faces FIRST
    init_db()
    global known_face_encodings, known_face_names, known_user_encodings, known_user_names, user_roles
    known_face_encodings, known_face_names = load_known_faces()
    known_user_encodings, known_user_names, user_roles = load_known_user_faces()
    
    st.title("🤖 Face Recognition And Detection System")

    # Sidebar navigation with role-based options
    def get_sidebar_options(is_admin_user):
        base_options = [
            "🏠 Home Page", 
            "📸 Capture Face", 
            "➕ Add Face", 
            "🎥 Real-Time Face Recognition", 
            "🔄 Compare Images",
            "📷 Compare with Camera",
            "😃 Face Emotion Detection",
            "👤 Age and Gender Detection", 
            "📋 Get Details",
            "🖼 View All Faces",
            "📜 Recognition Logs"
        ]
        
        if is_admin_user:
            base_options.extend([
                "🔄 Update Face",
                "❌ Delete Record",
                "📂 Export & Import Data"
            ])
        
        return base_options

    # Check if user is logged in and get their role
    is_logged_in = st.session_state.get("logged_in", False)
    current_user = st.session_state.get("username", None)
    current_role = "admin" if is_logged_in and is_admin(current_user) else "user"

    # Display sidebar based on user role
    st.sidebar.title(" 📌 Navigation")
    options = st.sidebar.radio("Choose an option", get_sidebar_options(current_role == "admin"))

    # Home Page with Login-First Flow
    if options == "🏠 Home Page":
        st.header(" 🏠 Welcome to the Face Recognition And Detection System")
        st.write("Please log in or sign up to access the features.")
        
        login_option = st.radio("Choose an option", ["Face Login", "Traditional Login", "Sign Up"])
        
        if login_option == "Face Login":
            face_recognition_login()
        elif login_option == "Traditional Login":
            st.subheader(" 🔑 Traditional Login")
            login_username = st.text_input(" 👤 Username")
            login_password = st.text_input(" 🔒 Password", type="password")
            
            if st.button(" 🔓 Login"):
                if login_username and login_password:
                    success, role = login(login_username, login_password)
                    if success:
                        st.success(f"Logged in successfully as {role}! Now you can access the features.")
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.role = role
                        st.rerun() 
                    else:
                        st.error("Invalid username or password. Please sign up if you don't have an account.")
                else:
                    st.error("Please enter both username and password.")
        elif login_option == "Sign Up":
            st.subheader(" 📝 Sign Up")
            signup_username = st.text_input(" 👤 Choose a Username")
            signup_password = st.text_input(" 🔒 Choose a Password", type="password")
            confirm_password = st.text_input(" 🔄 Confirm Password", type="password")
            
            if st.button(" 🆕 Sign Up"):
                if not signup_username or not signup_password:
                    st.error("Username and password are required.")
                elif signup_password != confirm_password:
                    st.error("Passwords do not match. Please try again.")
                elif user_exists(signup_username):
                    st.error("Username already exists. Please choose another username.")
                else:
                    success, message = sign_up(signup_username, signup_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    # Check if user is logged in before allowing access to features
    if not st.session_state.get("logged_in", False):
        if options != "🏠 Home Page":
            st.warning("Please log in or sign up to access the features.")
            st.stop()
    else:
        st.success(f"Welcome, {st.session_state.username}! (Role: {st.session_state.role})")
        
        # Main app features
        if options == "➕ Add Face":
            st.markdown('<h1 class="add-face-header">➕ Add a New Face</h1>', unsafe_allow_html=True)
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            dob = st.text_input("Date of Birth (YYYY-MM-DD)")
            gender = st.selectbox("Gender", ["Male", "Female", "Others"])
            phone = st.text_input("Phone Number")
            address = st.text_input("Address")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            
            if st.button("Add Face"):
                errors = []
                if not name:
                    errors.append("❗ Name is required and Must be Unique.")
                if not age or age < 0 or age > 120:
                    errors.append("❗ Age must be greater than 0.")
                if not uploaded_file:
                    errors.append("❗ An image upload is required.")
                if not dob:
                    errors.append("❗ Date of Birth is required and Must be in this format (YYYY-MM-DD).")
                if not phone:
                    errors.append("❗ Phone Number is required.")
                # Check phone number format only if it's not empty
                elif not phone.isdigit() or len(phone) != 10:
                    errors.append("❗ Phone Number must be exactly 10 digits.")
                if not address:
                    errors.append("❗ Address is required.")

                # If there are any errors, display them and stop
                if errors:
                    st.error("\n".join(errors))
                else:
                    # --- If validation passes, proceed with database logic ---
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_img)
                    
                    if not face_encodings:
                        st.error("No face detected in the uploaded image!")
                    else:
                        encoding = face_encodings[0]
                        image_blob = sqlite3.Binary(file_bytes)
                        
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                INSERT INTO faces (name, age, dob, gender, phone, address, marital_status, 
                                                encoding, image, timestamp, added_by)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (name, age, dob, gender, phone, address, marital_status, 
                                encoding.tobytes(), image_blob, datetime.now().isoformat(), st.session_state.username))
                            conn.commit()
                            st.success("✅ Face added successfully!")
                            # Refresh known faces
                            known_face_encodings, known_face_names = load_known_faces()
                        except sqlite3.IntegrityError as e:
                            # This will now primarily catch the UNIQUE constraint violation for the name
                            if "UNIQUE constraint failed: faces.name" in str(e):
                                st.error("Error: A record with this name already exists.")
                            else:
                                st.error(f"Database Integrity Error: {e}")
                        finally:
                            conn.close()

        elif options == "📸 Capture Face":
            st.markdown('<h1 class="capture-face-header">📸 Capture Face from Webcam</h1>', unsafe_allow_html=True)
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            dob = st.text_input("Date of Birth (YYYY-MM-DD)")
            gender = st.selectbox("Gender", ["Male", "Female", "Others"])
            phone = st.text_input("Phone Number")
            address = st.text_input("Address")
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

            if st.button("Capture Face"):
                # --- Pre-Submission Validation ---
                errors = []
                if not name:
                    errors.append("❗ Name is required and Must be Unique.")
                if not age or age < 0 or age > 120:
                    errors.append("❗ Age must be greater than 0.")
                if not dob:
                    errors.append("❗ Date of Birth is required and Must be in this format (YYYY-MM-DD).")
                if not phone:
                    errors.append("❗ Phone Number is required.")
                # Check phone number format only if it's not empty
                elif not phone.isdigit() or len(phone) != 10:
                    errors.append("❗ Phone Number must be exactly 10 digits.")
                if not address:
                    errors.append("❗ Address is required.")

                # If there are any errors, display them and stop
                if errors:
                    st.error("\n".join(errors))
                else:
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        st.error("Failed to capture image from webcam!")
                    else:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_frame)
                        if not face_encodings:
                            st.error("No face detected!")
                        else:
                            encoding = face_encodings[0]
                            _, buffer = cv2.imencode('.jpg', frame)
                            image_blob = sqlite3.Binary(buffer.tobytes())

                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    INSERT INTO faces (name, age, dob, gender, phone, address, marital_status,
                                                    encoding, image, timestamp, added_by)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (name, age, dob, gender, phone, address, marital_status,
                                        encoding.tobytes(), image_blob, datetime.now().isoformat(), st.session_state.username))
                                conn.commit()
                                st.success("Face captured and stored successfully!")
                                # Refresh known faces
                                known_face_encodings, known_face_names = load_known_faces()
                            except sqlite3.IntegrityError:
                                st.error("Name already exists!")
                            finally:
                                conn.close()
                                
        elif options == "📋 Get Details":
            st.markdown('<h1 class="get-details-header">📋 Retrieve Details by Name</h1>', unsafe_allow_html=True)
            name = st.text_input("Enter Name")
            
            if st.button("Get Details"):
                if not name:
                    st.error("Please Enter the Name of Person!")
                else:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM faces WHERE name = ?", (name,))
                    data = cursor.fetchone()
                    conn.close()
                    
                    if not data:
                        st.error("Name not found in database!")
                    else:
                        (id, name, age, dob, gender, phone, address, marital_status, 
                         _, image_blob, _, timestamp, added_by) = data
                        image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
                        st.image(image, channels="BGR", caption=f"Image of {name}", use_container_width=True)
                        
                        st.write(f"**ID:** {id}")
                        st.write(f"**Name:** {name}")
                        st.write(f"**Age:** {age}")
                        st.write(f"**Date of Birth:** {dob}")
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Phone Number:** {phone}")
                        st.write(f"**Address:** {address}")
                        st.write(f"**Marital Status:** {marital_status}")
                        st.write(f"**Added By:** {added_by}")
                        st.write(f"**Timestamp:** {timestamp}")
                        
        elif options == "🔄 Update Face":
            if current_role != "admin":
                st.error("⛔ Admin access required for this feature!")
            else:
                st.markdown('<h1 class="update-face-header">🔄 Update Face Details</h1>', unsafe_allow_html=True)
                st.markdown('<div class="admin-feature">', unsafe_allow_html=True)
            
                name = st.text_input("Enter the Name of the Person to Update", key="update_face_name_input")

                if name:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM faces WHERE name = ?", (name,))
                    record = cursor.fetchone()
                    conn.close()

                    if record:
                        st.write("### Current Details:")
                        st.write(f"ID: {record[0]}")
                        st.write(f"Name: {record[1]}")
                        st.write(f"Age: {record[2]}")
                        st.write(f"Date of Birth: {record[3]}")
                        st.write(f"Gender: {record[4]}")
                        st.write(f"Phone Number: {record[5]}")
                        st.write(f"Address: {record[6]}")
                        st.write(f"Marital Status: {record[7]}")
                        st.write(f"Added By: {record[12]}")

                        # Display current image
                        st.write("### Current Image:")
                        image = cv2.imdecode(np.frombuffer(record[9], np.uint8), cv2.IMREAD_COLOR)
                        st.image(image, channels="BGR", caption="Current Face Image", use_container_width=True)  

                        # Update form
                        st.subheader("Enter New Details to Update")
                        new_name = st.text_input("New Name", value=record[1], key="new_name_input")
                        new_age = st.number_input("New Age", min_value=0, max_value=120, value=record[2], key="new_age_input")
                        new_dob = st.text_input("New Date of Birth (YYYY-MM-DD)", value=record[3], key="new_dob_input")
                        
                        gender_options = ["Male", "Female", "Others"]
                        try:
                            gender_index = gender_options.index(record[4]) if record[4] in gender_options else 0
                        except (ValueError, TypeError):
                            gender_index = 0
                        new_gender = st.selectbox("New Gender", gender_options, index=gender_index, key="new_gender_selectbox")
                                          
                        new_phone = st.text_input("New Phone Number", value=record[5], key="new_phone_input")
                        new_address = st.text_input("New Address", value=record[6], key="new_address_input")

                        marital_options = ["Single", "Married", "Divorced", "Widowed"]
                        try:
                            marital_index = marital_options.index(record[7]) if record[7] in marital_options else 0
                        except (ValueError, TypeError):
                            marital_index = 0
                        new_marital_status = st.selectbox("New Marital Status", marital_options, index=marital_index, key="new_marital_status_selectbox")

                        # Image update options
                        st.subheader("Update Face Image")
                        update_image_option = st.radio("Choose an option to update the face image:", 
                                                ["Keep Current Image", "Upload New Image", "Take New Photo"], 
                                                key="update_image_option_radio")

                        new_image_blob = None
                        new_encoding = None

                        if update_image_option == "Upload New Image":
                            new_uploaded_file = st.file_uploader("Upload a new image", type=["jpg", "jpeg", "png"], 
                                                            key="upload_image_file_uploader")
                            if new_uploaded_file is not None:
                                try:
                                    file_bytes = np.asarray(bytearray(new_uploaded_file.read()), dtype=np.uint8)
                                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    face_encodings = face_recognition.face_encodings(rgb_img)
                                    if not face_encodings:
                                        st.error("No face detected in the new image!")
                                    else:
                                        new_encoding = face_encodings[0]
                                        new_image_blob = sqlite3.Binary(file_bytes)
                                        st.image(img, channels="BGR", caption="New Image to Upload", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error processing image: {str(e)}")

                        elif update_image_option == "Take New Photo":
                            if st.button("Capture New Photo"):
                                cap = cv2.VideoCapture(0)
                                ret, frame = cap.read()
                                cap.release()
                                if ret:
                                    st.image(frame, channels="BGR", caption="New Captured Image", use_container_width=True)
                                    rgb_captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    face_encodings = face_recognition.face_encodings(rgb_captured_image)
                                    if not face_encodings:
                                        st.error("No face detected in the captured image!")
                                    else:
                                        new_encoding = face_encodings[0]
                                        _, buffer = cv2.imencode('.jpg', frame)
                                        new_image_blob = sqlite3.Binary(buffer.tobytes())
                                        st.success("New photo captured and processed.")
                                else:
                                    st.error("Failed to capture new photo.")

                        # Update button
                        if st.button("Update Face", key=f"update_face_button_{name}"):
                            try:
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                            
                                if update_image_option != "Keep Current Image" and new_image_blob is not None and new_encoding is not None:
                                    # Update with new image and encoding
                                    cursor.execute("""
                                        UPDATE faces
                                        SET name = ?, age = ?, dob = ?, gender = ?, phone = ?, address = ?, 
                                            marital_status = ?, encoding = ?, image = ?, timestamp = ?
                                        WHERE name = ?
                                    """, (new_name, new_age, new_dob, new_gender, new_phone, new_address, 
                                        new_marital_status, new_encoding.tobytes(), new_image_blob, 
                                        datetime.now().isoformat(), name))
                                else:
                                    # Update only details
                                    cursor.execute("""
                                        UPDATE faces
                                        SET name = ?, age = ?, dob = ?, gender = ?, phone = ?, address = ?, 
                                            marital_status = ?, timestamp = ?
                                        WHERE name = ?
                                    """, (new_name, new_age, new_dob, new_gender, new_phone, new_address, 
                                        new_marital_status, datetime.now().isoformat(), name))
                                
                                conn.commit()
                                st.success("Face details updated successfully!")
                                # Refresh known faces
                                known_face_encodings, known_face_names = load_known_faces()
                            except Exception as e:
                                st.error(f"Error updating record: {str(e)}")
                            finally:
                                conn.close()
                    else:
                        st.error("No record found for the given name!")
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
        elif options == "📂 Export & Import Data":
            if current_role != "admin":
                st.error("⛔ Admin access required for this feature!")
            else:
                st.markdown('<h1 class="export-import-header">📂 Export & Import Data</h1>', unsafe_allow_html=True)
                st.markdown('<div class="admin-feature">', unsafe_allow_html=True)
                
                # Export Data Section
                st.subheader("Export Data")
                if st.button("Export Face Data to CSV"):
                    conn = sqlite3.connect(db_path)
                    try:
                        df = pd.read_sql_query("SELECT name, age, dob, gender, phone, address, marital_status, added_by, timestamp FROM faces", conn)
                        if df.empty:
                            st.warning("No face data found to export.")
                        else:
                            csv_data = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name="face_data.csv",
                                mime="text/csv",
                            )
                            st.success("Face data exported successfully!")
                    except Exception as e:
                        st.error(f"An error occurred during export: {e}")
                    finally:
                        conn.close()
                        
                # Import Data Section
                st.subheader("Import Data")
                uploaded_file = st.file_uploader("Upload a CSV file to import face data", type=["csv"])
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.write("### Preview of Uploaded Data")
                        st.write(df.head())
                        
                        required_columns = ["Name", "Age", "Date of Birth", "Gender", "Phone", "Address", "Marital Status"]
                        if not all(col in df.columns for col in required_columns):
                            st.error(f"CSV file must contain these columns: {', '.join(required_columns)}")
                        else:
                            conn = sqlite3.connect(db_path)
                            cursor = conn.cursor()
                            
                            success_count = 0
                            error_count = 0
                            
                            for _, row in df.iterrows():
                                try:
                                    name = row["Name"]
                                    # For import, we cannot create face encodings, so these will be blank.
                                    # Users would need to update the face image manually after import.
                                    blank_encoding = np.zeros(128).tobytes()
                                    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
                                    _, buffer = cv2.imencode('.jpg', blank_image)
                                    image_blob = buffer.tobytes()

                                    cursor.execute("""
                                        INSERT INTO faces (name, age, dob, gender, phone, address, marital_status, 
                                                        encoding, image, timestamp, added_by)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (name, row["Age"], row["Date of Birth"], row["Gender"], row["Phone"], 
                                        row["Address"], row["Marital Status"], blank_encoding, image_blob, 
                                        datetime.now().isoformat(), st.session_state.username))
                                    
                                    success_count += 1
                                except sqlite3.IntegrityError:
                                    st.warning(f"Skipping duplicate name: {name}")
                                    error_count += 1
                                except Exception as e:
                                    st.error(f"Error processing row for {name}: {str(e)}")
                                    error_count += 1
                            
                            conn.commit()
                            conn.close()
                            st.success(f"Import completed! Success: {success_count}, Errors/Duplicates: {error_count}")
                            # Refresh known faces
                            known_face_encodings, known_face_names = load_known_faces()
                            
                    except Exception as e:
                        st.error(f"Error processing CSV file: {str(e)}")
                        
                st.markdown('</div>', unsafe_allow_html=True)
                
        elif options == "🔄 Compare Images":
            st.markdown('<h1 class="compare-images-header">🔄 Compare Two Images</h1>', unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"], key="img1")
            uploaded_file2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"], key="img2")
            
            if st.button("Compare Images"):
                if not uploaded_file1 or not uploaded_file2:
                    st.error("Please upload both images!")
                else:
                    # Process first image
                    file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
                    img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)
                    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    face_encodings1 = face_recognition.face_encodings(rgb_img1)
                    
                    if not face_encodings1:
                        st.error("No face detected in the first image!")
                    else:
                        encoding1 = face_encodings1[0]
                        
                        # Process second image
                        file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
                        img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
                        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        face_encodings2 = face_recognition.face_encodings(rgb_img2)
                        
                        if not face_encodings2:
                            st.error("No face detected in the second image!")
                        else:
                            encoding2 = face_encodings2[0]
                            
                            # Compare faces
                            results = face_recognition.compare_faces([encoding1], encoding2)
                            match = results[0]
                            face_distance = face_recognition.face_distance([encoding1], encoding2)
                            similarity = (1 - face_distance[0]) * 100
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(img1, channels="BGR", caption="First Image", use_container_width=True)
                            with col2:
                                st.image(img2, channels="BGR", caption="Second Image", use_container_width=True)
                            
                            if match:
                                st.success(f"✅ Match Found! Similarity: {similarity:.2f}%")
                                log_recognition("Image Comparison", similarity / 100, st.session_state.username)
                            else:
                                st.error(f"❌ No Match Found. Similarity: {similarity:.2f}%")
                                
        elif options == "📷 Compare with Camera":
            st.markdown('<h1 class="compare-camera-header">📷 Compare Uploaded Image with Camera</h1>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload an image to compare against", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                uploaded_encodings = face_recognition.face_encodings(rgb_img)

                if not uploaded_encodings:
                    st.error("No face detected in the uploaded image!")
                else:
                    uploaded_encoding = uploaded_encodings[0]
                    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

                    st.write("### Camera Comparison")
                    if "run_comparison" not in st.session_state:
                        st.session_state.run_comparison = False
                    if "match_found" not in st.session_state:
                        st.session_state.match_found = False

                    start_button = st.button("Start Camera for Comparison")
                    stop_button = st.button("Stop Comparison", disabled=not st.session_state.run_comparison)

                    if start_button:
                        st.session_state.run_comparison = True
                        st.session_state.match_found = False
                    if stop_button:
                        st.session_state.run_comparison = False

                    FRAME_WINDOW = st.image([])

                    if st.session_state.run_comparison and not st.session_state.match_found:
                        cap = cv2.VideoCapture(0)
                        while st.session_state.run_comparison and not st.session_state.match_found:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to capture video from webcam!")
                                break

                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_frame)
                            cam_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                            for (top, right, bottom, left), cam_encoding in zip(face_locations, cam_encodings):
                                # Use numpy for fast distance calculation
                                face_distance = np.linalg.norm(uploaded_encoding - cam_encoding)
                                confidence = (1 - face_distance) * 100
                                match = face_distance > 0.6  # threshold for match

                                name = "Match Found" if match else "Unknown"
                                color = (0, 255, 0) if match else (0, 0, 255)

                                if match:
                                    log_recognition("Camera Match", confidence / 100, st.session_state.username)
                                    st.session_state.match_found = True
                                    label = f"{name} ({confidence:.2f}%)"
                                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    FRAME_WINDOW.image(frame, channels="BGR")
                                    st.success(f"✅ Match Found! Similarity: {confidence:.2f}%")
                                    st.info("You can stop the camera now.")
                                    break
                                else:
                                    label = f"{name} ({confidence:.2f}%)"
                                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            FRAME_WINDOW.image(frame, channels="BGR")
                            # Allow Streamlit to process UI events (like Stop button)
                            if not st.session_state.run_comparison or st.session_state.match_found:
                                break
                        cap.release()
                        cv2.destroyAllWindows()
                    if st.session_state.match_found:
                        st.info("Match found. Click 'Stop Comparison' to reset and try again.")

                    
        elif options == "🎥 Real-Time Face Recognition":
            st.markdown('<h1 class="real-time-recognition-header">🎥 Real-Time Face Tracking</h1>', unsafe_allow_html=True)

            if not known_face_encodings:
                st.warning("No known faces found in the database. Please add faces first.")
            else:
                if "run_real_time_recognition" not in st.session_state:
                    st.session_state.run_real_time_recognition = False

                start_button = st.button("Start Face Tracking")
                stop_button = st.button("Stop Face Tracking")

                if start_button:
                    st.session_state.run_real_time_recognition = True
                if stop_button:
                    st.session_state.run_real_time_recognition = False

                FRAME_WINDOW = st.image([])

                if st.session_state.run_real_time_recognition:
                    cap = cv2.VideoCapture(0)
                    while st.session_state.run_real_time_recognition:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to capture video")
                            break

                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                confidence = 1.0 - face_distances[best_match_index]

                                if confidence > 0.5:
                                    name = known_face_names[best_match_index]
                                    log_recognition(name, confidence, st.session_state.username)
                                else:
                                    name = "Unknown"

                                label = f"{name} ({confidence * 100:.2f}%)"
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        FRAME_WINDOW.image(frame, channels="BGR")
                        # Allow Streamlit to process UI events (like Stop button)
                        if not st.session_state.run_real_time_recognition:
                            break
                    cap.release()
                    cv2.destroyAllWindows()
                
        elif options == "🖼 View All Faces":
            st.markdown('<h1 class="view-all-faces-header">🖼 View All Stored Faces</h1>', unsafe_allow_html=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, image, added_by, timestamp FROM faces")
            faces = cursor.fetchall()
            conn.close()
            
            if not faces:
                st.info("No faces found in the database.")
            else:
                st.write(f"### Gallery of Stored Faces (Total: {len(faces)})")
                
                # Search and filter options
                search_name = st.text_input("Search by name")
                all_users = list(set([face[2] for face in faces if face[2]]))
                added_by_filter = st.selectbox("Filter by added by", ["All"] + all_users)
                
                # Apply filters
                filtered_faces = faces
                if search_name:
                    filtered_faces = [f for f in filtered_faces if search_name.lower() in f[0].lower()]
                if added_by_filter != "All":
                    filtered_faces = [f for f in filtered_faces if f[2] == added_by_filter]
                    
                if not filtered_faces:
                    st.warning("No faces match your filters.")
                else:
                    cols = st.columns(3)
                    for idx, (name, image_blob, added_by, timestamp) in enumerate(filtered_faces):
                        with cols[idx % 3]:
                            try:
                                image = cv2.imdecode(np.frombuffer(image_blob, np.uint8), cv2.IMREAD_COLOR)
                                st.image(image, channels="BGR", caption=name, use_container_width=True)
                                st.caption(f"Added by: {added_by or 'N/A'}")
                                st.caption(f"On: {timestamp or 'N/A'}")
                            except Exception as e:
                                st.error(f"Could not display image for {name}")
                            
        elif options == "📜 Recognition Logs":
            st.markdown('<h1 class="recognition-logs-header">📜 Face Recognition Logs</h1>', unsafe_allow_html=True)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                name_filter = st.text_input("Filter by recognized name")
            with col2:
                user_filter = st.text_input("Filter by recognized by")
            
            conn = sqlite3.connect(db_path)
            query = "SELECT timestamp, recognized_name, confidence_score, recognized_by FROM recognition_logs ORDER BY timestamp DESC"
            try:
                df = pd.read_sql_query(query, conn)
                
                if name_filter:
                    df = df[df['recognized_name'].str.contains(name_filter, case=False, na=False)]
                if user_filter:
                    df = df[df['recognized_by'].str.contains(user_filter, case=False, na=False)]
                
                if not df.empty:
                    st.write(f"### Recognition History (Showing {len(df)} records)")
                    st.dataframe(df)
                else:
                    st.info("No recognition logs found matching your filters.")
                    
            except Exception as e:
                st.error(f"Could not fetch logs: {e}")
            finally:
                conn.close()
                
        elif options == "❌ Delete Record":
            if current_role != "admin":
                st.error("⛔ Admin access required for this feature!")
            else:
                st.markdown('<h1 class="delete-record-header">❌ Delete a Record</h1>', unsafe_allow_html=True)
                st.markdown('<div class="admin-feature">', unsafe_allow_html=True)
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM faces")
                all_names = [row[0] for row in cursor.fetchall()]
                conn.close()

                if not all_names:
                    st.warning("No records to delete.")
                else:
                    name_to_delete = st.selectbox("Select the person to delete", all_names)
                    if st.button(f"Delete Record for {name_to_delete}"):
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM faces WHERE name = ?", (name_to_delete,))
                        conn.commit()
                        conn.close()
                        st.success(f"Record for {name_to_delete} deleted successfully!")
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
                
        elif options == "😃 Face Emotion Detection":
            st.markdown('<h1 class="emotion-detection-header">😃 Face Emotion Detection</h1>', unsafe_allow_html=True)
            st.write("Detect emotions from facial expressions.")
            emotion_option = st.radio("Choose an option:", ["Upload Image", "Use Webcam"], key="emotion_option")

            if emotion_option == "Upload Image":
                uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="emotion_upload")
                if uploaded_file:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)
                    try:
                        results = DeepFace.analyze(img, actions=["emotion"])
                        for i, result in enumerate(results):
                            st.write(f"### Face {i+1} Emotion Analysis")
                            dominant_emotion = result["dominant_emotion"]
                            st.write(f"**Dominant Emotion:** {dominant_emotion}")
                            emotions_df = pd.DataFrame.from_dict(result["emotion"], orient="index", columns=["Score"])
                            st.bar_chart(emotions_df)
                    except ValueError:
                        st.error("No face detected in the uploaded image!")
                    except Exception as e:
                        st.error(f"An error occurred during emotion analysis: {e}")

            elif emotion_option == "Use Webcam":
                if "run_emotion_detection" not in st.session_state:
                    st.session_state.run_emotion_detection = False

                start_button = st.button("Start Webcam for Emotion Detection")
                stop_button = st.button("Stop Emotion Detection")

                if start_button:
                    st.session_state.run_emotion_detection = True
                if stop_button:
                    st.session_state.run_emotion_detection = False

                FRAME_WINDOW = st.image([])

                if st.session_state.run_emotion_detection:
                    cap = cv2.VideoCapture(0)
                    while st.session_state.run_emotion_detection:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture video from webcam!")
                            break
                        try:
                            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                            for result in results:
                                box = result['region']
                                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                                emotion = result['dominant_emotion']
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        except Exception as e:
                            pass # Fail silently if no face is detected in a frame
                        FRAME_WINDOW.image(frame, channels="BGR")
                        if not st.session_state.run_emotion_detection:
                            break
                    cap.release()
                    cv2.destroyAllWindows()


        elif options == "👤 Age and Gender Detection":
            st.markdown('<h1 class="age-gender-header">👤 Age and Gender Detection</h1>', unsafe_allow_html=True)
            detection_mode = st.radio("Select detection mode:", ["Live Detection", "Detect from Upload"])

            if detection_mode == "Live Detection":
                if "run_detection" not in st.session_state:
                    st.session_state.run_detection = False

                start_button = st.button("Start Webcam for Age/Gender Detection")
                stop_button = st.button("Stop Detection")

                if start_button:
                    st.session_state.run_detection = True
                if stop_button:
                    st.session_state.run_detection = False

                FRAME_WINDOW = st.image([])

                if st.session_state.run_detection:
                    cap = cv2.VideoCapture(0)
                    while st.session_state.run_detection:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture video from webcam!")
                            break
                        try:
                            results = DeepFace.analyze(frame, actions=["age", "gender"], enforce_detection=False)
                            for result in results:
                                box = result['region']
                                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                                age = result['age']
                                gender = result['dominant_gender']
                                label = f"{gender}, {age}"
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        except Exception as e:
                            pass
                        FRAME_WINDOW.image(frame, channels="BGR")
                        # Allow Streamlit to process UI events (like Stop button)
                        if not st.session_state.run_detection:
                            break
                    cap.release()
                    cv2.destroyAllWindows()
            
            elif detection_mode == "Detect from Upload":
                uploaded_file = st.file_uploader("Upload an image for age/gender analysis", type=["jpg", "jpeg", "png"])
                if uploaded_file:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    st.image(img, channels="BGR", caption="Uploaded Image")
                    try:
                        results = DeepFace.analyze(img, actions=["age", "gender"])
                        for i, result in enumerate(results):
                            st.write(f"### Analysis for Face {i+1}")
                            age = result['age']
                            gender = result['dominant_gender']
                            st.metric("Estimated Age", f"{age} years")
                            st.metric("Predicted Gender", gender)
                            gender_df = pd.DataFrame.from_dict(result["gender"], orient="index", columns=["Confidence"])
                            st.bar_chart(gender_df)
                    except ValueError:
                        st.error("No face detected in the image.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

    # Logout button in sidebar
    if st.session_state.get("logged_in", False):
        if st.sidebar.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()