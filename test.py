import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pandas as pd

# Function to mark attendance in Excel sheet
def mark_attendance(student_id):
    with open('Attendance.csv', 'r+', encoding='utf-8') as f:
        my_data_list = f.readlines()
        id_list = []
        for line in my_data_list:
            entry = line.strip().split(',')
            id_list.append(entry[0])
        if student_id not in id_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{student_id},{dt_string}')


# Load known faces and student IDs from a folder
known_faces_encodings = []
known_faces_ids = []
known_faces_dir = 'known_faces'

for filename in os.listdir(known_faces_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        student_id = os.path.splitext(filename)[0]
        if ' (' in student_id:
            base_id, _ = student_id.split(' (')
            if base_id not in known_faces_ids:
                known_faces_ids.append(base_id)
                known_faces_encodings.append([])
            img = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            face_locations = face_recognition.face_locations(img)
            if len(face_locations) == 1:  
                face_encoding = face_recognition.face_encodings(img, face_locations)[0]
                known_faces_encodings[-1].append(face_encoding)
        else:
            if student_id not in known_faces_ids:
                known_faces_ids.append(student_id)
                img = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
                face_locations = face_recognition.face_locations(img)
                if len(face_locations) == 1: 
                    face_encoding = face_recognition.face_encodings(img, face_locations)[0]
                    known_faces_encodings.append([face_encoding])

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Find faces in frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for face_encoding, face_location in zip(face_encodings, face_locations):
        student_id = 'Unknown'

        # Check if face matches any known face
        for i, encodings in enumerate(known_faces_encodings):
            for known_face_encoding in encodings:
                match = face_recognition.compare_faces([known_face_encoding], face_encoding)
                if match[0]:
                    student_id = known_faces_ids[i]
                    break
            if student_id != 'Unknown':
                break

        # Draw rectangle around the face and display student ID
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, student_id, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Mark attendance if recognized
        if student_id != 'Unknown':
            mark_attendance(student_id)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

