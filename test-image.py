import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
import pandas as pd

# Function to mark attendance in CSV file
def mark_attendance(student_id, attendance_file):
    now = datetime.now()
    today_date = now.strftime('%Y-%m-%d')

    # Check if the attendance file exists
    if not os.path.isfile(attendance_file):
        # Create a new attendance file with headers
        with open(attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ['Date'] + known_faces_ids
            writer.writerow(headers)

    # Update the attendance record for the student ID on the current date
    with open(attendance_file, 'r+', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        found = False
        for row in rows:
            if row['Date'] == today_date:
                row[student_id] = 'Present'
                found = True
                break
        if not found:
            new_row = {'Date': today_date}
            for header in known_faces_ids:
                if header == student_id:
                    new_row[header] = 'Present'
                else:
                    new_row[header] = 'Absent'
            rows.append(new_row)

        # Write the updated rows to the attendance file
        fieldnames = ['Date'] + known_faces_ids
        with open(attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

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

# Load the single test image
test_image_path = 'test_image.jpeg'  # Change this to the path of your single test image
test_image = face_recognition.load_image_file(test_image_path)

# Find faces in the test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Initialize the attendance CSV file
attendance_file = 'Attendance-from-image.csv'

# Loop through each face found in the test image
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

    # Mark attendance if recognized
    if student_id != 'Unknown':
        mark_attendance(student_id, attendance_file)

# Display the resulting image
for (top, right, bottom, left), student_id in zip(face_locations, known_faces_ids):
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(test_image, student_id, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow('Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
