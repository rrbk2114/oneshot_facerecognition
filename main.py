import face_recognition
import cv2
import numpy as np
import os

# Path to the folder containing the images
database_path = "database"  # Path to your images folder

# Initialize arrays for storing face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Dynamically load all images and their encodings from the database folder
for filename in os.listdir(database_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Extract the person's name from the file name (e.g., "John_Doe.jpg" -> "John Doe")
        name = os.path.splitext(filename)[0].replace("_", " ")
        # Load the image and calculate face encodings
        image_path = os.path.join(database_path, filename)
        print(f"Loading image: {image_path}")  # Debugging: Check if the image is being accessed
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
        else:
            print(f"No face found in {filename}, skipping.")  # Skip images with no face

# Initialize some variables for the video capture
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Open the webcam for real-time face recognition
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare the faces in the current frame to the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Draw boxes and labels around detected faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame was scaled down to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with the person's name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")  # Inform the user when quitting
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
