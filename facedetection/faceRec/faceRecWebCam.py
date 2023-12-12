import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


def get_encoded_faces():
    """
    Looks through the faces folder and encodes all the faces
    :return: dict of (name, image encoded)
    """
    encoded = {}
    # Assume the faces are stored in the "./faces" directory
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith("jpg") or f.endswith("png"):
                face = fr.load_image_file("faces/" + f)
                print("** loading face ", face)
                encoding = fr.face_encodings(face)[0]
                print("** encoding face ", encoding)
                encoded[f.split(".")[0]] = encoding
    return encoded

def classify_face(frame, faces_encoded, known_face_names):
    """
    Will find all the faces in a given image frame and label them if it knows what they are
    :param frame: input image frame from the webcam
    :param faces_encoded: list of known face encodings
    :param known_face_names: list of known face names
    :return: list of face names
    """
    face_locations = fr.face_locations(frame)
    if not face_locations:
        print("No faces found in the frame")
        return []

    print("Detected", len(face_locations), "face(s) in the frame")
    print("frame ", frame, " ... face_locations ", face_locations)
    unknown_face_encodings = fr.face_encodings(frame, face_locations)
    face_names = []

    if not unknown_face_encodings:
        print("No faces recognized in the frame")
        return []

    print("Recognized", len(unknown_face_encodings), "face(s) in the frame")

    for face_encoding in unknown_face_encodings:
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        face_distances = fr.face_distance(faces_encoded, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left - 20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    return face_names

# Get known faces encodings and names
faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

# Start capturing frames from the webcam
video_capture = cv2.VideoCapture(1)  # Use 0 for the default webcam
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame = frame[:, :, ::-1]

    face_names = classify_face(frame, faces_encoded, known_face_names)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
