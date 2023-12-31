import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep

def get_encoded_faces():
  """
  looks through the faces folder and encodes all
  the faces
  I
  : return: dict of (name, image encoded)
  """
  encoded = {}
  for dirpath, dnames, fnames in os.walk("./faces"):
    for f in fnames:
      if f.endswith(" jpg") or f.endswith(" png"):
        face = fr.load_image_file("faces/" + f)
        encoding = fr.face_encodings(face)[0]
        encoded[f.split(".")[0]] = encoding
  return encoded

def unknown_image_encoded (img) :
  """
    encode a face given the file name
  """
  face = fr.load_image_file("faces/" + img)
  encoding = fr.face_encodings(face)[0]
  return encoding

def classify_face (im):
  """
  will find all of the faces in a given image and label
  them if it knows what they are
  : param im: str of file path
  : return: list of face names
  """
  print("classify_face", im)
  faces = get_encoded_faces ()
  faces_encoded = list(faces.values())
  known_face_names = list(faces.keys())
  img = cv2.imread(im, 1)
  #img = cv2. resize(img, (0, 0), fx=0.5, fy=0.5)
  #img = img[:,:,::-1]

  face_locations = face_recognition.face_locations(img)
  if not face_locations:
      print("No faces found in the image:", im)
      return []

  print("Detected", len(face_locations), "face(s) in the image:", im)

  unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
  face_names = []

  if not unknown_face_encodings:
    print("No faces recognized in the image:", im)
    return []

  print("Recognized", len(unknown_face_encodings), "face(s) in the image:", im)


  for face_encoding in unknown_face_encodings:
    matches = face_recognition.compare_faces(faces_encoded, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(faces_encoded, face_encoding)

    if len(face_distances) > 0:
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]

      face_names.append(name)  # Indentation corrected here
      print("face_encoded", faces_encoded)
      print("face_encoding", face_encoding)
      print("face_distances", face_distances)

      for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
        # Draw a label with a name below the face
        cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left-20, bottom+15), font, 1.0, (255, 255, 255), 2)

# Display the resulting image
  while True:
    cv2.imshow( 'Video', img)
    if cv2.waitKey (1) & 0xFF == ord('q'):
      return face_names

#print(os.walk("./testBjork.jpg").
print(classify_face("testKeanu.jpg"))
