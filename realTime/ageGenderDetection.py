import cv2

# Load pre-trained models for face detection, gender, and age prediction
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('realTime/models/gender_deploy.prototxt', 'realTime/models/gender_net.caffemodel')
age_model = cv2.dnn.readNetFromCaffe('realTime/models/age_deploy.prototxt', 'realTime/models/age_net.caffemodel')

# Access the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Failed to capture frame. Exiting...")
        break  # Exit the loop if frame not captured properly

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop for all the faces detected
    for faceBox in faces:
        # Extracting face as per the faceBox
        face = frame[max(0, faceBox[1] - 15):min(faceBox[1] + faceBox[3] + 15, frame.shape[0] - 1),
                     max(0, faceBox[0] - 15):min(faceBox[0] + faceBox[2] + 15, frame.shape[1] - 1)]

        # Extracting the main blob part
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Prediction of gender
        gender_model.setInput(blob)
        genderPreds = gender_model.forward()
        gender = "Male" if genderPreds[0].argmax() == 1 else "Female"

        # Prediction of age
        age_model.setInput(blob)
        agePreds = age_model.forward()
        age = int(agePreds[0].argmax())

        # Putting text of age and gender at the top of box
        cv2.putText(frame,
                    f'{gender}, Age: {age}',
                    (faceBox[0] - 50, faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA)
            # Display the frame
    cv2.imshow('Face Detection', frame)
    
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the frame with
