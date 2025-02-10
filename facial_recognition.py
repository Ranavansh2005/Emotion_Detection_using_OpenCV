#cv2: OpenCV library face detection aur video capture ke liye use hoti hai.
#DeepFace: Emotion detection ke liye pre-trained models ko access karta hai.
import cv2
from deepface import DeepFace

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#cv2.VideoCapture(0) webcam se live video capture karta hai.
#0 ka matlab hai default webcam.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

frame_count = 0  # Counter to limit DeepFace processing

while True:
    # cap.read():#Frame-by-frame webcam se video capture karta hai.
    #ret: Video capture sahi ho raha hai ya nahi, isko check karne ke liye.
    #frame: Ek single frame ka data store karta hai.
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Process every 10th frame for efficiency
        frame_count += 1
        if frame_count % 10 == 0:
            try:
                results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Handle DeepFace output
                if isinstance(results, list) and len(results) > 0:
                    analysis = results[0]  
                elif isinstance(results, dict):
                    analysis = results
                else:
                    analysis = None

                # Get emotion
                dominant_emotion = analysis['dominant_emotion'] if analysis and 'dominant_emotion' in analysis else "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error during emotion analysis: {e}")

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('p'): 
         cv2.waitKey(0)            # Wait till until another key is pressed 
   


cap.release()
cv2.destroyAllWindows()
