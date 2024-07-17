import cv2
import depthai as dai
from deepface import DeepFace
from emotion_recognition_using_speech.data_extractor import load_data

def detect_emotion():
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # print('Connected cameras:', device.getConnectedCameras())
        # print('Usb speed:', device.getUsbSpeed().name)
        print("Camera is Activated")
        
        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            inRgb = qRgb.get()  # Blocking call, will wait until a new data has arrived
            
            # Retrieve 'bgr' (opencv format) frame
            frame = inRgb.getCvFrame()

            # Convert frame to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                try:
                    # Analyze emotions using DeepFace
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']

                    # Draw rectangle around face and label with predicted emotion
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    return emotion  # Return the detected emotion

                except Exception as e:
                    print(f"Error analyzing face: {str(e)}")

            # Display the resulting frame
            #cv2.imshow('Real-time Emotion Detection', frame)

            # Press 'q' to exit
            # if cv2.waitKey(1) == ord('q'):
                # break

    cv2.destroyAllWindows()
