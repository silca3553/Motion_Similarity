import cv2
import mediapipe as mp

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file
sample1 = cv2.VideoCapture("sample4.mp4")
sample2 = cv2.VideoCapture("sample5.mp4")

# Get the video dimensions
width1 = int(sample1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(sample1.get(cv2.CAP_PROP_FRAME_HEIGHT))

width2 = int(sample2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(sample2.get(cv2.CAP_PROP_FRAME_HEIGHT))

data1 = []
data2 = []

# Loop over each frame of the video
while True:
    # Read a frame from the video
    ret1, frame1 = sample1.read()
    ret2, frame2 = sample2.read()
    
    if not ret1 or not ret2:
        break
    
    # Convert the frame from BGR to RGB
    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Run the pose model on the frame
    results1 = pose.process(image1)
    results2 = pose.process(image2)
    
    landmark1 = []
    landmark2 = []
    
    #for i in range(33):
    #    marki = results1.pose_landmarks.landmark[i]
    #    landmark1 += [(marki.x,marki.y,marki.z)]
    
    #for i in range(33):
    #    marki = results2.pose_landmarks.landmark[i]
        #landmark2 += [(marki.x,marki.y,marki.z)]
        
    data1+= [landmark1]
    data2+= [landmark2]
    

    # Draw the pose skeleton on the frame
    mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0)),
                             mp_drawing.DrawingSpec(color=(0, 255, 0)))
    
    mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0)),
                              mp_drawing.DrawingSpec(color=(0, 255, 0)))
    # Resize the frame
    resized_frame1 = cv2.resize(frame1,(width1//4,height1//4))
    resized_frame2 = cv2.resize(frame2,(width2//2,height2//2))

    # Show the frame
    cv2.imshow("Video1", resized_frame1)
    cv2.imshow("Video2", resized_frame2)
    
    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
sample1.release()
sample2.release()

#print(data1[0][1])