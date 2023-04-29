import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video file
sample1 = cv2.VideoCapture("sample1.mp4")
sample2 = cv2.VideoCapture("sample4.mp4")

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
    
    for i in range(33):
        if i>0 & i<11: # 1,2,3,4,5,6,7,8,9,10 점 제외
           continue
        if i>16 & i<23: # 17,18,19,20,21,22 점 제외
            continue
        if i>28 & i<33: # 29,30,31,32 점 제외
            continue
        marki = results1.pose_landmarks.landmark[i]
        landmark1 += [(marki.x,marki.y,marki.z)]
    
    for i in range(33):
        if i>0 & i<11: # 1,2,3,4,5,6,7,8,9,10 점 제외
           continue
        if i>16 & i<23: # 17,18,19,20,21,22 점 제외
            continue
        if i>28 & i<33: # 29,30,31,32 점 제외
            continue
        marki = results2.pose_landmarks.landmark[i]
        landmark2 += [(marki.x,marki.y,marki.z)]
        
    data1+= [landmark1]
    data2+= [landmark2]
    
    # 두 node를 넣으면 벡터로 변환하여 리턴한는 함수
    def makevector(landmarkbig,landmarksmall):
            point1 = np.array(landmarkbig) # 숫자가 더 큰 node
            point2 = np.array(landmarksmall)

            #calculate vector between the two points (큰 노드에서 작은 노드로)
            vector = point2-point1

            return vector
    
    #13개의 vector를 저장하는 lists
    vectordata1 = [] # video 1의 것
    vectordata2 = [] # video 2의 것

    #landmark1에 들어있는 점들로 벡터 만들어서 vectordata에 저장 (13개)
    vectordata1[0] = makevector(landmark1[12] + landmark1[11]/2, landmark1[0])
    vectordata1[1] = makevector(landmark1[12], landmark1[11])
    vectordata1[2] = makevector(landmark1[13], landmark1[11])
    vectordata1[3] = makevector(landmark1[14], landmark1[12])
    vectordata1[4] = makevector(landmark1[15], landmark1[13])
    vectordata1[5] = makevector(landmark1[16], landmark1[14])
    vectordata1[6] = makevector(landmark1[23], landmark1[11])
    vectordata1[7] = makevector(landmark1[24], landmark1[12])
    vectordata1[8] = makevector(landmark1[24], landmark1[23])
    vectordata1[9] = makevector(landmark1[25], landmark1[23])
    vectordata1[10] = makevector(landmark1[26], landmark1[24])
    vectordata1[11] = makevector(landmark1[27], landmark1[25])
    vectordata1[12] = makevector(landmark1[28], landmark1[26])

    #landmark2에 들어있는 점들로 벡터 만들어서 vectordata에 저장 (13개)
    vectordata2[0] = makevector(landmark1[12] + landmark1[11]/2, landmark1[0])
    vectordata2[1] = makevector(landmark1[12], landmark1[11])
    vectordata2[2] = makevector(landmark1[13], landmark1[11])
    vectordata2[3] = makevector(landmark1[14], landmark1[12])
    vectordata2[4] = makevector(landmark1[15], landmark1[13])
    vectordata2[5] = makevector(landmark1[16], landmark1[14])
    vectordata2[6] = makevector(landmark1[23], landmark1[11])
    vectordata2[7] = makevector(landmark1[24], landmark1[12])
    vectordata2[8] = makevector(landmark1[24], landmark1[23])
    vectordata2[9] = makevector(landmark1[25], landmark1[23])
    vectordata2[10] = makevector(landmark1[26], landmark1[24])
    vectordata2[11] = makevector(landmark1[27], landmark1[25])
    vectordata2[12] = makevector(landmark1[28], landmark1[26])              

    # 두 벡터의 코사인 차이의 평균을 구하는 함수
    def cos_sum(vector1, vector2):
        cos = [0] *13 #초기화
        for i in range(13):
            cos[i] = np.dot(vector1[i], vector2[i]) / (np.linalg.norm(vector1[i]) * np.linalg.norm(vector2[i]))
            #코사인 값 구하기
    
        return sum(cos)/13
    
    cos_sim = cos_sum(vectordata1, vectordata2)

    
    # Draw the pose skeleton on the frame
    mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0)),
                             mp_drawing.DrawingSpec(color=(0, 255, 0)))
    
    mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0)),
                              mp_drawing.DrawingSpec(color=(0, 255, 0)))
    # Resize the frame
    resized_frame1 = cv2.resize(frame1,(width1//4,height1//4))
    resized_frame2 = cv2.resize(frame2,(width2//4,height2//4))

    # Show the frame
    cv2.imshow("Video1", resized_frame1)
    cv2.imshow("Video2", resized_frame2)
    
    # Wait for the user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
sample1.release()
sample2.release()

print(len(data1))