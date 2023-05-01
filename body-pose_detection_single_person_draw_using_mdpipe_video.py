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

framedata1 = [] #total video1 vector data by frame
framedata2 = [] #total video2 vector data by 
 
# 두 node를 넣으면 벡터로 변환하여 리턴하는 함수
def makevector(pointA,pointB):
        vec1 = np.array(pointA)
        vec2 = np.array(pointB)

        #calculate vector between the two points (큰 노드에서 작은 노드로)
        vector = vec2-vec1
        return vector

# 두 vector data들의 cos 값 평균을 구하는 함수
def cos_sum(vectordata1, vectordata2):
    cos = [ 0 for i in range(13) ] #초기화
    for i in range(13):
        cos[i] = abs( np.dot(vectordata1[i], vectordata2[i]) / (np.linalg.norm(vectordata1[i]) * np.linalg.norm(vectordata2[i])) )
        #코사인 절댓값 구하기

    return sum(cos)/13

# Loop over each frame of the video 
# 두 영상의 각 frame별 landmark들을 추출하여 리스트에 저장 (data1,data2)
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
        if i>1 & i<11: # 1,2,3,4,5,6,7,8,9,10 점 제외
           continue
        elif i>16 & i<23: # 17,18,19,20,21,22 점 제외
            continue
        elif i>28 & i<33: # 29,30,31,32 점 제외
            continue

        marki = results2.pose_landmarks.landmark[i]
        landmark2 += [(marki.x,marki.y,marki.z)]
        
    #13개의 vector를 저장하는 lists
    vectordata1 = [] # video 1의 것
    vectordata2 = [] # video 2의 것
    
    #landmark1에 들어있는 점들로 벡터 만들어서 vectordata에 저장 (13개)
    vectordata1.append( makevector((landmark1[1] + landmark1[2])/2, landmark1[0]) )
    vectordata1.append( makevector(landmark1[5], landmark1[3]) )
    vectordata1.append( makevector(landmark1[3], landmark1[1]) )
    vectordata1.append( makevector(landmark1[2], landmark1[1]) )
    vectordata1.append( makevector(landmark1[4], landmark1[2]) )
    vectordata1.append( makevector(landmark1[6], landmark1[4]) )
    vectordata1.append( makevector(landmark1[8], landmark1[2]) )
    vectordata1.append( makevector(landmark1[7], landmark1[1]) )
    vectordata1.append( makevector(landmark1[8], landmark1[7]) )
    vectordata1.append( makevector(landmark1[10], landmark1[8]) )
    vectordata1.append( makevector(landmark1[12], landmark1[10]) )
    vectordata1.append( makevector(landmark1[11], landmark1[9]) )
    vectordata1.append( makevector(landmark1[9], landmark1[7]) )

    #landmark2에 들어있는 점들로 벡터 만들어서 vectordata에 저장 (13개)
    vectordata2.append( makevector((landmark2[1] + landmark2[2])/2, landmark2[0]) )
    vectordata2.append( makevector(landmark2[5], landmark2[3]) )
    vectordata2.append( makevector(landmark2[3], landmark2[1]) )
    vectordata2.append( makevector(landmark2[2], landmark2[1]) )
    vectordata2.append( makevector(landmark2[4], landmark2[2]) )
    vectordata2.append( makevector(landmark2[6], landmark2[4]) )
    vectordata2.append( makevector(landmark2[8], landmark2[2]) )
    vectordata2.append( makevector(landmark2[7], landmark2[1]) )
    vectordata2.append( makevector(landmark2[8], landmark2[7]) )
    vectordata2.append( makevector(landmark2[10], landmark2[8]) )
    vectordata2.append( makevector(landmark2[12], landmark2[10]) )
    vectordata2.append( makevector(landmark2[11], landmark2[9]) )
    vectordata2.append( makevector(landmark2[9], landmark2[7]) )
        
    framedata1+= [vectordata1]
    framedata2+= [vectordata2]
    
    
    
    # Draw the pose skeleton on the frame
    # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(255, 0, 0)),
    #                          mp_drawing.DrawingSpec(color=(0, 255, 0)))
    
    # mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(255, 0, 0)),
    #                           mp_drawing.DrawingSpec(color=(0, 255, 0)))
    # Resize the frame
    # resized_frame1 = cv2.resize(frame1,(width1//4,height1//4))
    # resized_frame2 = cv2.resize(frame2,(width2//4,height2//4))

    # Show the frame
    # cv2.imshow("Video1", resized_frame1)
    # cv2.imshow("Video2", resized_frame2)
    
    # Wait for the user to press 'q' to exit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


# Release the video file and close the window
sample1.release()
sample2.release()