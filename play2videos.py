import cv2

video1_path = './runs/detect/R_2_noewmodel.mp4'
video2_path = './demo01.mp4'

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

height = max(height1, height2)

width = width1 + width2

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break
    print("KK")
    frame1 = cv2.resize(frame1, (width1, height))
    frame2 = cv2.resize(frame2, (width2, height))

    combined_frame = cv2.hconcat([frame1, frame2])

    cv2.imshow('Two Videos Side by Side', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
