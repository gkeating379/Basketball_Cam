from ultralytics import YOLO
import cv2
from perspective_estimation import homography, motion


# model = YOLO('models/New Model/best(1)_20.pt')

# results = model.predict('Input_Videos\knicks.mp4', save=True)


cap = cv2.VideoCapture('Input_Videos\\knicks.mp4')
court = cv2.imread('perspective_estimation\court.jpg')
court_canny = cv2.imread('perspective_estimation\court_boundary.png')

frame_num = 0

old_frame = None
frame = None
features = None
old_features = None
H = None
court_corners = None

while cap.isOpened():
    frame_num += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if there was not a frame read
    if not ret:
        break

    #manual kill
    if cv2.waitKey(1) == ord('q'):
        break

    
    #camera motion
    H, court_corners, old_frame, old_features = motion.update_homography(frame, court, court_canny, old_frame, old_features, court_corners)
    homography.draw_four(frame, court_corners[0], court_corners[1], court_corners[2], court_corners[3], (0, 0, 255))
    
    cv2.imshow('frame', frame)
    
    cv2.imwrite('frame.png',frame)





cap.release()
cv2.destroyAllWindows()