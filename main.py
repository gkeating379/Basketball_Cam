from ultralytics import YOLO
import cv2
from perspective_estimation import homography, motion
from tracking import tracking, jersey

import time


model = YOLO('models/New Model Medium/100/best.pt') #MED runs about 200 ms
#model = YOLO('models\\New Model\\best(1)_20.pt') #max runs about 1000ms

# results = model.predict('Input_Videos\knicks.mp4', save=True)


cap = cv2.VideoCapture('Input_Videos\\olympics.mp4')
court = cv2.imread('perspective_estimation\\court.jpg')

track = tracking.Track()

frame_num = 0

old_frame = None
frame = None
features = None
old_features = None
H = None
court_corners = None
detections = []

while cap.isOpened():
    frame_num += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if there was not a frame read or manual kill
    if not ret or cv2.waitKey(1) == ord('q'):
        break

    #player location
    cur_pred = model.predict(frame, save=False, classes=[1,2]) #ignore ball handlers
    tracked = track.get_object_tracks(frame, cur_pred[0])
    
    detections += [cur_pred[0].boxes.xyxy]
    detections = detections[-5:]

    start_time = time.time()
    #camera motion
    H, court_corners, old_frame, old_features = motion.update_homography(frame, court, old_frame, old_features, court_corners, detections)
    if court_corners[0] is not None:
        homography.draw_four(frame, court_corners[0], court_corners[1], court_corners[2], court_corners[3], (0, 0, 255))

    #track players over frames
    if H is not None:
        track.proj_track(tracked, H, court, frame)

    
    cv2.imshow('frame', frame)
    #cv2.imwrite('frame.png',frame)
    print(f'Frame Runtime: {(time.time() - start_time)} seconds')






cap.release()
cv2.destroyAllWindows()