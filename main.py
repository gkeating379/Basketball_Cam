from ultralytics import YOLO
import cv2
from perspective_estimation import homography, motion
from tracking import tracking

# input paths

input_video_path = 'Input_Videos\\knicks.mp4'
output_court_img = 'perspective_estimation\\court.jpg'
model_path = 'models/New Model Medium/100/best.pt'

# read model/video
cap = cv2.VideoCapture(input_video_path)
court = cv2.imread(output_court_img)
model = YOLO(model_path)
# MED runs about 200 ms
# model = YOLO('models\\New Model\\best(1)_20.pt') #max runs about 1000ms

# output paths
output_file = 'output_video.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = court.shape[:2][::-1]

# Create VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# inits
track = tracking.Track()
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0
old_frame = None
frame = None
features = None
old_features = None
H = None
court_corners = None
detections = []

while cap.isOpened():
    print(f'\rFrame {frame_num}/{total_frames}', end='')
    frame_num += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if there was not a frame read or manual kill
    if not ret or cv2.waitKey(1) == ord('q'):
        break

    # player location
    cur_pred = model.predict(frame, save=False, classes=[1, 2], verbose=False)  # ignore ball handlers
    tracked = track.get_object_tracks(frame, cur_pred[0])

    detections += [cur_pred[0].boxes.xyxy]
    detections = detections[-5:]

    # camera motion
    H, court_corners, old_frame, old_features = motion.update_homography(frame, court, old_frame, old_features, court_corners, detections)
    if court_corners[0] is not None:
        homography.draw_four(frame, court_corners[0], court_corners[1], court_corners[2], court_corners[3], (0, 0, 255))

    # track players over frames
    if H is not None:
        res = track.proj_track(tracked, H, court, frame)
    else:
        res = court

    out.write(res)

    # cv2.imshow('frame', frame)
    # cv2.imwrite('frame.png',frame)

out.release()
cap.release()
cv2.destroyAllWindows()
