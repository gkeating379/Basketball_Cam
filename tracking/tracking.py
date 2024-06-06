import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np


font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1
font_color = (58, 37, 245)
thickness = 2
line_type = 2


class Track:
    def __init__(self):    
        self.tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=100, minimum_matching_threshold=0.7)
        self.label = sv.LabelAnnotator()
        self.jersey_model = YOLO('models\\Jersey\\med\\best.pt')

    def get_object_tracks(self, frame, results):
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)
        # labels = [ f"{tracker_id}" for tracker_id in tracked_detections.tracker_id ]
        # print(tracked_detections.tracker_id)
        # print(len(detections), len(labels))
        # annotated_frame = self.label.annotate(scene=frame.copy(), detections=tracked_detections, labels=labels)
        # cv2.imshow('tracked', annotated_frame)

        return tracked_detections

    def proj_track(self, tracked_detections, H, court, frame):
        '''Draws the circle icons for each detected player onto the court

        tracked_detections => current tracked detections in the frame
        H => homography for the frame
        court => image to warp to
        '''
        marked = np.copy(court)
        for i, id in enumerate(tracked_detections.tracker_id):
            # project the coords (middle of bottom of box)
            y = tracked_detections.xyxy[i][3] - (tracked_detections.xyxy[i][3] - tracked_detections.xyxy[i][3])/10
            x = (tracked_detections.xyxy[i][2] + tracked_detections.xyxy[i][0])/2
            court_coords = np.matmul(H, np.atleast_2d([x, y, 1]).T).T[0]  # its ugly I hate np 1d arrays
            court_coords = (court_coords/court_coords[2])[:2]
            court_coords = np.ndarray.astype(court_coords, np.int64)

            # get jersey number (didn't work well)
            # jersey.get_jersey_number(frame, tracked_detections.xyxy[i], self.jersey_model)

            # draw icon onto court
            r = 20
            cv2.circle(marked, court_coords, r, [0, 0, 0], 2)

            # get the bottom-left corner of the text
            (text_width, text_height), baseline = cv2.getTextSize(f'{id}', font, font_scale, thickness)
            text_x = court_coords[0] - text_width // 2
            text_y = court_coords[1] + text_height // 2

            cv2.putText(marked, f'{id}',
                        (text_x, text_y),
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

        # cv2.imshow('Players on Court', marked)
        return marked

    def player_interp(bound, last_frame, frame):
        '''Attempt to track a bounding box into the new frame
        Used to find previous detections that the model cannot find'''
        (x1, y1, x2, y2) = bound

        # block out non bounding box
        mask = np.zeros_like(last_frame)
        mask[y1:y2, x1:x2] = 1

        last_features = cv2.goodFeaturesToTrack(last_frame, maxCorners=20, qualityLevel=0.1, minDistance=3, mask=mask, blockSize=5)
        new_feats, _, _ = cv2.calcOpticalFlowPyrLK(last_frame, frame, last_features, None, winSize=(11,11), maxLevel=3, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_COUNT,50,0.1))

        x_dist = np.average(new_feats[:, 0])
        y_dist = np.average(new_feats[:, 1])

        x1 += x_dist
        x2 += x_dist
        y1 += y_dist
        y2 += y_dist

        cv2.imshow('Interp', frame[y1:y2, x1:x2])

        return (x1, y1, x2, y2)
