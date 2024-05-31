import supervision as sv
import cv2
import numpy as np
from . import jersey


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255,255,255)
thickness = 1
line_type = 2
class Track:
    def __init__(self):    
        self.tracker = sv.ByteTrack()
        self.label = sv.LabelAnnotator()

    def get_object_tracks(self, frame, results):
        detections = sv.Detections.from_ultralytics(results)
        tracked_detections = self.tracker.update_with_detections(detections)
        labels = [ f"{tracker_id}" for tracker_id in tracked_detections.tracker_id ]
        annotated_frame = self.label.annotate(scene=frame.copy(), detections=tracked_detections, labels=labels)

        cv2.imshow('tracked', annotated_frame)

        return tracked_detections

    def proj_track(self, tracked_detections, H, court, frame):
        '''Draws the circle icons for each detected player onto the court
        
        tracked_detections => current tracked detections in the frame
        H => homography for the frame
        court => image to warp to
        frame => image of frame'''
        marked = np.copy(court)
        for i, id in enumerate(tracked_detections.tracker_id):
            #project the coords (middle of bottom of box)
            y = tracked_detections.xyxy[i][3] - (tracked_detections.xyxy[i][3] - tracked_detections.xyxy[i][3])/10
            x = (tracked_detections.xyxy[i][2] + tracked_detections.xyxy[i][0])/2 
            court_coords = np.matmul(H, np.atleast_2d([x, y, 1]).T).T[0] #its ugly I hate np 1d arrays
            court_coords = (court_coords/court_coords[2])[:2]
            court_coords = np.ndarray.astype(court_coords, np.int64)

            #get jersey number

            #draw icon onto court
            r = 20
            cv2.circle(marked, court_coords, r, [252, 98, 3])

            #get the bottom-left corner of the text
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

        cv2.imshow('projas', marked)