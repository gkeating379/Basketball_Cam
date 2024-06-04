import cv2
import numpy as np
from . import homography

base_values = {'top-right': (546, 107), 'top-left': (53, 107), 'bot-left': (53, 1039), 'bot-right': (546, 1039),
               'top-paint-left': (220, 107), 'top-paint-right':(380, 107), 'bot-paint-left': (220,1039), 'bot-paint-right':(380,1039)}


def get_dist(p1, p2):
    '''Gets euclidean distance'''
    out = 0
    for x1, x2 in zip(p1, p2):
        out += (x1 - x2)**2

    return out**0.5

def update_drift(drift_x, drift_y, points):
    '''Update the points by adding the drift to each point'''
    for i in range(len(points)):
        points[i][0] += drift_x
        points[i][1] += drift_y
    return points

def update_drift_full(M, points):
    '''Update the points by preforming the affine transform to each point'''
    for i in range(len(points)):
        points[i] = np.matmul(M, np.append(points[i], 1).T)
    return np.array(points)

def create_mask(im, court_corners, detections):
    '''Creates mask of which regions to include for feature tracking
    
    im=> image to mask
    court_corners => four corners of court
    detections => list of object detections in xyxy
    
    Return
    mask => mask over im where 1 is region to not consider'''
    mask = np.ones_like(im)

    #mask everything above the court
    if court_corners is not None:
        lowest_val = len(im)
        for (x, y) in court_corners:
            lowest_val = int(y) if y < lowest_val else lowest_val
        mask[:lowest_val] = 0 
    else: #guess top third is crowd
        mask[:len(mask)//3] = 0

    #mask all players
    buffer = 30
    for detection in detections:
        for (x1, y1, x2, y2) in detection:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1-buffer:y2+buffer, x1-buffer:x2+buffer] = 0

    #mask bottom part of screen where score is
    #remove static regions
    mask[-len(mask)//4:] = 0

    #remove left and right bounds where detection gets bad
    mask[:, -len(mask[0])//10:] = 0
    mask[:, :len(mask[0])//10] = 0

    return mask
    
def get_affine_transform(frame, cur_frame, old_frame, old_features, court_corners, detections):
    '''Computes the affine transform that best maps between the features in the 
    previous frame and the current frame.  Feature correspondec is determined through
    optical flow
    
    frame => current frame normal (used for displaying feature corresondences)
    cur_frame => current frame in greyscale
    old_frame => last frame in greyscale
    old_features => previous set of features from last frame (find correspondences with)
    detections => list of object detections in xyxy
    
    Returns 
    M => affine transform matrix
    features => features found in the cur_frame'''
    mask = create_mask(cur_frame, court_corners, detections)
    #cv2.imshow('masked image', mask*cur_frame)
    features = cv2.goodFeaturesToTrack(cur_frame, maxCorners=200, qualityLevel=0.1, minDistance=10, mask=mask, blockSize=13)
    M = None
    if old_frame is not None:
        new_feats, _, _ = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, old_features, features, winSize=(21,21), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_COUNT,50,0.1))
        M, _ = cv2.estimateAffinePartial2D(old_features, new_feats)

        #draw line between previous and current location in diplay image
        for old, new in zip(old_features, new_feats):
            old_p = old.ravel().astype(np.int64)
            new_p = new.ravel().astype(np.int64)
            cv2.line(frame, old_p, new_p, [255,0,0], 3)

        
    return M, features

def draw_projection(frame, court, H):
    '''Project the frame onto the court image
    
    frame => current frame
    court => top down court view to project to
    H => homography of the projection'''
    img_proj = cv2.warpPerspective(frame, H, (court.shape[1], court.shape[0]))
    proj_on_court = np.where(img_proj == 0, court, img_proj)
    cv2.imshow('projection.png', proj_on_court)

    return proj_on_court


def update_homography(frame, court, court_canny, old_frame, old_features, court_corners, detections):
    '''Updates the homography between frame and court.  First tries to find corner by intersection
    of Hough Lines.  If this is not possible, then updates previous court boundary with motion found
    by the optical flow transform
    
    frame => current frame
    court => image of court to map to
    court_canny => image of boundary of court
    old_frame => previous frame
    old_features => features found in previous fram
    court_corners => 4 corners marking the boundary of the court in the previous frame
    detections => list of object detections in xyxy
    
    Returns
    H => homography from frame to court
    court_corners => 4 corners marking the boundary of the court in the current frame
    frame_grey => current_frame in greyscale
    features => features found in this frame'''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    #homography by lines
    H_lines, n1, n2, n3, n4, canny = homography.get_best_frame_homography(frame, court, court_canny)
    #TODO testing affine transformation
    #H_lines = None
    #court_corners = np.array([n1, n2, n3, n4])

    #homography by affine
    M, features = get_affine_transform(frame, frame_gray, old_frame, old_features, court_corners, detections)
    #TODO testing homography
    #M = None if H_lines is not None else M
    if M is not None:
        dst_points = np.array([base_values['top-left'], base_values['top-right'], base_values['bot-left'], base_values['bot-right']])
        court_corners = update_drift_full(M, court_corners)
        H_affine, _ = cv2.findHomography(court_corners, dst_points)
        
        if H_lines is not None:
                
            affine_score = (homography.eval_line(court_corners[0], court_corners[1], canny) + homography.eval_line(court_corners[0], court_corners[2], canny)) / 2
            lines_score = (homography.eval_line(n1, n2, canny) + homography.eval_line(n1, n3, canny)) / 2
            
            H = H_affine if affine_score > lines_score else H_lines
            court_corners = court_corners if affine_score > lines_score else np.array([n1, n2, n3, n4])
            if affine_score > lines_score:
                print('AFFINE')
            else:
                print('Lines')
        else:
            H = H_affine
            print('AFFINE')

    else:
        H = H_lines
        print('Lines')
        court_corners = np.array([n1, n2, n3, n4])

    # verify the projection visually
    # if H is not None:
    #     draw_projection(frame, court, H)

    return H, court_corners, frame_gray, features

