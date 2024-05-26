import cv2
import numpy as np
import time
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

def create_mask(im):
    '''Creates mask of which regions to include for feature tracking
    
    im=> image to mask'''
    mask = np.zeros_like(im)
    #block out bottom with score stuff
    mask[:len(mask)//5] = 1 

    return mask
    
def get_affine_transform(cur_frame, old_frame, old_features):
    '''Computes the affine transform that best maps between the features in the 
    previous frame and the current frame.  Feature correspondec is determined through
    optical flow
    
    cur_frame => current frame in greyscale
    old_frame => last frame in greyscale
    old_features => previous set of features from last frame (find correspondences with)
    
    Returns 
    M => affine transform matrix
    features => features found in the cur_frame'''
    mask = create_mask(cur_frame)
    #TODO decide to recompute 100 each frame or prune over time
    features = cv2.goodFeaturesToTrack(cur_frame, maxCorners=200, qualityLevel=0.3, minDistance=3, mask=mask, blockSize=13)

    M = None
    if old_frame is not None:
        new_feats, _, _ = cv2.calcOpticalFlowPyrLK(old_frame, cur_frame, old_features, features, winSize=(21,21), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_COUNT,50,0.1))
        M, _ = cv2.estimateAffinePartial2D(old_features, new_feats)

        #draw line between previous and current location in diplay image
        # for old, new in zip(old_features, new_feats):
        #     old_p = old.ravel().astype(np.int64)
        #     new_p = new.ravel().astype(np.int64)
        #     cv2.line(frame, old_p, new_p, [255,0,0], 3)
        
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


def update_homography(frame, court, court_canny, old_frame, old_features, court_corners):
    '''Updates the homography between frame and court.  First tries to find corner by intersection
    of Hough Lines.  If this is not possible, then updates previous court boundary with motion found
    by the optical flow transform
    
    frame => current frame
    court => image of court to map to
    court_canny => image of boundary of court
    old_frame => previous frame
    old_features => features found in previous fram
    court_corners => 4 corners marking the boundary of the court in the previous frame
    
    Returns
    H => homography from frame to court
    court_corners => 4 corners marking the boundary of the court in the current frame
    frame_grey => current_frame in greyscale
    features => features found in this frame'''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    #homography 
    H, n1, n2, n3, n4 = homography.get_best_frame_homography(frame, court, court_canny)
    M, features = get_affine_transform(frame_gray, old_frame, old_features)
    H = None #TODO testing

    if H is None and court_corners is not None: #compute homography by the movement from the last used points if we cannot find a line
        dst_points = np.array([base_values['top-left'], base_values['top-right'], base_values['bot-left'], base_values['bot-right']])
        court_corners = update_drift_full(M, court_corners)
        H, _ = cv2.findHomography(court_corners, dst_points)

    else:
        court_corners = np.array([n1, n2, n3, n4])

    if H is not None:
        draw_projection(frame, court, H)

    return H, court_corners, frame_gray, features

