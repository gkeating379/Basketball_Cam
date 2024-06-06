import cv2
import numpy as np
from . import homography

base_values = {'top-right': (546, 107), 'top-left': (53, 107),
               'bot-left': (53, 1039), 'bot-right': (546, 1039),
               'top-paint-left': (220, 107), 'top-paint-right': (380, 107),
               'bot-paint-left': (220, 1039), 'bot-paint-right': (380, 1039)}


def get_dist(p1, p2):
    '''Gets euclidean distance between two same dimension points

    p1 => n dimension vector
    p2 => n dimension vector

    Return euclidean distance between p1 and p2'''
    out = 0
    for x1, x2 in zip(p1, p2):
        out += (x1 - x2)**2

    return out**0.5


def update_drift(drift_x, drift_y, points):
    '''Update the points by adding the drift to each point

    drift_x => camera motion in the x axis of the image
    drift_y => camera motion in the y axis of the image
    points => points to apply the drift to

    Return the points with the x and y drift applied'''
    for i in range(len(points)):
        points[i][0] += drift_x
        points[i][1] += drift_y
    return points


def update_drift_full(M, points):
    '''Update the points by preforming the affine transform to each point

    M => affine matrix
    points => points to apply the matrix

    Returns M*points'''
    for i in range(len(points)):
        points[i] = np.matmul(M, np.append(points[i], 1).T)
    return np.array(points)


def create_mask(im, court_corners, detections):
    '''Creates mask of which regions to include for feature tracking

    im => image to mask
    court_corners => four corners of court
    detections => list of object detections in xyxy

    Return
    mask => mask over im where 1 is region to not consider
            mask out all regions above the court, inside a detection box
            and too far to the bottom of the image'''
    mask = np.ones_like(im)

    # mask everything above the court
    if court_corners is not None:
        lowest_val = len(im)
        for (x, y) in court_corners:
            lowest_val = int(y) if y < lowest_val else lowest_val
        mask[:lowest_val] = 0
    else:  # guess top third is crowd
        mask[:len(mask)//3] = 0

    # mask all players
    buffer = 30
    for detection in detections:
        for (x1, y1, x2, y2) in detection:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            mask[y1-buffer:y2+buffer, x1-buffer:x2+buffer] = 0

    # mask bottom part of screen where score is
    # remove static regions
    mask[-len(mask)//4:] = 0

    # remove left and right bounds where detection gets bad
    mask[:, -len(mask[0])//10:] = 0
    mask[:, :len(mask[0])//10] = 0

    return mask


def get_affine_transform(frame, cur_frame, old_frame, old_features,
                         court_corners, detections):
    '''Computes the affine transform that best maps between the features in the
    previous frame and the current frame.  Feature correspondce is determined
    through optical flow

    frame => current frame normal (used for displaying feature corresondences)
    cur_frame => current frame in greyscale
    old_frame => last frame in greyscale
    old_features => previous set of features from last frame
                    (find correspondences with)
    detections => list of object detections in xyxy

    Returns
    M => affine transform matrix
    features => features found in the cur_frame'''
    if court_corners is None or court_corners[0] is None:
        return None, None
    mask = create_mask(cur_frame, court_corners, detections)
    features = cv2.goodFeaturesToTrack(cur_frame, maxCorners=200,
                                       qualityLevel=0.1, minDistance=10,
                                       mask=mask, blockSize=13)
    M = None
    if old_frame is not None and old_features is not None:
        criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_COUNT, 50, 0.1)
        new_feats, _, _ = cv2.calcOpticalFlowPyrLK(old_frame,
                                                   cur_frame,
                                                   old_features,
                                                   features,
                                                   winSize=(21, 21),
                                                   maxLevel=5,
                                                   criteria=criteria)
        M, _ = cv2.estimateAffinePartial2D(old_features, new_feats)

        # draw line between previous and current location in diplay image
        for old, new in zip(old_features, new_feats):
            old_p = old.ravel().astype(np.int64)
            new_p = new.ravel().astype(np.int64)
            cv2.line(frame, old_p, new_p, [255, 0, 0], 3)

    return M, features


def draw_projection(frame, court, H):
    '''Project the frame onto the court image and show the result

    frame => current frame
    court => top down court view to project to
    H => homography of the projection

    Return projected image'''
    img_proj = cv2.warpPerspective(frame, H, (court.shape[1], court.shape[0]))
    proj_on_court = np.where(img_proj == 0, court, img_proj)
    cv2.imshow('projection.png', proj_on_court)

    return proj_on_court


def update_homography(frame, court, old_frame, old_features, court_corners,
                      detections):
    '''Updates the homography between frame and court.  First tries to find
    corner by intersection of Hough Lines.  If this is not possible, then
    updates previous court boundary with motion found by the optical flow
    transform

    frame => current frame
    court => image of court to map to
    old_frame => previous frame
    old_features => features found in previous fram
    court_corners => 4 corners marking the boundary of the court in the
                     previous frame
    detections => list of object detections in xyxy

    Returns
    H => homography from frame to court
    court_corners => 4 corners marking the boundary of the court in the
                     current frame
    frame_grey => current_frame in greyscale
    features => features found in this frame'''
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # homography by lines
    H_lines, n1, n2, n3, n4, canny = homography.get_best_frame_homography(frame, court)
    n_corners = np.array([n1, n2, n3, n4])

    # homography by affine
    M, features = get_affine_transform(frame, frame_gray, old_frame,
                                       old_features, court_corners, detections)

    # if there is an affine
    if M is not None:
        # assign correct court orientation
        if court_side(court_corners):
            dst_points = np.array([base_values['top-left'],
                                   base_values['top-right'],
                                   base_values['bot-left'],
                                   base_values['bot-right']])
        else:
            dst_points = np.array([base_values['bot-left'],
                                   base_values['bot-right'],
                                   base_values['top-left'],
                                   base_values['top-right']])

        court_corners = update_drift_full(M, court_corners)
        H_affine, _ = cv2.findHomography(court_corners, dst_points)

        # if there is an affine and a lines pick the one with lowest line error
        if H_lines is not None:

            affine_l1 = homography.eval_line(court_corners[0], court_corners[1], canny)
            affine_l2 = homography.eval_line(court_corners[0], court_corners[2], canny)
            affine_score = (affine_l1 + affine_l2) / 2

            lines_l1 = homography.eval_line(n1, n2, canny)
            lines_l2 = homography.eval_line(n1, n3, canny)
            lines_score = (lines_l1 + lines_l2) / 2

            if affine_score > lines_score:
                H = H_affine
                court_corners = court_corners
            else:
                H = H_lines
                court_corners = n_corners

        else:
            H = H_affine

    else:
        H = H_lines
        court_corners = np.array([n1, n2, n3, n4])

    # verify the projection visually
    # if H is not None:
    #     draw_projection(frame, court, H)

    return H, court_corners, frame_gray, features


def court_side(corners):
    '''Returns True if the court is looking at the right side

    corners => 4 court corners'''
    if corners[0][0] < corners[2][0]:
        return False
    return True
