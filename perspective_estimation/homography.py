from collections import defaultdict
import numpy as np
import cv2
import time
import math

#global (based on output image of court)
base_values = {'top-right': (546, 107), 'top-left': (53, 107), 'bot-left': (53, 1039), 'bot-right': (546, 1039),
               'top-paint-left': (220, 107), 'top-paint-right':(380, 107), 'bot-paint-left': (220,1039), 'bot-paint-right':(380,1039)}

def frame_to_2means(frame):
    # Convert the image to a 1D array of floats
    frame = cv2.GaussianBlur(frame,(5,5),2,2)
    data = frame.reshape((-1, 3)).astype(np.float32)

    # Perform KMeans clustering
    K = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 50)
    attempts = 1
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, flags)

    # Convert the centers to integer values and fixed B/W
    centers = np.uint8(centers)
    centers[0], centers[1] = ([255,255,255], [0,0,0]) if np.sum(centers[0]) < np.sum(centers[1]) else ([0,0,0], [255,255,255])
    
    # Assign each pixel its center cluster color
    segmented_image = centers[labels]

    # Reshape the segmented image to match the original image shape
    segmented_image = segmented_image.reshape(frame.shape)

    #TODO see if this speeds up overall processing
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    return segmented_image

def get_polar_intersection(rho1, theta1, rho2, theta2):
    '''Gets the point where two polar lines intersect
    
    Follows https://math.stackexchange.com/questions/4704222/find-intersection-of-two-lines-in-polar-coordinates'''
    if theta1 - theta2 == 0: #parallel lines do not intersect
        return None
    x = (rho2 * math.sin(theta1) - rho1 * math.sin(theta2)) / (math.sin(theta1 - theta2))
    y = (rho1 * math.cos(theta2) - rho2 * math.cos(theta1)) / (math.sin(theta1 - theta2))

    return x, y

def get_dist(p1, p2):
    '''Gets euclidean distance'''
    out = 0
    for x1, x2 in zip(p1, p2):
        out += (x1 - x2)**2

    return out**0.5

def get_four_corners(l1, l2, offset, im_shape):
    '''Takes two lines forming the top and side of the court
    Returns the 4 points of the court corners
    Assumes they intersect

    l1 => top line (length)
    l2 => side line (width)
    offset => distance alon side line by y offset (eg corner is placed at intercept plus offset on the y)
    im_shape => width and height of the image'''

    length_to_width = 1.88
    rho1, theta1 = l1
    rho2, theta2 = l2

    #p1 is the intersection of the two lines
    x1, y1 = get_polar_intersection(rho1, theta1, rho2, theta2)
    right_side = theta2 > 1 #x1 > im_shape[1]/2

    #p2 is the point along the width specified by offset
    y2 = y1 + offset
    x2 = (rho2 - y2*math.sin(theta2)) / math.cos(theta2)

    #get length of sides of trapezoid
    minor_l = get_dist((x1,y1), (x2,y2))
    triangle_edge = minor_l * math.cos(theta2) if right_side else -minor_l * math.cos(theta2)
    major_l = minor_l*length_to_width - 2*triangle_edge

    #p3 is the point along the length 
    x3 = x1 + -major_l * math.sin(theta1) if right_side else x1 + major_l * math.sin(theta1) #negative if on the right
    y3 = y1 + major_l * math.cos(theta1) if right_side else y1 + -major_l * math.cos(theta1)

    #p4 is the diagonal with p1
    x4 = x3 + -minor_l * math.sin(theta2) if right_side else x3 + minor_l * math.sin(theta2)
    y4 = y3 + -minor_l * math.cos(theta2) if right_side else y3 + minor_l * math.cos(theta2)

    return (x1, y1), (x2, y2), (x3, y3), (x4, y4), right_side

def draw_point(image, p, color):
    '''Draws a point that may be float or int'''
    x, y = p
    if 0 <= x < len(image[0]) and 0 <= y < len(image):
        image[int(y)][int(x)] = color

    return image

def draw_polar_line(image, line, color):
    '''Draws a line in polar coords'''
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 20 * (-b))
    y1 = int(y0 + 20 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), color, 2)

    return image

def draw_four(image, p1, p2, p3, p4, color):
    '''Draws the quadraleteral connecting four points'''
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    p3 = (int(p3[0]), int(p3[1]))
    p4 = (int(p4[0]), int(p4[1]))

    cv2.line(image, p1, p2, color, 1)
    cv2.line(image, p1, p3, [0, 255, 255], 1)
    cv2.line(image, p3, p4, [255, 255, 0], 1)
    cv2.line(image, p2, p4, color, 1)

    return image

def find_best_projection(image, verts, horzs, court, court_canny):
    '''Finds the best vertical and horizontal bound for the court
    Returns: Homography from current view to top down view
    
    image => base input from camera
    verts => list of vertical lines in polar form [rho, theta]
    horzs => list of horizontal lines in polar form
    court => top down image of court
    court_canny => Canny edges of the top down image'''
    #compare horizontal and vertical line intersections
    #find the best two lines to represent court boundaries
    best_err = 99999999999999999999
    proj = []
    c = 625
    for vert in verts:
        for horz in horzs:
            #drawing for debugging
            p1, p2, p3, p4, right_side = get_four_corners(horz, vert, c, image.shape[:2])

            draw_four(image, p1, p2, p3, p4, (0, 0, 255))

            #determine if court end is on the right
            if right_side:
                dst = [base_values['top-left'], base_values['top-right'], base_values['bot-left'], base_values['bot-right']]
            else:
                dst = [base_values['bot-left'], base_values['bot-right'], base_values['top-left'], base_values['top-right']]

            img_proj, H = warp_image([p1, p2, p3, p4], dst, image, court)
            
            cur_err = get_reprojection_error(img_proj, court_canny)

            #update best H and the points that made it
            if cur_err < best_err:
                best_err = cur_err
                proj = H
                b1, b2, b3, b4 = p1, p2, p3, p4 
                
    return proj, b1, b2, b3, b4

def draw_k_best_lines(image, add_k, court, court_canny):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform Hough Line Transform
    # P is faster but needs to be converted and gave only vertical lines??
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    if lines is not None:
        lines = lines[:add_k] if len(lines) > add_k else lines  # Get the k best lines or all lines if less than k
    
        # Draw the k best lines on the original image
        verts = []
        horzs = []
        for line in lines:
            line = line[0]
            rho, theta = line

            if theta < 1 or 2 < theta < 2.6:
                verts.append(line)
                draw_polar_line(image, line, [255, 0, 0])
            else:
                horzs.append(line) 
                draw_polar_line(image, line, [0, 255, 0])    


        if verts == [] or horzs == []:
            return image, None, None, None, None, None
        
        H, p1, p2, p3, p4 = find_best_projection(image, verts, horzs, court, court_canny)

    return image, H, p1, p2, p3, p4

def warp_image(src_points, dst_points, img_from, img_to):
    '''Computes homography and warps image onto another'''

    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    H, _ = cv2.findHomography(src_points, dst_points)

    img_proj = cv2.warpPerspective(img_from, H, (img_to.shape[1], img_to.shape[0]), borderValue = [255,255,255])

    return img_proj, H

def get_reprojection_error(src, dst):
    '''Gets the reprojection error between two images
    Src is assumed to have empty regions without importance
    Consider using the Canny line representations'''  

    unweighted = np.where(dst != 0, src, dst)
    
    lowest_row = np.sum(np.sum(unweighted, axis=2), axis=1)
    lowest_row = np.where(lowest_row != len(src[0]) * 255 * 3)[0]
    lowest_row = lowest_row[-1] if len(lowest_row) > 0 else 0

    total = np.sum(unweighted[:lowest_row]) / lowest_row

    #total area within the court that is "court colored"

    return total

def get_best_frame_homography(frame, court, court_canny):
    '''Returns the best homography from the frame to the court'''
    start_time = time.time()

    kmeans = frame_to_2means(frame)
    lines, H, p1, p2, p3, p4 = draw_k_best_lines(kmeans, 10, court, court_canny)

    print(f'Frame Runtime: {(time.time() - start_time)} seconds')

    return H, p1, p2, p3, p4