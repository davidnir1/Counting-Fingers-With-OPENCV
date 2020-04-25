import cv2
from sklearn import metrics
import numpy as np
import time

# Display relevant globals
DRAW_PALM_CIRCLE = True
MARK_FALSE_DETECTIONS = True
SHOW_DETECTED_FINGERS_NUM_TEXT = True
MARK_DETECTED_FINGERS_TEXT = True
MARK_DETECTED_FINGERS = True
DRAW = True
MARK_PALM_CENTER_TEXT = True
MARK_PALM_CENTER = True
MARK = True
DRAW_CONTOURS_ON_DETECTED_HAND = True
DRAW_ROI_BOX = True
SHOW_DETECTION_VIEW = True
SHOW_PALM_CIRCLE_SUBTRACTED = True
USER_GIVE_COORDS_MSG = "Left button to select ROI boundaries, middle button to restart"
USER_CLICK_LEFT_TO_START = "Points set, Left click to start!"

# Other modifiers
REASONABLE_HEIGHT_RADIUS_MODIFIER = 0.1
REASONABLE_HEIGHT_PALM_DIAM_MODIFIER = 0.1
MIN_DIST_PALM_RADIUS_MODIFIER = 0.5
PALM_RADIUS_FACTOR = 0.6
BLUR_KERNEL_SHAPE = (5, 5)
ESC_BUTTON = 27
DEFAULT_FONT = cv2.QT_FONT_NORMAL

# Colors
BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)

def draw_square_around_pixel(image,pixel,dist,color,border_width=1):
    x,y = pixel
    top_left = (x-dist,y-dist)
    bot_right = (x+dist,y+dist)
    cv2.rectangle(image,top_left,bot_right,color,border_width)

def mouse_click_handler(event, x, y, flags, param):
    """
    Handles the event of a mouse click.
    Assumes param is a list [ coordinates_list, should_finish_bool ].
    """
    coords = (x,y)
    # param is expected to be [ coordinates_list, should_finish_bool ]
    coordinates = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        print("User clicked {} with button {}:".format(coords, str(event)))
        if len(coordinates) == 2: # no need for more points
            print("\tAlready have all needed coordinates, this means the user wants to start.")
            param[1] = True
        else:
            print("\tUser selected a new coord, adding to list.")
            coordinates.append(coords)
    if event == cv2.EVENT_MBUTTONDOWN: # user wants to reset his points
        coordinates.clear() # delete the old points
        print("User chose to restart by clicking middle mouse, dropped the old points.")

def order_points_for_rectangle(coordinates):
    """
    Assumes len(coordinates) == 2
    Returns the top left and bottom right corners inferred from given coordinates.
    """
    p1, p2 = coordinates
    xs = (p1[0],p2[0])
    ys = (p1[1],p2[1])
    top_left = (min(xs),min(ys))
    bot_right = (max(xs),max(ys))
    return top_left,bot_right

def get_ROI_coords(camera,seconds_between_frames=0.01):
    """
    Allows the user to click on frames from the camera to select two points that represent a square (our ROI).
    Left click to select 2 points, if 2 points are selected left click will start the algorithm.
    If clicked middle mouse, resets the selected points and starts over.
    ESC triggers an exit.
    Eventually returns the top_left and bot_right coordinates of the selected rectangle.
    """
    if camera is None:
        raise Exception("Camera is None.. are you kidding me??")
    get_coords_window_name = "ROI Coordinates Grabber"
    cv2.namedWindow(get_coords_window_name, cv2.WINDOW_AUTOSIZE)
    mouse_callback_param = [ [] , False ]
    cv2.setMouseCallback(get_coords_window_name,mouse_click_handler, mouse_callback_param)
    output_coords = None
    while not mouse_callback_param[1]: # while not finished...
        time.sleep(seconds_between_frames)
        _, frame = camera.read()
        frame = cv2.flip(frame, 1)
        y_center = frame.shape[0]//2
        coordinates = mouse_callback_param[0]
        for coord in coordinates: # draw the selected coords
            draw_square_around_pixel(frame, coord, 1, BGR_GREEN, 1)
        user_msg, user_msg_color = USER_GIVE_COORDS_MSG, BGR_RED
        if len(coordinates) == 2: # user selected 2 points, change the message and draw a rectangle from the two points
            user_msg, user_msg_color = USER_CLICK_LEFT_TO_START, BGR_GREEN
            p1,p2 = order_points_for_rectangle(coordinates) # get the top_left and bot_right points of the rectangle
            output_coords = (p1,p2)
            cv2.rectangle(frame, p1, p2, BGR_BLUE, 1)
        cv2.putText(frame, user_msg, (0, y_center), cv2.QT_FONT_NORMAL, 0.6, user_msg_color, 1)
        cv2.imshow(get_coords_window_name, frame)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ESC_BUTTON: # escape button means exit and finish the run
            print("User pressed the exit button (ESC), exiting...")
            exit()
    cv2.destroyAllWindows()
    print("Do not move the camera from here on..")
    return output_coords

def calibrate_background_image(camera,ROI_coords,num_of_frames=30):
    """
    Computes an average over num_of_frames of the region in the camera's frames dictated by the ROI_coords parameter.
    1. Crops, converts to GS (from BGR instead of RGB, as this distinguishes skin color better) and blurs the ROI region
    2. Computes accumulated average of all these crops
    3. Returns the background it computed
    """
    print("Calibrating background based on {} frames, it will take a few seconds.".format(num_of_frames))
    roi_tl,roi_br = ROI_coords
    background = None
    current_frame = 0
    while current_frame < num_of_frames:
        current_frame+=1
        _,frame = camera.read()
        frame = cv2.flip(frame,1)
        frame_and_roi_details = (frame,roi_tl,roi_br)
        gs_blurred_roi = get_gs_blurred_roi(frame_and_roi_details).astype("float")
        if background is None:
            background = gs_blurred_roi
        else:
            cv2.accumulateWeighted(gs_blurred_roi,background,0.5)
    print("Done calibrating, starting task...\n")
    return background.astype("uint8")

def get_gs_blurred_roi(frame_and_roi_details,cv2_conversion_modifier=cv2.COLOR_BGR2GRAY):
    """
    Crops the ROI region from the frame, converts to GS and blurs, returns the crop.
    We default to BGR to GRAY conversion, as BGR allows us to differentiate skin color from wall color better.
    Maybe it's just me being too white.
    """
    frame,roi_tl,roi_br = frame_and_roi_details
    roi_pixels = frame[roi_tl[1]:roi_br[1], roi_tl[0]:roi_br[0]]
    # convert to bgr as it separates skin color from wall colors better (i'm too white..)
    roi_gray = cv2.cvtColor(roi_pixels.copy(), cv2_conversion_modifier)
    roi_gray = cv2.GaussianBlur(roi_gray, BLUR_KERNEL_SHAPE, 0)

    return roi_gray

def get_max_contour_in_ROI(frame, background, threshold=30):
    """
    1. Computes diff between frame and background, then applies a binary threshold for 0-1 separation.
    2. Attempts to find contours on the result of the 0-1 separation.
    3. Returns the contour with the biggest area, as it is asumed to be the hand.
    """
    # 1
    diff = cv2.absdiff(background.astype("uint8"), frame)
    # 2
    _, separated_frame = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # 3
    contours, _ = cv2.findContours(separated_frame.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    max_contour = None
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
    return separated_frame,max_contour

def get_finger_contours(frame, ROI_coords, separated_frame, max_contour):
    """
    The bread and butter of this algorithm:
    1. Compute the convex hull of the maximum contour we have (we assume the max contour corresponds to the hand)
    2. Compute the 4 most outer points (top, bottom, left, right) so we can get some kind of bounding box for the hand
    3. Compute the center of the palm by getting averages of top-bottom and left-right (and go a bit lower on the
        computed y's value, since the palm's center is lower than the total center of the box we have)
    4. Compute the biggest distance from the palm's center
    5. Use the distance we have as radius to compute a circle around the palm
    6. Subtract the circle from the hand's contour, this will give us a circumference around the palm's center with
        holes in it wherever there are fingers and a wrist
    7. Find the "holes" by looking for contours in the result of 6
    """
    # 1
    convex_hull = cv2.convexHull(max_contour)
    # 2
    extreme_top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    extreme_left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    extreme_right = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
    # 3
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    dY = extreme_bottom[1] - extreme_top[1]
    dX = extreme_right[0] - extreme_left[0]
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2) + int(dY / 6)  # approx position of palm
    # 4
    euc_distances = \
    metrics.pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    max_palm_radius = euc_distances[euc_distances.argmax()]
    # 5
    palm_radius = int(max_palm_radius * PALM_RADIUS_FACTOR)
    palm_circle = np.zeros(separated_frame.shape[:2], dtype="uint8")
    cv2.circle(palm_circle, (cX, cY), palm_radius, 255, 1)
    # 6
    palm_circle_subtracted = cv2.bitwise_and(separated_frame, separated_frame, mask=palm_circle)
    if SHOW_PALM_CIRCLE_SUBTRACTED:
        cv2.imshow("Seperated frame - Palm",palm_circle_subtracted)
    # 7
    contours, _ = cv2.findContours(palm_circle_subtracted.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ROI_diam = max(dX, dY)
    return contours,(cX,cY),palm_radius,ROI_diam


def filter_out_bad_contours(ROI_coords,finger_contours,palm_center,palm_radius):
    """
    Receives the contours detected by our algorithm and attempts to filter out outliers by using the following heuristic
    A valid finger has to:
    1. Have a reasonable y value (it cannot be much lower than the palm's center, as we assume upwards pointed hands)
    2. Have a reasonable distance from the other valid fingers (cannot be too close)
    3. Have a reasonable distance from the palm's center (computed with some factor of the palm's radius)
    Returns valid fingers, invalid contours and the palm's real center coords.
    """
    top_left, bot_right = ROI_coords
    accepted_contours = []
    dropped_contours = []
    for contour in finger_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_coords = (x + top_left[0], y + top_left[1])
        min_dist = palm_radius * MIN_DIST_PALM_RADIUS_MODIFIER
        # 1
        point_in_reasonable_height = (y < palm_center[1] + palm_radius * REASONABLE_HEIGHT_RADIUS_MODIFIER)
        # 2
        point_in_reasonable_distance_from_others = not detected_point_too_close_to_existing(accepted_contours,contour_coords)
        # 3
        point_in_reasonable_distance_from_center = not point_too_close_to_center(contour_coords, palm_center, min_dist)
        if point_in_reasonable_height and point_in_reasonable_distance_from_others and point_in_reasonable_distance_from_center:
            accepted_contours.append(contour_coords)
        else:
            dropped_contours.append(contour_coords)
    real_palm_center = (palm_center[0]+top_left[0], palm_center[1]+top_left[1])
    output = (accepted_contours,dropped_contours,real_palm_center)
    return output



def detect_fingers(frame, ROI_coords, separated_frame, max_contour):
    """
    Manages the finger detection algorithm.
    1. Find all the contours that can be classified as fingers
    2. Filter out the outliers
    Returns finger coordinates, outlier coordinates, real palm center coordinates and palm radius.
    """
    # find all finger contours
    finger_contours,palm_center,palm_radius, ROI_diam = get_finger_contours(frame, ROI_coords, separated_frame, max_contour)
    # filter the contours to get only fingers
    finger_contours,false_contours,real_palm_center = filter_out_bad_contours(ROI_coords,finger_contours,palm_center,palm_radius)
    if SHOW_DETECTION_VIEW:
        cv2.imshow("Detection view",separated_frame)
    output = (finger_contours,false_contours,real_palm_center,palm_radius)
    return output


def points_clustered_badly(points):
    """
    Checks to see if average distance between points is lower than 10 pixels (this negates reading a single finger as
    more than 1 point in some cases.
    """
    if len(points) < 2:
        return False
    x, y = points[0]
    max_x, max_y, min_x, min_y = x, y, x, y
    for x, y in points[1:]:
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y
    dx = max_x - min_x
    dy = max_y - min_y
    return dx < 10 * len(points)

def euclidean_dist(x1, y1, x2, y2):
    """
    Computes basic euclidean distance.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def point_too_close_to_center(point, center, min_dist):
    """
    Decides if a point is too close to the given center by computing the euclidean distance between them and the minimal
    distance given.
    """
    p_x, p_y = point
    x, y = center
    euc_dist = euclidean_dist(p_x, p_y, x, y)
    return euc_dist < min_dist


def detected_point_too_close_to_existing(existing_points, new_point):
    """
    This tests if a point is too close on y axis or x axis or euclidean distance and return True if it's too close.
    This negates contours detected on the same finger (for example a finger that half of it is in the light).
    :param existing_points:
    :param new_point:
    :return:
    """
    x, y = new_point
    for p_x, p_y in existing_points:
        dx = np.abs(p_x - x)
        dy = np.abs(p_y - y)
        euc_dist = euclidean_dist(p_x, p_y, x, y)
        orthogonal_dist_too_small = dx < 15 and dy < 75
        euc_too_close = euc_dist < 15
        if orthogonal_dist_too_small and euc_too_close:
            return True
    return False

def draw_frame_with_overlay(window_name,frame,ROI_coords,finger_detection_results):
    """
    Draws the output to the screen, according to global variables at top of this script.
    """
    ROI_tl, ROI_br = ROI_coords
    if finger_detection_results is None: # no hand in ROI
        if DRAW_ROI_BOX:
            cv2.rectangle(frame, ROI_tl, ROI_br, BGR_BLUE, 1)
    else: # found a hand in ROI
        entire_hand_coords,finger_contours, false_contours, real_palm_center, palm_radius = finger_detection_results
        if DRAW_ROI_BOX:
            cv2.rectangle(frame, ROI_tl, ROI_br, BGR_GREEN, 1)
        if DRAW_CONTOURS_ON_DETECTED_HAND:
            cv2.drawContours(frame, [entire_hand_coords], -1, BGR_BLUE)
        if MARK_PALM_CENTER:
            draw_square_around_pixel(frame, real_palm_center, 2, BGR_RED, 1)
        if MARK_PALM_CENTER_TEXT:
            cv2.putText(frame, "Palm {}".format(real_palm_center), real_palm_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, BGR_RED, 1)
        if DRAW_PALM_CIRCLE:
            cv2.circle(frame, real_palm_center, palm_radius, BGR_RED, 1)
        for contour in finger_contours:
            if MARK_DETECTED_FINGERS:
                draw_square_around_pixel(frame, contour, 2, BGR_GREEN, 2)
            if MARK_DETECTED_FINGERS_TEXT:
                cv2.putText(frame,"Finger {}".format(contour), contour, cv2.FONT_HERSHEY_SIMPLEX, 0.3, BGR_GREEN, 1)
            if SHOW_DETECTED_FINGERS_NUM_TEXT:
                fingers_num_text = "Detected {} fingers".format(len(finger_contours))
                cv2.putText(frame, fingers_num_text, ROI_tl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR_RED, 1)
        for contour in false_contours:
            if MARK_FALSE_DETECTIONS:
                draw_square_around_pixel(frame, contour, 2, BGR_BLUE, 1)
    # show everything
    cv2.imshow(window_name, frame)


def find_hand_and_count_fingers(camera,ROI_background,ROI_coords,seconds_between_frames=0.01):
    """
    Draws frames from the camera, looks for a hand and fingers in them and outputs to the screen the results.
    Assumes given background is an already calibrated image that corresponds to the ROI_coords given.
    Will exit if ESC is pressed.
    """
    finished = False
    roi_tl,roi_br = ROI_coords
    find_hand_window_name = "Finger counter"
    cv2.namedWindow(find_hand_window_name, cv2.WINDOW_AUTOSIZE)
    while not finished:
        time.sleep(seconds_between_frames) # to slow down the loop a bit...
        _, frame = camera.read()
        frame = cv2.flip(frame,1)
        frame_and_roi_details = (frame, roi_tl, roi_br)
        roi_gray = get_gs_blurred_roi(frame_and_roi_details)
        # attempt to detect a hand in the ROI
        separated_frame, entire_hand_contour = get_max_contour_in_ROI(roi_gray,ROI_background)
        if entire_hand_contour is None: # we didn't find a hand
            results = None
        else:
            entire_hand_coords =  entire_hand_contour + roi_tl
            finger_contours,false_contours,real_palm_center,palm_radius = detect_fingers(frame, ROI_coords, separated_frame, entire_hand_contour)
            results = (entire_hand_coords,finger_contours,false_contours,real_palm_center,palm_radius)
        # output everything to the screen
        draw_frame_with_overlay(find_hand_window_name,frame,ROI_coords,results)
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ESC_BUTTON: # escape button means exit and finish the run
            print("User pressed the exit button (ESC), exiting...")
            finished = True
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    # get the coordinates for the Region Of Interest (ROI) from the user
    ROI_coords = get_ROI_coords(camera)
    print("\nGot ROI coords: {}".format(ROI_coords))
    if ROI_coords is None or len(ROI_coords) == 0:
        raise Exception("Error: Roi_coords are empty??")
    # calibrate our background image for the ROI (this is used to detect motion in the ROI)
    ROI_background = calibrate_background_image(camera,ROI_coords)
    if ROI_background is None:
        raise Exception("Error: background is empty??")
    # start analyzing live frames from the camera to find fingers
    find_hand_and_count_fingers(camera, ROI_background, ROI_coords)