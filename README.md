# Counting fingers with opencv

> Quick code which counts how many fingers there are using contour detection.

> I built this to learn how contour detection works in this context.

>Requirements:
>> opencv (cv2), sklearn and numpy

>>Built in Python3.6

---

## Algorithm

1. Lets user choose two points that define a rectangle, this rectangle will be our Region Of Interest (ROI).
2. Once an ROI has been established, it will use 30 frames to calibrate the background image of that ROI.
A background image is an average of the 30 frames that will be used to detect motion (in our case, the hand) in front of
it.
3. Once we have a background image, iterate over frames from the camera until a movement has been detected in the ROI:
    3. Convert the ROI to grayscale and blur it, this helps computation.
    3. Movements are detected by subtracting the new ROI from the background ROI and applying a threshold filter to the
    result, this way everything equal to the background is turned to black and everything new (our "movement") gets
    turned into white pixels.
4. Compute the contour of the movement (our hand) in the ROI (we use the maximum contour detected, since it is the 
biggest thing the ROI, which is our hand).
5. Compute the convex hull of the max contour, it's center, palm center (bit lower than the hull's center) and the palm
radius (distance to furthest point in the hull).
6. Compute a circle around the palm center (some factor of the radius we computed earlier).
7. Use a bitwise AND between the circle and the max contour, this will give us a circle with some holes where there are
points that might be fingers.
8. Compute contours on the circle with holes we computed, every contour we get might be a finger (or a palm)!
9. Filter out the contours we got using some technical criteria to determine which ones represent a finger.
10. Now we have all contours that represent fingers!
11. do this all over again on each frame.
