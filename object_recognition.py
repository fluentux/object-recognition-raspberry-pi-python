"""Object recognition application with OpenCV on Raspberry Pi

Detects objects from video image from Raspberry Pi Camera. 
Training of the objects is done by user.

"""

import cv2
import numpy
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
from scipy.spatial import distance


class DetectedObject:
    """An object detected in the image.
    
    Detected objects are used in training and classifying the objects 
    recognized in image.
    
    Attributes:
        name: Name for the object.
        contour: Contour points.
        color: Object's color (blue, green, red).
    """
    
    def __init__(self, name, contour, color):
        self.name = name
        self.contour = contour
        self.color = color


def get_average_color(image, contour):
    """Calculates average BGR color value of the specified contour in image.
    
    Average color is calculated by using pixels inside contour area in the 
    image.
    
    Args:
        image: Image where the pixels inside contour are read.
        contour: Contour points that define the area where to calculate the 
                 average color.
                 
    Returns:
        Average BGR color inside contour in the image.
    """
    
    # Create mask for the detected object
    mask = numpy.zeros(image.shape[:2], dtype='uint8')
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Erode the mask slightly to reduce pixels from background
    mask = cv2.erode(mask, None, iterations=2)
    
    # Calculate average color (BGR) of the object
    return cv2.mean(image, mask=mask)[:3]


def classify_object(color, trained_objects):
    """Classifies object by comparing average color to trained object average 
       colors.
    
    Classification is done by color difference between object's average color 
    and trained object's average color. Object is classified to be the 
    trained object which has the smallest euclidean difference in average 
    color.
    
    Args:
        color: Average color of the object to be classified.
        trained_objects: List of trained objects which name is known.
    
    Returns:
        Name of the trained object that the object is classified to be.
    """
    
    min_distance = numpy.inf
    name = "unknown"
    for x in trained_objects:
        d = distance.euclidean(x.color, color)        
        if d < min_distance:
            name = x.name
            min_distance = d
            
    return name


def detect_objects(image, trained_objects):
    """Detects objects in image and tries to classify them.
    
    Objects are extracted from background with Otsu's thresholding method. 
    Then their contours are found. Each detected object is classified by 
    average color of the area inside object contour. Classification is done 
    by comparing the average color agains trained object average colors.
    
    Args:
        image: Image where the objects are detected.
        trained_objects: Known objects to compare detected objects against.
            
    Returns:
        Detected objects in the image.
    """
    
    detected_objects = []

    # Define minimum and maximum size of the objects
    image_height, image_width = image.shape[:2]
    object_minimum_width = 0.05 * image_width
    object_minimum_height = 0.05 * image_height
    object_maximum_width = 0.95 * image_width
    object_maximum_height = 0.95 * image_height

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    retval, gray = cv2.threshold(gray, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, 
                                                cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (w >= object_minimum_width and h >= object_minimum_height and \
            w <= object_maximum_width and h <= object_maximum_height):
            average_color = get_average_color(image, contour)
            name = classify_object(average_color, trained_objects)
            detected_object = DetectedObject(name, contour, average_color)
            detected_objects.append(detected_object)

    return detected_objects


def draw_detected_objects(image, detected_objects, clicked_object):
    """Draws detected objects on image.
    
    Bounding rectangle and name are drawn for each detected object. 
    
    Args:
        image: Image where the detected objects are drawn.
        detected_objects: List of detected objects to be drawn.
        clicked_object: Detected object that is clicked and is drawn with 
                        highlighting.
    """
    
    green_color = (0, 255, 0)
    red_color = (0, 0, 255)
    blue_color = (255,255,0)
    object_color = green_color
    text_color = blue_color
    text_offset_y = 10
    line_thickness = 2
    
    for detected_object in detected_objects:
        x,y,w,h = cv2.boundingRect(detected_object.contour)
        
        if detected_object is clicked_object:
            object_color = red_color
        else:
            object_color = green_color

        # Draw bounding rectangle for the detected object
        cv2.rectangle(image, (x, y), (x + w, y + h), object_color, 
                      line_thickness)
        
        # Draw name of the detected object
        cv2.putText(image, detected_object.name, (x, y - text_offset_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def get_clicked_object(detected_objects, click_position):
    """Determines which object is clicked if any.
    
    Tests if any object's bounding rectangle was hit and returns the object 
    that was closest to click position.
    
    Args:
        detected_objects: List of detected objects that are tested for 
                          clicking.
        click_position: Mouse-clicked position
        
    Returns:
        The object that was clicked if click location hit any object, 
        otherwise NoneType.
    """
    
    clicked_object = None
    
    if click_position != None:
        clicked_x = click_position[0]
        clicked_y = click_position[1]
        min_distance = numpy.inf
        
        # Determine the closest object to the mouse click position
        for detected_object in detected_objects:
            x,y,w,h = cv2.boundingRect(detected_object.contour)
            dist = distance.euclidean((x,y), (clicked_x, clicked_y))
            
            if dist < min_distance and \
            clicked_x >= x and clicked_x <= x + w and \
            clicked_y >= y and clicked_y <= y + h:
                clicked_object = detected_object
                min_distance = dist
    
    return clicked_object


def object_mouse_click(event, x, y, flags, param):
    """Handles mouse event click used for training objects.
    
    Sets the click position to mouse-clicked position when user clicks with 
    left mouse button.
    
    Args:
        event: Mouse event.
        x: The x-coordinate of the mouse event.
        y: The y-coordinate of the mouse event.
        flags: Mouse event flags.
        param: Optional parameters.
    """
    
    global click_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)


def main():
    """Sets up the camera and performs object recognition from live video
    
    Each image from camera is analyzed and objects are recognized. User can 
    train an object by clicking it on image.
    """
    
    # Initialize camera settings
    camera = PiCamera()
    camera.resolution = (1024, 768)
    camera.framerate = 25
    rawCapture = PiRGBArray(camera, size=(1024, 768))
    
    # Warmup the camera
    time.sleep(0.1)
    
    global click_position
    click_position = None
    trained_objects = []
    main_window_name = 'Object recognition'
    
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, 600, 200)
    cv2.setMouseCallback(main_window_name, object_mouse_click)
    
    # Video loop
    for frame in camera.capture_continuous(rawCapture, format="bgr", 
                                           use_video_port=True):
        image = frame.array
        
        detected_objects = detect_objects(image, trained_objects)
        
        clicked_object = get_clicked_object(detected_objects, click_position)
        
        draw_detected_objects(image, detected_objects, clicked_object)
        
        cv2.imshow(main_window_name, image)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Request name for an object if user clicked detected object
        if clicked_object != None:
            name = input("Type name for the object: ")
            if name:
                clicked_object.name = name
                trained_objects.append(clicked_object)
            click_position = None
        
        # Check if window ESC is pressed or window is closed
        if key == 27 or cv2.getWindowProperty(main_window_name, 0) == -1:
            break
    
        # Clear the stream to prepare for the next frame
        rawCapture.truncate(0)
    
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
