# OpenCV Cheatsheet

## 1. Importing OpenCV
- import cv2  # Import OpenCV

## 2. Reading and Displaying Images
- img = cv2.imread('image.jpg')  # Read an image
- cv2.imshow('Window Name', img)  # Display image in a window
- cv2.waitKey(0)  # Wait for a key press
- cv2.destroyAllWindows()  # Close all OpenCV windows

## 3. Saving Images
- cv2.imwrite('output.jpg', img)  # Save image to file

## 4. Image Properties
- height, width = img.shape[:2]  # Get image dimensions
- img_size = img.size  # Get image size (in pixels)

## 5. Resizing Images
- img_resized = cv2.resize(img, (new_width, new_height))  # Resize image

## 6. Cropping Images
- img_cropped = img[y:y+h, x:x+w]  # Crop image

## 7. Converting Color Spaces
- img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
- img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV

## 8. Drawing Shapes
- cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)  # Draw rectangle
- cv2.circle(img, (center_x, center_y), radius, color, thickness)  # Draw circle
- cv2.line(img, (x1, y1), (x2, y2), color, thickness)  # Draw line

## 9. Adding Text
- cv2.putText(img, 'Text', (x, y), font, font_scale, color, thickness)  # Add text

## 10. Image Filtering
- img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)  # Apply Gaussian blur
- img_edges = cv2.Canny(img, threshold1, threshold2)  # Detect edges using Canny

## 11. Thresholding
- _, img_thresh = cv2.threshold(img_gray, threshold_value, max_value, cv2.THRESH_BINARY)  # Apply binary threshold

## 12. Morphological Operations
- img_dilated = cv2.dilate(img, kernel, iterations)  # Dilation
- img_eroded = cv2.erode(img, kernel, iterations)  # Erosion

## 13. Contours
- contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
- cv2.drawContours(img, contours, contour_index, color, thickness)  # Draw contours

## 14. Face Detection
- face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Load Haar Cascade
- faces = face_cascade.detectMultiScale(img_gray, scaleFactor, minNeighbors)  # Detect faces

## 15. Video Capture
- cap = cv2.VideoCapture(0)  # Start video capture
- while True:
  - ret, frame = cap.read()  # Read frame from video
  - cv2.imshow('Video', frame)  # Display frame
  - if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
    - break
- cap.release()  # Release video capture
- cv2.destroyAllWindows()  # Close all OpenCV windows
