import cv2
import numpy as np

# Tuning Sliders for Color
def nothing(x): pass
cv2.namedWindow('Tuning')
# HSV range for Yellow (can be adjusted for your specific LED lighting)
cv2.createTrackbar('H_Low', 'Tuning', 20, 179, nothing)
cv2.createTrackbar('S_Low', 'Tuning', 100, 255, nothing)
cv2.createTrackbar('V_Low', 'Tuning', 100, 255, nothing)

cap = cv2.VideoCapture(0)
count = 0
line_y = 400 # The "Tripwire" height
was_above = True

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. Convert to HSV (Better for color isolation than RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Get Slider Values
    h = cv2.getTrackbarPos('H_Low', 'Tuning')
    s = cv2.getTrackbarPos('S_Low', 'Tuning')
    v = cv2.getTrackbarPos('V_Low', 'Tuning')
    
    # Define Yellow Range (Upper is usually consistent, lower needs tuning)
    lower_yellow = np.array([h, s, v])
    upper_yellow = np.array([40, 255, 255])
    
    # 3. Create Mask & Clean Noise
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.medianBlur(mask, 7) 
    
    # 4. Find the "Mass" of the ball
    # Moments are much faster than Hough for simple counting
    M = cv2.moments(mask)
    if M["m00"] > 500: # If we see enough yellow pixels
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 2)
        
        # 5. Tripwire Logic
        if cy > line_y and was_above:
            count += 1
            was_above = False
            print(f"Juggles: {count}")
        elif cy < line_y:
            was_above = True

    # Draw the Tripwire
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Tripwire Tracker', frame)
    cv2.imshow('Color Mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()