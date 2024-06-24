import cv2
import numpy as np
import pygame
import random
import time

def nothing(x):
    pass

# Initialize pygame
pygame.init()
font = pygame.font.SysFont(None, 48)

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bottle Controlled Hand")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the HSV range for light green
lower_green = np.array([22, 32, 113])
upper_green = np.array([81, 255, 255])

# Function to detect the color object and get its centroid
def detect_color_object(frame, lower_bound, upper_bound):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1000:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy, mask
    
    return None, None, mask

# Load the hand and apple images
hand_image = pygame.image.load("hand.png")
hand_image = pygame.transform.scale(hand_image, (128, 128))
hand_rect = hand_image.get_rect()
hand_width, hand_height = hand_rect.size

apple_image = pygame.image.load("apple.png")
apple_image = pygame.transform.scale(apple_image, (64, 64))
apple_rect = apple_image.get_rect()

# Initial hand position
smooth_hand_x, smooth_hand_y = width // 2, height // 2

# Smoothing factor
alpha = 0.2

# Score
score = 0

# Game loop
running = True
clock = pygame.time.Clock()
background = pygame.image.load("background.png")

def get_apple_coords():
    corner_coords = [(20, 20), (20, 500), (700, 20), (700, 500)]
    num = random.randint(0, 3)
    return corner_coords[num]

apple_coords = get_apple_coords()
progress_start_time = None

def display():
    screen.blit(background, (0, 0))
    screen.blit(apple_image, apple_coords)
    screen.blit(hand_image, (int(smooth_hand_x - hand_width // 2), int(smooth_hand_y - hand_height // 2)))
    score_text = font.render(f'Score: {score}', True, black)
    screen.blit(score_text, (screen.get_width()//2 - 50, 20))

def check_collision(hand_rect, apple_coords):
    hand_rect.x = smooth_hand_x - hand_width // 2
    hand_rect.y = smooth_hand_y - hand_height // 2
    apple_rect.x = apple_coords[0]
    apple_rect.y = apple_coords[1]
    return hand_rect.colliderect(apple_rect)

def display_progress_bar(apple_coords, progress):
    progress_bar_width = 64
    progress_bar_height = 10
    current_width = int(progress * progress_bar_width)
    
    # Draw the progress bar background
    pygame.draw.rect(screen, white, (apple_coords[0], apple_coords[1] + 64, progress_bar_width, progress_bar_height))
    # Draw the progress
    pygame.draw.rect(screen, green, (apple_coords[0], apple_coords[1] + 64, current_width, progress_bar_height))

while running:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect the light green object
    cx, cy, mask = detect_color_object(frame, lower_green, upper_green)
    
    if cx is not None and cy is not None:
        target_x = int(cx * width / frame.shape[1])
        target_y = int(cy * height / frame.shape[0])
        
        # Apply the smoothing
        smooth_hand_x = alpha * target_x + (1 - alpha) * smooth_hand_x
        smooth_hand_y = alpha * target_y + (1 - alpha) * smooth_hand_y
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill(black)
    
    display()

    if progress_start_time is not None:
        elapsed_time = time.time() - progress_start_time
        if elapsed_time <= 3:
            progress = elapsed_time / 3.0
            display_progress_bar(apple_coords, progress)
        else:
            score += 1
            apple_coords = get_apple_coords()
            progress_start_time = None
            print("Score:", score)
    else:
        if check_collision(hand_rect, apple_coords):
            progress_start_time = time.time()
    

    pygame.display.flip()
    clock.tick(60)
    
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # cv2.imshow('Frame', frame)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', result)
    
    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()
