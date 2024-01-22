# Code reference: "https://www.computervision.zone/courses/pong-game-using-hand-gestures/"
# Football field image by Bence Balla-Schottner from "https://unsplash.com/photos/aerial-view-of-football-field-deGn9vSwXIM"

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Importing all images
img_background = cv2.imread("Stuffs/background.png")
img_game_over = cv2.imread("Stuffs/game_over.png")
img_ball = cv2.imread("Stuffs/football.png", cv2.IMREAD_UNCHANGED)
img_player1 = cv2.imread("Stuffs/player_1.png", cv2.IMREAD_UNCHANGED)
img_player2 = cv2.imread("Stuffs/player_2.png", cv2.IMREAD_UNCHANGED)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Variables
ballPos = [100, 100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]
score1_incremented = False
score0_incremented = False
winner = None

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, img_background, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            # Get the coordinates of the index finger tip
            x, y = hand['lmList'][8][0], hand['lmList'][8][1]

            h1, w1, _ = img_player1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 500)

            if hand['type'] == "Right":
                if img_player1.shape[2] == 4:  # Check if imgBat1 has an alpha channel
                    img[y1:y1 + h1, 59:59 + w1] = img_player1[:, :, :3] * (
                            1 - img_player1[:, :, 3:] / 255.0) + img_player1[:, :, 3:] / 255.0 * img[y1:y1 + h1, 59:59 + w1]
                else:
                    img[y1:y1 + h1, 59:59 + w1] = img_player1

                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    

            if hand['type'] == "Left":
                if img_player2.shape[2] == 4:  # Check if imgBat2 has an alpha channel
                    img[y1:y1 + h1, 1195 - w1:1195] = img_player2[:, :, :3] * (
                            1 - img_player2[:, :, 3:] / 255.0) + img_player2[:, :, 3:] / 255.0 * img[y1:y1 + h1, 1195 - w1:1195]
                else:
                    img[y1:y1 + h1, 1195 - w1:1195] = img_player2

                if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    

    # Game Over
    if (ballPos[0] <= 40 and 236 <= ballPos[1] <= 432) or (ballPos[0] >= 1200 and 236 <= ballPos[1] <= 432):
        gameOver = True

    if gameOver:
        img = img_game_over.copy()

        if (ballPos[0] >= 1200 and 236 <= ballPos[1] <= 432):
            winner = "Player 1"
        else:
            winner = "Player 2"

        # Update the score based on the winner
        if winner == "Player 1" and not score1_incremented:
            score[0] += 1
            score1_incremented = True
        elif winner == "Player 2" and not score0_incremented:
            score[1] += 1
            score0_incremented = True

        cv2.putText(img, f"{winner} Wins!", (390, 177), cv2.FONT_HERSHEY_COMPLEX, 2, (200, 0, 200), 5)
        cv2.putText(img, f"Score: {score[0]} - {score[1]}", (480, 345), cv2.FONT_HERSHEY_COMPLEX, 1.5, (200, 0, 200), 3)

    # If game not over move the ball
    else:
        # Move the Ball
        if ballPos[1] >= 670 or ballPos[1] <= 10:
            speedY = -speedY
        # Add conditions for bouncing back when the ball reaches specific areas
        if (ballPos[0] > 1200 and speedX > 0) or (ballPos[0] < 40 and speedX < 0):
            speedX = -speedX

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Draw the ball
        if img_ball.shape[2] == 4:  # Check if imgBall has an alpha channel
            img[ballPos[1]:ballPos[1] + img_ball.shape[0], ballPos[0]:ballPos[0] + img_ball.shape[1]] = \
                img_ball[:, :, :3] * (1 - img_ball[:, :, 3:] / 255.0) + img_ball[:, :, 3:] / 255.0 * \
                img[ballPos[1]:ballPos[1] + img_ball.shape[0], ballPos[0]:ballPos[0] + img_ball.shape[1]]
        else:
            # Resize the ball image to match the size of the region
            ball_region = img[ballPos[1]:ballPos[1] + img_ball.shape[0], ballPos[0]:ballPos[0] + img_ball.shape[1]]

            # Ensure the dimensions are valid before resizing
            if ball_region.shape[0] > 0 and ball_region.shape[1] > 0:
                resized_ball = cv2.resize(img_ball, (ball_region.shape[1], ball_region.shape[0]))

                # Assign the resized ball image to the region
                img[ballPos[1]:ballPos[1] + img_ball.shape[0], ballPos[0]:ballPos[0] + img_ball.shape[1]] = resized_ball



        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('p'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score0_incremented = False
        score1_incremented = False
    
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        score0_incremented = False
        score1_incremented = False
