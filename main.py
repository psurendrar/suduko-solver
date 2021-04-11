from typing import Optional, Any

import streamlit as st
from PIL import Image
from utils import *
import suduko_solver
import cv2
######################################################
def solve_suduko(img):
    heightImg = 450
    widthImg = 450
    model = intializePredectionModel()  # LOAD THE CNN MODEL

    # 1. PREPARE THE IMAGE
    # img=np.array(img.convert('RGB'))
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgThreshold = preProcess(img)

    # 2. FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # DRAW ALL DETECTED CONTOURS

    # 3. FIND THE BIGGEST COUNTOUR AND USE IT AS SUDOKU
    biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    # print(biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        # print(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)  # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        # print(len(boxes))
        # cv2.imshow("Sample",boxes[65])
        numbers = getPredection(boxes, model)
        # print(numbers)
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)
        # print(posArray)

        # 5. FIND SOLUTION OF THE BOARD
        board = np.array_split(numbers, 9)
        # print(board)
        try:
            suduko_solver.solve(board)
        except:
            pass

        flatList = []
        for sublist in board:
            for item in sublist:
                flatList.append(item)
        solvedNumbers = flatList * posArray
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

        # #### 6. OVERLAY SOLUTION
        pts2 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
        img = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
        return img

def main():
    """Suduko Solver App"""

    st.title("Suduko Solver")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Suduko Solver WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        img = Image.open(image_file)
        st.text("Original Image")
        st.image(img)

    if st.button("Solve"):
        result_img= solve_suduko(img)
        st.image(result_img)

#
if __name__ == '__main__':
    main()

