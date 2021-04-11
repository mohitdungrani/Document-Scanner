import cv2
import numpy as np


imageWidth = 480
imageHeight = 630
cap = cv2.VideoCapture(0)
cap.set(3,imageWidth)
cap.set(4,imageHeight)
cap.set(10,130)

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,300)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=2)
    return imgThres

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0
    greatest = np.array([])
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 5000:
            # cv2.drawContours(imgCount,cont,-1,(255,0,0))
            peri = cv2.arcLength(cont,True)
            approx = cv2.approxPolyDP(cont,0.02*peri, True)
            if max_area < area and len(approx) == 4:
                greatest = approx
                max_area = area
    cv2.drawContours(imgCount,greatest,-1,(255,0,0),20)
    print(greatest)
    return greatest

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWrap(img, biggest):
    if len(biggest) == 4:
        biggest = reorder(biggest)
        pst1 = np.float32(biggest)
        pst2 = np.float32(([0,0],[imageWidth,0],[0,imageHeight],[imageWidth,imageHeight]))
        matrix = cv2.getPerspectiveTransform(pst1,pst2)
        # print(imageWidth,imageHeight)
        imgOutput = cv2.warpPerspective(img,matrix,(imageWidth,imageHeight))
        # imgOutput = cv2.wrapPerspective(img,matrix, (imageWidth, imageHeight))

        return imgOutput
    return(img)

while True:
    flage, img = cap.read()
    cv2.resize(img,(imageWidth,imageHeight))
    imgCount = img.copy()
    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    imgWraped = getWrap(img,biggest)

    cv2.imshow('scanned',imgWraped)
    cv2.imshow('thresh',imgThres)
    cv2.imshow(" Live Video ", imgCount)
    if cv2.waitKey(1) == ord('q'):
        break
