import urllib.request
import cv2
import numpy as np


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter
    gause_img = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow('gausse', gause_img)

    # https://learnopencv.com/otsu-thresholding-with-opencv/
    # OTSU threshold
    otsu_threshold, image_result = cv2.threshold(gause_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image_result


# tou a inventar para reconhecer a bandeja
def find_circle(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening', opening)


def main():
    # url = 'http://192.168.1.66:4747/shot.jpg'
    # capture = cv2.VideoCapture(url)
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        cv2.imshow('video', frame)

        img = pre_processing(frame)
        cv2.imshow('pre_processing', img)
        find_circle(img)

        canny = cv2.Canny(img, 100, 75)
        cv2.imshow('canny', canny)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
