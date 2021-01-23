import cv2
import numpy as np


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter
    gause_img = cv2.GaussianBlur(img, (5, 5), 0)

    # https://learnopencv.com/otsu-thresholding-with-opencv/
    # OTSU threshold
    otsu_threshold, image_result = cv2.threshold(gause_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_CLOSE, kernel)

    image_result = cv2.Canny(image_result, 100, 75)

    return image_result


def get_type(n, w, h):
    if n == 3:
        type = 'triangulo'

    elif n == 4:
        calc = w / float(h)
        if 0.95 <= calc <= 1.05:
            type = "quadrado"
        else:
            type = "retangulo"

    elif n > 4:
        type = "circulo"

    else:
        type = ''

    return type


def get_counturs(clone, img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in countours:
        area = cv2.contourArea(c)
        if area > 800:  # para encontrar só elementos grandes (bandeja)
            # cv2.drawContours(clone, c, -1, (0, 255, 0), 2)
            perimetro = cv2.arcLength(c, True)
            cantos = cv2.approxPolyDP(c, 0.02*perimetro, True)
            x, y, w, h = cv2.boundingRect(cantos)

            cut_img = clone[y:y+h, x:x+w]

            cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 255, 0), 2)

            cv2.putText(clone, get_type(len(cantos), w, h), (x + (w // 2), (y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('contornos', clone)

            return cut_img
    return clone


def main():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        cv2.imshow('video', frame)

        img = pre_processing(frame)

        # res = frame.copy()
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
        # res[:, :, 3] = mask
        # if cv2.waitKey(1) & 0xFF == ord("p"):
        #     cv2.imwrite('retina_masked.png', res)

        imgc = frame.copy()
        img = get_counturs(imgc, img)
        cv2.imshow('cut_img', img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
