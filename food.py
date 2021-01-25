import cv2
import numpy as np


def printImageFeatures(image):
    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    print("Image Height:", height)
    print("Image Width:", width)
    print("Number of channels:", nchannels)
    print("Number of elements:", image.size)


def pre_processing2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter
    gause_img = cv2.GaussianBlur(img, (9, 9), 0)

    # https://learnopencv.com/otsu-thresholding-with-opencv/
    # OTSU threshold
    otsu_threshold, image_result = cv2.threshold(gause_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('thres', image_result)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
    image_result = cv2.dilate(image_result, kernel, iterations=1)
    # cv2.imshow('erode_square', image_result)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('thing', image_result)

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


def cut_plate(clone, img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # TODO encontrar maior e curtar
    for c in countours:
        perimetro = cv2.arcLength(c, True)
        cantos = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        x, y, w, h = cv2.boundingRect(cantos)

        cut_img = clone[y:y + h, x:x + w]

        return cut_img


def get_counturs(clone, img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in countours:
        area = cv2.contourArea(c)
        print(area)
        if 1000 < area < 20000:  # para encontrar só elementos grandes (bandeja)
            cv2.drawContours(clone, c, -1, (0, 255, 0), 2)

            cv2.imshow('contornos', clone)


def main():
    img = "Comida/1-pepper/1-pepper-0.JPG"
    img2 = "Comida/2-eggplants/2-eggplants-0.JPG"
    img3 = "Comida/3-bacon/3-bacon-0.JPG"
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    if np.shape(image) == ():
        # Failed Reading
        print("Image file could not be open!")
        exit(-1)
    cv2.imshow('original', image)
    printImageFeatures(image)

    img = pre_processing2(image)
    original = image.copy()
    imgcut = cut_plate(original, img)
    n = pre_processing2(imgcut)
    cv2.imshow('pre', n)
    get_counturs(imgcut, n)

    # res = imgcut.copy()
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
    # res[:, :, 3] = n
    # cv2.imwrite('retina_masked.png', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
