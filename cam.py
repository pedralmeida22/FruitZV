import cv2
import numpy as np
colors = dict()
colors["Green"]=([0,51,25],[178,255,102])
colors["Red"]=([0,0,154],[204,204,255])
colors["Blue"]=([102,51,0],[255,153,153])
colors["Yellow"]=([0,244,244],[204,255,255])
colors["Orange"]=([0,128,255],[153,204,255])


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter
    gause_img = cv2.GaussianBlur(img, (5, 5), 0)

    # https://learnopencv.com/otsu-thresholding-with-opencv/
    # OTSU threshold
    otsu_threshold, image_result = cv2.threshold(gause_img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        if area > 2000:  # para encontrar só elementos grandes (bandeja)
            # cv2.drawContours(clone, c, -1, (0, 255, 0), 2)
            perimetro = cv2.arcLength(c, True)
            cantos = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            x, y, w, h = cv2.boundingRect(cantos)

            cut_img = clone[y:y + h, x:x + w]

            cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.putText(clone, get_type(len(cantos), w, h), (x + (w // 2), (y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.imshow('contornos', clone)

            return cut_img, True
    return clone, False



def get_counturs_fruits(clone, img):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruits = []

    for c in countours:
        area = cv2.contourArea(c)
        if 500 < area < 2000:
            # cv2.drawContours(clone, c, -1, (0, 255, 0), 1)
            perimetro = cv2.arcLength(c, True)
            cantos = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            x, y, w, h = cv2.boundingRect(cantos)
            fruits.append(c)

    if len(fruits) > 0:
        # print(str(len(fruits)))
        cv2.putText(clone, str(len(fruits)), ((x + 30), (y + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return clone

def testColor(color,img):
    low = np.array([colors[color][0][0],colors[color][0][1],colors[color][0][2]])
    high = np.array([colors[color][1][0],colors[color][1][1],colors[color][1][2]])
    mask = cv2.inRange(img, low, high)

    out = cv2.bitwise_and(img,img,mask = mask)
    return out

def filterColor(img):

    red = testColor("Red", img)
    yellow = testColor("Yellow", img)
    green = testColor("Green", img)
    blue = testColor("Blue", img)
    orange = testColor("Orange",img)
    #cv2.imshow('TESTE', orange)

    coloredPixels=[]
    coloredPixels.append((cv2.countNonZero(pre_processing(red)), "Red"))
    coloredPixels.append((cv2.countNonZero(pre_processing(yellow)), "Yellow"))
    coloredPixels.append((cv2.countNonZero(pre_processing(green)), "Green"))
    coloredPixels.append((cv2.countNonZero(pre_processing(blue)), "Blue"))
    coloredPixels.append((cv2.countNonZero(pre_processing(orange)), "Orange"))

    max = 0
    col = ""

    for c in coloredPixels:
        # print(c)
        if c[0] > max:
            max=c[0]
            col=c[1]

    return col

def main():
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        cv2.imshow('video', frame)

        img = pre_processing(frame)
        # cv2.imshow('pre', img)


        print(filterColor(frame))

        # res = frame.copy()
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
        # res[:, :, 3] = mask
        # if cv2.waitKey(1) & 0xFF == ord("p"):
        #     cv2.imwrite('retina_masked.png', res)

        original = frame.copy()
        img, bandeja = get_counturs(original, img)
        # cv2.imshow('cut_img', img)

        if bandeja:  # encontrou a bandeja
            pre_img_cut = pre_processing(img)
            # cv2.imshow('pre cut_img', pre_img_cut)

            img = get_counturs_fruits(original, pre_img_cut)

        else:
            cv2.putText(img, "Base not found", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("n_fruits", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
