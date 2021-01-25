import cv2
import numpy as np
import sys
from food import pre_processing2

colors = dict()
colors["Green"] = ([0, 51, 25], [178, 255, 102])
colors["Red"] = ([0, 0, 154], [204, 204, 255])
colors["Blue"] = ([102, 51, 0], [255, 153, 153])
colors["Yellow"] = ([0, 244, 244], [204, 255, 255])
colors["Orange"] = ([0, 128, 255], [153, 204, 255])
colors["Black"] = ([0, 0, 0], [80, 80, 80])
colors["Pink"] = ([177, 127, 218], [217, 211, 255])
# colors["Green"]=([37,110,153],[80,240,255])
# colors["Red"]=([0,110,153],[15,240,255])
# colors["Blue"]=([80,110,153],[140,240,255])
# colors["Yellow"]=([28,110,153],[32,240,255])
# colors["Orange"]=([15,110,153],[28,240,255])

food = {
    "Red Apple": [(1000, 4000), ("Red")],
    "Sausage": [(1000, 4000), ("Red", "Pink")],
    "Pepper": [(1000, 4000), ("Red", "Green", "Yellow")],
    "Eggplants": [(1000, 4000), ("Yellow", "Green")],
    "Bacon": [(1000, 4000), ("Red", "Pink")],
    "Spinach": [(1000, 4000), ("Green")],
    "Broccoli": [(1000, 4000), ("Green")],
    "Olives": [(1000, 4000), ("Green", "Black")],
    "Shrimp": [(1000, 4000), ("Orange", "Pink")],
    "Green Apple": [(1000, 4000), ("Green")],
    "Banana": [(1000, 4000), ("Yellow", "Green")],
    "Orange": [(1000, 4000), ("Orange")]
}


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


def get_counturs_fruits(clone, img, frame):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruits = []

    for c in countours:
        area = cv2.contourArea(c)
        if 500 < area < 2000:
            perimetro = cv2.arcLength(c, True)
            cantos = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            x, y, w, h = cv2.boundingRect(cantos)
            cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cut_img = clone[y:y + h, x:x + w]
            fruits.append(cut_img)

    if len(fruits) > 0:
        # print(str(len(fruits)))
        cv2.putText(frame, str(len(fruits)), (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return clone, fruits


def getFood(dim1, dim2, color):
    area=dim1*dim2
    resultados = []
    for key, value in food.items():
        if area in range(value[0][0], value[0][1]):
            for colors in value[1]:
                if colors == color:
                    resultados.append(key)

    return resultados


def testColor(color, img):
    low = np.array([colors[color][0][0], colors[color][0][1], colors[color][0][2]])
    high = np.array([colors[color][1][0], colors[color][1][1], colors[color][1][2]])
    mask = cv2.inRange(img, low, high)

    out = cv2.bitwise_and(img, img, mask=mask)
    return out


def filterColor(img):
    red = testColor("Red", img)
    yellow = testColor("Yellow", img)
    green = testColor("Green", img)
    blue = testColor("Blue", img)
    orange = testColor("Orange", img)
    black = testColor("Black", img)
    pink = testColor("Pink", img)
    # cv2.imshow('TESTE', orange)

    coloredPixels = []
    coloredPixels.append((cv2.countNonZero(pre_processing(red)), "Red"))
    coloredPixels.append((cv2.countNonZero(pre_processing(yellow)), "Yellow"))
    coloredPixels.append((cv2.countNonZero(pre_processing(green)), "Green"))
    coloredPixels.append((cv2.countNonZero(pre_processing(blue)), "Blue"))
    coloredPixels.append((cv2.countNonZero(pre_processing(orange)), "Orange"))
    coloredPixels.append((cv2.countNonZero(pre_processing(black)), "Black"))
    coloredPixels.append((cv2.countNonZero(pre_processing(pink)), "Pink"))

    max = 0
    col = ""

    for c in coloredPixels:
        # print(c)
        if c[0] > max:
            max = c[0]
            col = c[1]

    return col


def show_food(img, f):
    offset = 1
    for i in f:
        cv2.putText(img, i, (30, 30 + (offset * 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        offset += 1


def main():

    capture = cv2.VideoCapture(0)

    while True:
        if len(sys.argv) == 1:
            ret, frame = capture.read()
        else:
            frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
            frame = cv2.resize(frame, (600, 430))

        cv2.imshow('t e s t e', frame)
        #cv2.imshow('POW', frame2)

        img = pre_processing(frame)
        # cv2.imshow('pre', img)

        original = frame.copy()
        img, bandeja = get_counturs(original, img)

        if bandeja:  # encontrou a bandeja

            pre_img_cut = pre_processing(img)
            # cv2.imshow('pre cut_img', pre_img_cut)
            original = img.copy()
            img, fruit_imgs = get_counturs_fruits(original, pre_img_cut, frame)
            # fruit_imgs é uma lista em que cada elemento é uma foto
            i = 0
            comidas = []
            print("EY")
            for a in fruit_imgs:
                # cv2.imshow("OIOI", testColor("Orange",a))
                print(filterColor(a))
                print("EY2")
                comidas.append(getFood(a.shape[0], a.shape[1], filterColor(a)))
                i += 1
                # cv2.imshow("t e s t e" + str(i), a)

            if comidas:
                comidas = [item for sublist in comidas for item in sublist]
                show_food(frame, comidas)
                print(comidas)

        else:
            cv2.putText(frame, "Base not found", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("n_fruits", img)
        cv2.imshow("ooo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
