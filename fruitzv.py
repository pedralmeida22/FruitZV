import cv2
import numpy as np
import sys
from food import pre_processing2
import os


colors = dict()
colors["Green"] = ([0, 51, 25], [178, 255, 102])
colors["Red"] = ([17, 15, 100], [50, 56, 207])
colors["Blue"] = ([86, 31, 4], [220, 88, 50])
colors["Yellow"] = ([25, 146, 190], [175, 226, 247])
colors["Orange"] = ([49, 101, 222], [122, 194, 255])
colors["Black"] = ([0, 0, 0], [80, 80, 80])
colors["Pink"] = ([177, 127, 218], [217, 211, 255])
colors["White"] = ([230, 230, 230], [255, 255, 255])

food = {
    "Red Apple": [(1000, 40000), ("Red", "Pink")],
    "Sausage": [(1000, 40000), ("Red", "Pink")],
    "Eggplants": [(1000, 40000), ("Yellow", "Green")],
    "Vegetables": [(1000, 40000), ("Green", "White")],
    "Olives": [(1000, 40000), ("Black","Black")],
    "Green Apple": [(1000, 40000), ("Green")],
    "Banana": [(1000, 40000), ("Yellow")],
    "Orange": [(1000, 40000), ("Orange")],
    "Spaghetti": [(1000, 40000), ("Yellow", "Orange")]
}


def pre_processing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter
    gause_img = cv2.GaussianBlur(img, (5, 5), 0)

    # https://learnopencv.com/otsu-thresholding-with-opencv/
    # OTSU threshold
    otsu_threshold, image_result = cv2.threshold(gause_img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("otsu", image_result)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_OPEN, kernel)
    image_result = cv2.morphologyEx(image_result, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("ke", image_result)
    image_result = cv2.Canny(image_result, 100, 75)
    # cv2.imshow("canny", image_result)

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


def get_counturs(clone, img, frame):
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

            cv2.putText(frame, get_type(len(cantos), w, h), (550, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.imshow('contornos', clone)

            return cut_img, True
    return clone, False


def get_counturs_fruits(clone, img, frame):
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruits = []
    offset = 10
    for c in countours:
        area = cv2.contourArea(c)
        if 500 < area < 2000:
            perimetro = cv2.arcLength(c, True)
            cantos = cv2.approxPolyDP(c, 0.02 * perimetro, True)
            x, y, w, h = cv2.boundingRect(cantos)
            cv2.rectangle(clone, (x-offset, y-offset), (x + w + offset, y + h + offset), (255, 255, 0), 2)
            cut_img = clone[y - offset:y + h + offset, x - offset:x + w + offset]
            fruits.append(cut_img)

    if len(fruits) > 0:
        # print(str(len(fruits)))
        cv2.putText(frame, str(len(fruits)), (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 120, 120), 2)
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
        cv2.putText(img, i, (30, 30 + (offset * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        offset += 1


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append((filename, img))
    return images


def count_white_pixels(img):
    height, width = img.shape
    count = 0
    for i in range(0, height):
        for j in range(0, width):
            if img.item(i, j) == 255:
                count += 1
    return count


def to_binary(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gause_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    otsu_threshold, image_result = cv2.threshold(gause_img, 120, 255, cv2.THRESH_BINARY)

    return image_result


def find_match(dataset, frame):
    # print("finding match")
    min_diff = 160 * 120
    index = None

    i = 0
    for img in dataset:
        # cv2.imshow("i"+str(i), img)

        d = to_binary(img)
        x = to_binary(frame)

        d = cv2.resize(d, (160, 120))
        x = cv2.resize(x, (160, 120))

        sub = d - x
        # cv2.imshow("sub" + str(i), sub)
        w = count_white_pixels(sub)
        # print(w)
        if w < min_diff:
            min_diff = w
            index = i

        i += 1

    if min_diff > 500:
        return None
    else:
        return index


def get_nome_by_index(lista, index):
    tmp = lista[index][0].split('.')

    return tmp[0]


def main():
    # carregar dataset
    lista = load_images_from_folder('dataset')
    dataset = []
    for tuple in lista:
        # t[0]-nome t[1]-imagem
        dataset.append(tuple[1])

    capture = cv2.VideoCapture(0)

    isCamera = False
    while True:
        if len(sys.argv) == 1:
            ret, frame = capture.read()
            isCamera = True
        else:
            frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
            frame = cv2.resize(frame, (600, 430))

        # cv2.imshow('t e s t e', frame)
        #cv2.imshow('POW', frame2)

        img = pre_processing(frame)

        original = frame.copy()
        img, bandeja = get_counturs(original, img, frame)

        if bandeja:  # encontrou a bandeja

            pre_img_cut = pre_processing(img)
            # cv2.imshow('pre cut_img', pre_img_cut)

            original = img.copy()
            img, fruit_imgs = get_counturs_fruits(original, pre_img_cut, frame)
            # fruit_imgs é uma lista em que cada elemento é uma foto

            if isCamera is False:

                i = 0
                comidas = []
                for a in fruit_imgs:
                    # cv2.imshow("OIOI", testColor("Orange",a))
                    comidas.append(getFood(a.shape[0], a.shape[1], filterColor(a)))
                    i += 1

                if comidas:
                    comidas = [item for sublist in comidas for item in sublist]
                    show_food(frame, comidas)
            #--------------------
            else:

                frutas = []
                i = 0
                for f in fruit_imgs:
                    m = find_match(dataset, f)
                    # print(filterColor(f))
                    i += 1
                    # cv2.imshow("t e s t e" + str(i), testColor("Orange", f))

                    if m is not None:
                        fruta = get_nome_by_index(lista, m)
                        # print(fruta)
                        frutas.append(fruta)

                if frutas:
                    show_food(frame, frutas)

                # take photo
                # i = 0
                # for a in fruit_imgs:
                #     cv2.imwrite("dataset/pera.jpg", a)
                #     i += 1
                #     cv2.imshow("t e s t e" + str(i), a)

        else:
            cv2.putText(frame, "Base not found", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("n_fruits", img)
        cv2.imshow("Imagem", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
