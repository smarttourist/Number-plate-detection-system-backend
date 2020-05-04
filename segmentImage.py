import cv2
import numpy as np
from recognise import recognise
kernal=np.ones((1,1),np.uint8)

def square(img):

    """
    This function resize non square image to square one (height == width)
    :param img: input image as numpy array
    :return: numpy array
    """

    # image after making height equal to width
    squared_image = img

    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))

        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))

        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image


def segmentImage(platel):
    resStr=''
    plate=cv2.cvtColor(platel, cv2.COLOR_BGR2GRAY)
    plate = cv2.bilateralFilter(plate, 5, 15, 15)  #21515
    height=plate.shape[0]
    width=plate.shape[1]
    area = height * width
    #print(height,width)
    scale1 = 0.01
    scale2 = 0.1
    area_condition1 = area * scale1
    area_condition2 = area * scale2





    cv2.imshow(" Plate", plate)
    ret3, th3 = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(hierarchy)
    except Exception as e:
        print(e)

    contours = sorted(contours, key=cv2.boundingRect, reverse=False)
    cropped = []
    results = []
    i=0
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)


        if (w * h > area_condition1 and w * h < area_condition2 and w / h > 0.3 and h / w > 0.3):
            cv2.drawContours(plate, [cnt], 0, (0, 255, 0), 1)

            cv2.rectangle(platel, (x, y), (x + w, y + h), (255, 0, 0), 1)
            c = th3[y:y + h, x:x + w]
            c = np.array(c)
            c = cv2.bitwise_not(c)
            #c = square(c)
            c = cv2.resize(c, (28, 28), interpolation=cv2.INTER_AREA)
            cv2.imshow("c", c)
            cv2.imshow(str(i), c)

            result=recognise(c)
            #print(c)
            cropped.append(c)
            if(result!=None):
                results.append(result)
            i+=1
    i=0
    cv2.imshow("done", platel)
    for i in results:
        resStr += str(i)
    #print(resStr)
    return resStr

