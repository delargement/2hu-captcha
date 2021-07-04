import cv2
import sys
import os.path

CASCADE_FILE = "../model/lbpcascade_animeface.xml"


def detect(filename, outputname, cascade_file=CASCADE_FILE):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(50, 50))
    i = 0
    for (x, y, w, h) in faces:
        cropped = image[y: y + h, x: x + w]
        cv2.imwrite(outputname, cropped)
        i = i + 1
    return len(faces)
