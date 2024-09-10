import cv2
import numpy as np

def detect_faces(img, cascade):
    # converte pra gray scale
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade eh um objeto pre treinado de um ml, talvez n de pra usar!
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame, biggest  # Return the face and its coordinates