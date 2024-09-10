import cv2
import numpy as np

def detect_faces(img, cascade):
    # converte pra gray scale, pra trabalhar melhor com o cascade
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade é um objeto pre treinado de um ml
    coords = cascade.detectMultiScale(gray_frame, 1.1, 5)
    # retornando as coordenadas das faces encontradas a partir do modelo do ml
    frame = []
    for (x, y, w, h) in coords:
        frame.append(img[y:y + h, x:x + w])
    return frame

def detect_eyes(img, cascade):
    # converte pra gray scale, pra trabalhar melhor com o cascade
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cascade é um objeto pre treinado de um ml
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)
    # numero de colunas da imagem
    width = np.size(img, 1)  
    # numero de linhas da imagem
    height = np.size(img, 0)  
    #inicializa olho esquerdo e direito
    left_eye = None
    right_eye = None
    
    # x e y são as coordenadas do top-left do olho sendo analisado (que é um crop da imagem)
    for (x, y, w, h) in eyes:
        # eliminando casos onde uma falsa detecção do olho acontece
        if y > height / 2:
            pass
        # pega a coordenada central do olho
        eyecenter = x + w / 2 
        # verifica se é o olho esquerdo ou direito
        if eyecenter < width * 0.5:
            left_eye = (x, y, w, h)
        else:
            right_eye = (x, y, w, h)
    
    return left_eye, right_eye

def cut_eyebrows(img):
    # pegando altura e largura da imagem (ignorando os channels)
    height, width = img.shape[:2]
    # valor da altura para sobrancelha encontrado depois de testes
    eyebrow_h = int(height / 4)
    
    img = img[eyebrow_h:height, 0:width]
    return img

def redness_measure(image):
    # Converte a imagem para o formato ponto flutuante para precisão
    image_float = image.astype(np.float32)

    # Separa os canais Vermelho, Azul e Verde
    B, G, R = cv2.split(image_float)

    # Calcula a  medida de vermelhidão a partir da formula do paper
    redness = (R - np.minimum(B, G)) * 255 / (np.maximum(B, G) + 1) 
    redness = np.clip(redness, 0, 255).astype(np.uint8)
    
    return redness

def skin_color_removal(image):
    # O YCbCr é um espaço de cores usado em sistemas de vídeo e fotografia digital
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # definindo nossos limites
    lower_skin = np.array([0, 133, 77])
    upper_skin = np.array([255, 173, 127])

    # definindo a máscara para pele
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # aplicando a máscara inversa, para remover a pele
    non_skin_mask = cv2.bitwise_not(skin_mask)
    
    return non_skin_mask

# Parâmetros: imagem, limite do vermelho, limite de proporcao de vermelho (obtido a partir de testes)
def detect_red_eye(image, redness_threshold=150, redness_ratio_threshold=0.007):
    
    # detectando vermelhidao na imagem
    redness = redness_measure(image)

    # mascara "nao pele"
    non_skin_mask = skin_color_removal(image)
    
    # aplicando a mascara
    redness[non_skin_mask == 0] = 0

    # preenchendo os espacos utilizando o "closing" operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_redness = cv2.morphologyEx(redness, cv2.MORPH_CLOSE, kernel)

    # rotula os componentes conectados
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(closed_redness, connectivity=4)

    # criando uma imagem "em branco" com o tamanho da imagem de entrada, do tipo RGB com 3 canais de cores
    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # pegando a area total da imagem
    total_eye_area = np.size(image, 0) * np.size(image, 1)
    total_red_area = 0
    
    for i in range(1, num_labels):  # Começa em 1 porque 0 é o background
        mask = (labels == i)
        
        # calculando numero de pixels "vermelhos"
        red_pixels = redness[mask]
        red_pixel_count = np.sum(red_pixels > redness_threshold)
        
        # area total vermelha
        total_red_area += red_pixel_count
        
        # colocando cores pros labels
        output[mask] = np.random.randint(0, 255, 3).tolist()

    # proporcao de vermelho na area do olho
    redness_ratio = total_red_area / total_eye_area

    # resultado
    # conclue se é vermelho de acordo com o limite determinado
    is_red_eye = (redness_ratio > redness_ratio_threshold)

    return output, redness, non_skin_mask, closed_redness, is_red_eye


def main():
    # le a imagem
    img = cv2.imread("../crianca.png")
    # algoritmo de detecções utilizados no código
    haar_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
    # verificando se imagem foi carregada
    if img is None:
        print("Falha ao carregar imagem.")
        return
    
    # detectando face
    faces = detect_faces(img, haar_cascade)
    
    # tratando caso nenhuma face tenha sido detectada
    if faces is None:
        print("Nenhuma face detectada")
        return

    # loop principal
    for face in faces:
        # indentificando olhos para cada face indentificada
        eyes = detect_eyes(face, eye_cascade)
        for eye_coords in eyes:
            if eye_coords is not None:
                (x, y, w, h) = eye_coords
                eye_img = face[y:y + h, x:x + w]
                eye_img = cut_eyebrows(eye_img)
                output, redness, non_skin_mask, closed_redness, is_red_eye = detect_red_eye(eye_img)
                cv2.imshow('Original Image', eye_img)
                cv2.imshow('Redness Detection', redness)
                cv2.imshow('Non-Skin Mask', non_skin_mask)
                cv2.imshow('Closed Redness', closed_redness)
                cv2.imshow('Labeled Red-Eye Regions', output)
                print(is_red_eye)
                cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
