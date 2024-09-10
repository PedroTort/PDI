import cv2
import numpy as np
import os

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
    redness = (R**2)/(G**2 + B **2 + 0.0001)

    # Aplica o filtro: pixels com vermelhidão > 3 são definidos como 255, caso contrário, 0
    redness = np.where(redness > 3, 255, 0).astype(np.uint8)
    
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

def shape_filtering(labels, stats, face_w, face_h):
    # Definindo os parâmetros
    Kwmin, Khmin = 1/50, 1/50
    Kwmax, Khmax = 1/12, 1/12
    
    filtered_labels = np.zeros(labels.shape, dtype=np.uint8)
    
    # Itera sobre os componentes conectados (ignora o rótulo 0, que é o fundo)
    for i in range(1, len(stats)):
        wi = stats[i, cv2.CC_STAT_WIDTH]
        hi = stats[i, cv2.CC_STAT_HEIGHT]
        ai = stats[i, cv2.CC_STAT_AREA]

        # Verifica largura e altura em relação ao rosto
        if not (Kwmin * face_w <= wi <= Kwmax * face_w and
                Khmin * face_h <= hi <= Khmax * face_h):
            continue
        
        # Verifica razão largura/altura
        if not (0.5 <= wi / hi <= 2):
            continue
        
        # Verifica a compactação
        if ai < (wi * hi) / 2:
            continue
        
        # Se passou por todos os filtros, mantém o componente
        filtered_labels[labels == i] = 255
    
    return filtered_labels

def detect_red_eye(image, face):
    
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
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed_redness, connectivity=4)
    filtered_redness = shape_filtering(labels, stats, face.shape[1], face.shape[0])

    return filtered_redness, redness, non_skin_mask, closed_redness

def main():
    # algoritmo de detecções utilizados no código
    haar_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    
    folder_path = '../figures'
    files = os.listdir(folder_path)
    png_files = [file for file in files if file.lower().endswith('.png')]
    sorted_files = sorted(png_files)
    number_of_eyes_detected = []
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)
        img = cv2.imread(file_path)
        # img = cv2.imread('../figures/12.png')
        # verificando se imagem foi carregada
        if img is None:
            print("Falha ao carregar imagem.")
            return
        
        # detectando face
        faces = detect_faces(img, haar_cascade)
        face_w_redeyes_list = []
        red_eye_highlight_list = []
        
        # tratando caso nenhuma face tenha sido detectada
        if faces is None:
            print("Nenhuma face detectada")
            return

        # loop principal
        for face in faces:
            # indentificando olhos para cada face indentificada
            eyes = detect_eyes(face, eye_cascade)
            red_eye_highlight = np.zeros_like(face)
            face_w_redeyes = face.copy()
            for eye_coords in eyes:
                if eye_coords is not None:
                    (x, y, w, h) = eye_coords
                    eye_img1 = face[y:y + h, x:x + w]
                    eye_img = cut_eyebrows(eye_img1)
                    number_of_eyes_detected.append(eye_img)
                    y = y + (eye_img1.shape[0] - eye_img.shape[0])
                    output, redness, non_skin_mask, closed_redness = detect_red_eye(eye_img, face)

                    red_eyes = cv2.merge((np.zeros_like(output), np.zeros_like(output), output))

                    non_black_mask = np.any(red_eyes != [0, 0, 0], axis=-1)

                    red_eye_highlight[y:y + eye_img.shape[0], x:x + w] = red_eyes
                    face_w_redeyes[y:y + eye_img.shape[0], x:x + w][non_black_mask] = red_eyes[non_black_mask]

            face_w_redeyes_list.append(face_w_redeyes)
            red_eye_highlight_list.append(red_eye_highlight)
        cv2.destroyAllWindows()
        for i,face in enumerate(faces):
            # mostrando face detectada 
            cv2.imshow(f'Original Image {i}', face) 

            # mostrando face com olhos vermlhos
            if i < len(face_w_redeyes_list):
                cv2.imshow(f'Image with Red Eyes {i}', face_w_redeyes_list[i]) 

            # mostrando olhos vermelhos detectados
            if i < len(red_eye_highlight_list):
                cv2.imshow(f'Red Eyes Detected {i}', red_eye_highlight_list[i]) 

            cv2.waitKey(0)
            cv2.destroyAllWindows()  # Close all windows after each image is displayed
        cv2.destroyAllWindows()

    print(len(number_of_eyes_detected))
if __name__ == "__main__":
    main()