import cv2
import numpy as np

def processar_frame(frame):
    # Converter para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar um filtro Gaussiano para reduzir ruído
    suavizado = cv2.GaussianBlur(cinza, (5, 5), 0)
    
    # Aplicar detecção de bordas Canny
    bordas = cv2.Canny(suavizado, 50, 150)
    
    # Definir a região de interesse (ROI)
    altura, largura = bordas.shape
    mascara = np.zeros_like(bordas)
    regioes = np.array([[(100, altura), (largura-100, altura), (largura//2, altura//2)]], dtype=np.int32)
    cv2.fillPoly(mascara, regioes, 255)
    roi = cv2.bitwise_and(bordas, mascara)
    
    # Detectar linhas com a Transformada de Hough
    linhas = cv2.HoughLinesP(roi, 1, np.pi/180, 100, minLineLength=150, maxLineGap=50)
    
    # Criar lista para armazenar linhas filtradas
    linhas_filtradas = []
    
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            inclinacao = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 20 < abs(inclinacao) < 160:  # Filtrar linhas muito horizontais
                linhas_filtradas.append((x1, y1, x2, y2))
    
    # Criar imagem com as linhas filtradas
    linha_img = np.zeros_like(frame)
    for x1, y1, x2, y2 in linhas_filtradas:
        cv2.line(linha_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combinar imagem original com as linhas detectadas
    resultado = cv2.addWeighted(frame, 0.8, linha_img, 1, 0)
    return resultado

# Capturar vídeo da câmara ou de um ficheiro
cap = cv2.VideoCapture('video-para-teste1.mp4')  # Substituir pelo caminho do vídeo

# Criar um writer para guardar o vídeo processado
fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('saida.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_processado = processar_frame(frame)
    saida.write(frame_processado)
    
    cv2.imshow('Detecção de Faixas', frame_processado)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
saida.release()
cv2.destroyAllWindows()
