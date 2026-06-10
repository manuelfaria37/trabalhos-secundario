import cv2
import numpy as np
from ultralytics import YOLO
import os
from sklearn.cluster import KMeans
from collections import deque

# Carregar o modelo YOLOv8n pré-treinado
model = YOLO("yolov8n.pt")

# Definir as classes de interesse (baseado no COCO)
classes_of_interest = {
    "car": "Automovel",
    "bus": "Autocarro",
    "motorcycle": "Mota"
}
class_ids = {
    "car": 2,        # ID da classe 'car' no COCO
    "bus": 5,        # ID da classe 'bus' no COCO
    "motorcycle": 3  # ID da classe 'motorcycle' no COCO
}
colors = {
    "car": (0, 255, 0),        # Verde para automóveis
    "bus": (255, 0, 255),      # Magenta para autocarros
    "motorcycle": (0, 255, 255) # Ciano para motas
}

# Configurações
conf_thres = 0.25  # Limiar de confiança geral
motorcycle_conf_thres = 0.35  # Limiar de confiança específico para motociclos
iou_thres = 0.5    # Limiar de IoU para NMS
min_area = 50      # Reduzido para capturar motociclos menores

# Ajustes para evitar classificação incorreta
car_aspect_ratio_range = (1.0, 3.0)      # Proporção largura/altura típica de carros
bus_aspect_ratio_range = (0.5, 1.5)      # Proporção largura/altura típica de autocarros
motorcycle_aspect_ratio_range = (0.3, 1.0)  # Proporção mais estreita para motociclos
bus_conf_thres = 0.4  # Limiar de confiança mais alto para autocarros

# Configurações para estimativa de velocidade
pixels_per_meter = 50  # Aproximação: 50 pixels por metro (ajustar com base na sua câmera)
fps = 30  # Taxa de quadros do vídeo (ajustar conforme o vídeo)
prev_positions = {}  # Armazenar posições anteriores para rastreamento (formato: {id: ((center_x, center_y), frame_idx)})
speed_history = {}   # Armazenar histórico de velocidades para suavização
speed_window = 5     # Janela para média móvel de velocidades
object_ids = {}      # Atribuir IDs únicos aos objetos
next_id = 0  # Contador para IDs de objetos

# Definir intervalos de cores no espaço HSV (ajustados e ampliados)
color_ranges = {
    "vermelho": (np.array([0, 70, 70]), np.array([15, 255, 255])),
    "vermelho2": (np.array([160, 70, 70]), np.array([180, 255, 255])),
    "azul": (np.array([90, 70, 70]), np.array([140, 255, 255])),
    "verde": (np.array([35, 70, 70]), np.array([85, 255, 255])),
    "amarelo": (np.array([15, 70, 70]), np.array([35, 255, 255])),
    "branco": (np.array([0, 0, 200]), np.array([180, 30, 255])),
    "preto": (np.array([0, 0, 0]), np.array([180, 255, 50])),
    "cinza": (np.array([0, 0, 50]), np.array([180, 30, 200]))
}

def detect_color(frame, box):
    """Detecta a cor dominante dentro da caixa delimitadora usando k-means com pré-processamento."""
    x1, y1, x2, y2 = map(int, box[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return "desconhecido"

    # Extrair a região dentro da caixa
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "desconhecido"

    # Pré-processamento: equalizar o canal de valor (V) no HSV para melhorar a iluminação
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge((h, s, v))

    # Usar k-means para encontrar a cor dominante
    pixels = hsv.reshape(-1, 3)
    mask = (pixels[:, 1] > 30) & (pixels[:, 2] > 30) & (pixels[:, 2] < 230)
    pixels = pixels[mask]
    if len(pixels) < 20:  # Aumentado o mínimo de pixels para análise
        return "desconhecido"

    # Aumentar o número de clusters para capturar mais variações
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    dominant_hue = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))][0]

    # Verificar a cor dominante com base nos intervalos ajustados
    for color_name, (lower, upper) in color_ranges.items():
        if color_name == "vermelho2":
            if dominant_hue >= 160:
                return "vermelho"
        elif dominant_hue >= lower[0] and dominant_hue <= upper[0]:
            return color_name

    return "desconhecido"

def calculate_iou(box1, box2):
    """Calcula o IoU entre duas caixas delimitadoras."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
    return iou

def soft_nms(detections, iou_thres):
    """Aplica Soft-NMS para ajustar pontuações de caixas sobrepostas."""
    if len(detections) == 0:
        return detections
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []
    while detections:
        max_det = detections[0]
        keep.append(max_det)
        remaining = []
        for det in detections[1:]:
            iou = calculate_iou(max_det[:4], det[:4])
            if iou >= iou_thres:
                new_det = list(det)
                new_det[4] *= (1 - iou)
                if new_det[4] > conf_thres:
                    remaining.append(new_det)
            else:
                remaining.append(det)
        detections = remaining
    return keep

def process_frame(frame, frame_idx):
    """Processa um quadro e retorna o quadro com caixas delimitadoras desenhadas, incluindo velocidade."""
    global next_id, prev_positions, object_ids, speed_history

    if frame is None or frame.size == 0:
        print("Erro: Quadro inválido ou vazio.")
        return frame

    # Pré-processamento
    img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

    # Inferência com YOLOv8
    results = model(img, conf=conf_thres, iou=iou_thres)

    # Extrair detecções
    current_detections = {}
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas (x1, y1, x2, y2)
        scores = result.boxes.conf.cpu().numpy()  # Confianças
        classes = result.boxes.cls.cpu().numpy()  # Classes

        for box, score, cls in zip(boxes, scores, classes):
            cls = int(cls)
            for class_name, class_id in class_ids.items():
                if cls == class_id:
                    # Ajustar limiar de confiança para motociclos
                    if class_name == "motorcycle" and score < motorcycle_conf_thres:
                        continue

                    x1, y1, x2, y2 = box
                    # Escalar coordenadas de volta para o tamanho original do quadro
                    h_orig, w_orig = frame.shape[:2]
                    x1 = int(x1 * w_orig / 640)
                    y1 = int(y1 * h_orig / 640)
                    x2 = int(x2 * w_orig / 640)
                    y2 = int(y2 * h_orig / 640)
                    area = (x2 - x1) * (y2 - y1)
                    if area < min_area:
                        continue

                    # Calcular proporção largura/altura
                    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else float('inf')

                    # Ajustar classificação com base na proporção
                    if class_name == "bus":
                        if not (bus_aspect_ratio_range[0] <= aspect_ratio <= bus_aspect_ratio_range[1]) or score < bus_conf_thres:
                            class_name = "car"  # Reclassificar como carro
                    elif class_name == "motorcycle":
                        if not (motorcycle_aspect_ratio_range[0] <= aspect_ratio <= motorcycle_aspect_ratio_range[1]):
                            class_name = "car"  # Reclassificar como carro se a proporção não for típica de moto

                    # Detectar cor
                    color_detected = detect_color(frame, [x1, y1, x2, y2])

                    # Calcular centro da caixa para rastreamento
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Tentar associar a detecção a um objeto existente
                    matched_id = None
                    for obj_id, (prev_center, _) in prev_positions.items():
                        prev_x, prev_y = prev_center
                        dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        if dist < 100:  # Aumentado o limiar para melhorar o rastreamento
                            matched_id = obj_id
                            break

                    if matched_id is None:
                        matched_id = next_id
                        next_id += 1
                        speed_history[matched_id] = deque(maxlen=speed_window)  # Inicializar histórico de velocidades

                    # Atualizar ou inicializar a posição e velocidade
                    if matched_id in prev_positions:
                        prev_center, prev_frame = prev_positions[matched_id]
                        delta_pixels = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                        frames_elapsed = frame_idx - prev_frame
                        if frames_elapsed > 0:
                            speed_pixels_per_frame = delta_pixels / frames_elapsed
                            speed_m_per_s = speed_pixels_per_frame / pixels_per_meter * fps
                            speed_km_per_h = speed_m_per_s * 3.6
                            # Filtrar velocidades improváveis
                            if speed_km_per_h > 200 or speed_km_per_h < 0:
                                speed_km_per_h = 0
                            # Suavizar a velocidade com média móvel
                            speed_history[matched_id].append(speed_km_per_h)
                            speed_km_per_h = np.mean(speed_history[matched_id]) if speed_history[matched_id] else 0
                        else:
                            speed_km_per_h = 0
                    else:
                        speed_km_per_h = 0

                    # Armazenar a nova posição
                    prev_positions[matched_id] = ((center_x, center_y), frame_idx)
                    object_ids[(x1, y1, x2, y2)] = matched_id

                    current_detections[matched_id] = [x1, y1, x2, y2, score, class_name, color_detected, speed_km_per_h]

    # Aplicar Soft-NMS
    detections_to_nms = [[d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]] for d in current_detections.values()]
    kept_detections = soft_nms(detections_to_nms, iou_thres)

    # Filtrar detecções mantidas após NMS
    final_detections = {k: v for k, v in current_detections.items() if any(
        (v[0] == kd[0] and v[1] == kd[1] and v[2] == kd[2] and v[3] == kd[3]) for kd in kept_detections)}

    # Desenhar caixas no quadro original
    for obj_id, det in final_detections.items():
        x1, y1, x2, y2, score, class_name, color_detected, speed = det
        color = colors[class_name]
        label = f"ID {obj_id}: {classes_of_interest[class_name]} ({color_detected}) {score:.2f} {speed:.1f} km/h"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Atualizar prev_positions para conter apenas os objetos ainda visíveis
    new_prev_positions = {}
    for obj_id, det in final_detections.items():
        center_x = (det[0] + det[2]) / 2
        center_y = (det[1] + det[3]) / 2
        new_prev_positions[obj_id] = ((center_x, center_y), frame_idx)
    prev_positions.clear()
    prev_positions.update(new_prev_positions)

    return frame

def process_video(input_path, output_path):
    """Processa um vídeo, exibe em tempo real e salva o resultado com caixas delimitadoras."""
    if not os.path.exists(input_path):
        print(f"Erro: O arquivo {input_path} não foi encontrado. Verifique o caminho.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo. Tente instalar o K-Lite Codec Pack ou verifique o caminho do arquivo.")
        return

    global fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_path = os.path.splitext(output_path)[0] + ".avi"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Erro ao criar o vídeo de saída. Verifique se o codec XVID está instalado ou se você tem permissões de escrita.")
        cap.release()
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, frame_idx)
        cv2.imshow("Vehicle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Vídeo processado salvo em: {output_path}")

# Exemplo de uso
if __name__ == "__main__":
    input_video = r"E:\Física\Python\Desafio14\IMG_0735.mp4"  # Caminho ajustado
    output_video = r"E:\Física\Python\Desafio14\output_video.avi"
    process_video(input_video, output_video)
