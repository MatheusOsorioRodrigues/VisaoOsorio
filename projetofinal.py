import cv2
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "torch"  # Mantém PyTorch como backend
from keras.models import load_model

# ================= CONFIGURAÇÕES =================
os.environ["KERAS_BACKEND"] = "torch"  # Usando PyTorch como backend
MODEL_PATH = 'modelo/modelo_moedas.keras'  # Caminho do modelo
CLASSES = ["1 real", "25 cent", "50 cent"]  # Ordem deve bater com o treino
CONF_THRESHOLD = 0.85  # Aumentado para evitar falsos positivos
MIN_COIN_AREA = 1500   # Área mínima em pixels para considerar uma moeda

# ================= FUNÇÕES =================
def preProcess(img):
    """Pré-processamento aprimorado para moedas"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    edges = cv2.Canny(blurred, 50, 150)  # Thresholds ajustáveis
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def detectar_moeda(img, model):
    """Classificação com verificações de confiança"""
    img_processed = cv2.resize(img, (224, 224))
    img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    img_processed = (img_processed.astype(np.float32) / 127.0) - 1
    
    predictions = model.predict(np.expand_dims(img_processed, axis=0))[0]
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    
    # Verifica se a confiança é suficiente e significativamente maior que outras classes
    if (confidence > CONF_THRESHOLD and 
        confidence > np.max(np.delete(predictions, class_idx)) + 0.15):
        return CLASSES[class_idx], confidence
    return "Indeterminado", 0

# ================= PRINCIPAL =================
def main(image_path):
    # Carrega modelo
    model = load_model(MODEL_PATH, compile=False)
    
    # Carrega imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar {image_path}")
        return
    
    # Processamento
    img_pre = preProcess(img)
    contours, _ = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detecção
    qtd = 0
    resultados = []
    output_img = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_COIN_AREA:
            x,y,w,h = cv2.boundingRect(cnt)
            roi = img[y:y+h, x:x+w]
            
            classe, conf = detectar_moeda(roi, model)
            
            # Desenha resultados
            color = (0, 255, 0) if classe != "Indeterminado" else (0, 0, 255)
            cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 2)
            cv2.putText(output_img, f"{classe} ({conf:.0%})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Calcula valores
            if classe == "1 real":
                qtd += 1
            elif classe == "25 cent":
                qtd += 0.25
            elif classe == "50 cent":
                qtd += 0.5
            
            resultados.append((classe, conf, (x,y,w,h)))

    # Exibe resultados
    print("\n=== RESULTADOS ===")
    print(f"Total identificado: R$ {qtd:.2f}")
    print("\nDetalhes:")
    for i, (classe, conf, _) in enumerate(resultados, 1):
        print(f"{i}. {classe} - Confiança: {conf:.1%}")
    
    # Mostra imagem
    cv2.imshow("Moedas Detectadas", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Uso: python deteccao_imagem.py <caminho_da_imagem>")
        # Exemplo de teste automático (opcional)
        #TEST_IMG = "moedas1.jpg"
        #TEST_IMG = "moedas2.jpg"
        TEST_IMG = "moedas3.jpg"
        #TEST_IMG = "moedas4.png"
        #TEST_IMG = "moedas5.jpg"
        if os.path.exists(TEST_IMG):
            print(f"\nExecutando teste com {TEST_IMG}...")
            main(TEST_IMG)