import os
os.environ["KERAS_BACKEND"] = "torch"  # Usa PyTorch como backend
import keras
from keras import layers, callbacks
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Configurações
IMG_SIZE = (224, 224)  # Mantém igual ao seu código original
BATCH_SIZE = 32
EPOCHS = 20

# Classes correspondentes às subpastas
CLASSES = ["1 real", "25 cent", "50 cent"]

def load_data_from_subfolders(data_dir):
    images = []
    labels = []
    
    # Mapeamento das subpastas para índices de classe
    subfolder_to_class = {
        "1": 0,   # "1 real"
        "25": 1,  # "25 cent"
        "50": 2   # "50 cent"
    }
    
    print("Carregando imagens das subpastas...")
    
    for subfolder, class_idx in subfolder_to_class.items():
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"AVISO: Subpasta '{subfolder}' não encontrada!")
            continue
            
        # Processa TODAS imagens da subpasta (1.jpg, 2.jpg, ..., fl1.jpg, etc.)
        for img_file in os.listdir(subfolder_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                img_path = os.path.join(subfolder_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Erro ao ler: {img_path}")
                    continue
                    
                # Pré-processamento idêntico ao seu código original
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img.astype(np.float32) / 127.0) - 1
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Erro no arquivo {img_file}: {str(e)}")
    
    return np.array(images), np.array(labels)

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        
        # Blocos convolucionais (ajustados para moedas)
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Verificação inicial da estrutura
    DATA_DIR = "imagens"
    print("\nVerificando estrutura de pastas...")
    print(f"Diretório atual: {os.getcwd()}")
    print(f"Conteúdo de '{DATA_DIR}': {os.listdir(DATA_DIR)}")
    
    # Carrega dados
    X, y = load_data_from_subfolders(DATA_DIR)
    
    if len(X) == 0:
        print("\nERRO: Nenhuma imagem válida encontrada!")
        print("Verifique se:")
        print(f"1. Existe a pasta '{DATA_DIR}' com subpastas '1', '25', '50'")
        print("2. Cada subpasta contém imagens .jpg/.png (ex: '1.jpg', 'fl25.jpg')")
        print("3. O script está na mesma pasta que a pasta 'imagens'")
        exit()
    
    print(f"\nDados carregados com sucesso!")
    print(f"Total de imagens: {len(X)}")
    print(f"Distribuição por classe: {np.bincount(y)}")
    print(f"Classes: {CLASSES}")
    
    # Divide os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Constroi e treina o modelo
    model = build_model()
    model.summary()
    
    print("\nIniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Avaliação final
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia no conjunto de teste: {test_acc:.2%}")
    
    # Salva o modelo
    model.save("modelo_moedas.keras")
    print("Modelo salvo como 'modelo_moedas.keras'")