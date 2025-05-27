import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset_path = os.path.join(os.getcwd(), 'cats_and_dogs_filtered')

train_dir = os.path.join(dataset_path, 'train')
validation_dir = os.path.join(dataset_path, 'validation')

batch_size = 32
img_height = 64
img_width = 64

all_images_dir = os.path.join(dataset_path, 'all_data_temp')

if not os.path.exists(all_images_dir):
    os.makedirs(all_images_dir)
    for folder in ['cats', 'dogs']:
        os.makedirs(os.path.join(all_images_dir, folder), exist_ok=True)
        for file in os.listdir(os.path.join(train_dir, folder)):
            src = os.path.join(train_dir, folder, file)
            dst = os.path.join(all_images_dir, folder, file)
            if not os.path.exists(dst):
                os.link(src, dst)
        for file in os.listdir(os.path.join(validation_dir, folder)):
            src = os.path.join(validation_dir, folder, file)
            dst = os.path.join(all_images_dir, folder, file)
            if not os.path.exists(dst):
                os.link(src, dst)

full_ds = image_dataset_from_directory(
    all_images_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
    shuffle=True,
    seed=42
)

def dataset_to_numpy(ds):
    images = []
    labels = []
    for batch_images, batch_labels in ds:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images), np.concatenate(labels)

x_all, y_all = dataset_to_numpy(full_ds)

x_all = x_all / 255.0

x_train, x_test, y_train, y_test = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

class_names = ['gato', 'cachorro']

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),  # Atualizado
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=13,
    batch_size=batch_size,
    validation_split=0.1,
    shuffle=True
)

y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=class_names))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Acurácia no conjunto de teste: {test_acc:.4f}')

plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.title('Desempenho da CNN (Gato vs Cachorro)')
plt.show()

diretorio = os.path.join(os.getcwd(), 'imagens')

if not os.path.exists(diretorio):
    print(f"Pasta '{diretorio}' não encontrada. Por favor, crie a pasta e coloque imagens .jpg ou .jpeg para inferência.")
else:
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg')):
            caminho_imagem = os.path.join(diretorio, nome_arquivo)

            img = tf.keras.utils.load_img(caminho_imagem, target_size=(img_height, img_width))  # Atualizado
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            classe = np.argmax(pred, axis=1)[0]
            classe_nome = class_names[classe] if classe < len(class_names) else f'Classe {classe}'

            img_cv = cv2.imread(caminho_imagem)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_cv, (128, 128))
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            gaussian = cv2.GaussianBlur(resized_rgb, (5, 5), 0)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)

            plt.figure(figsize=(10, 4))
            plt.suptitle(f'Classe prevista: {classe_nome}', fontsize=14)

            plt.subplot(1, 4, 1)
            plt.imshow(img_rgb)
            plt.title("Original")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(resized_rgb)
            plt.title("128x128")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(gaussian)
            plt.title("Gaussiano")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(equalized, cmap='gray')
            plt.title("Equalizado")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

cv2.destroyAllWindows()
