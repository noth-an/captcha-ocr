import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from captcha.image import ImageCaptcha
from tensorflow import keras

# Символы, которые будут использоваться в CAPTCHA
# Исключаем похожие символы: 0/O, 1/l/I — чтобы упростить задачу
CHARACTERS = "23456789abcdefghjkmnpqrstuvwxyz"

# Длина строки CAPTCHA (количество символов)
CAPTCHA_LENGTH = 5

# Размер изображения
IMG_WIDTH  = 200
IMG_HEIGHT = 50

# Количество генерируемых изображений
NUM_SAMPLES = 1000

# Пути к папкам
DATA_DIR    = Path("data/samples")
OUTPUT_DIR  = Path("outputs")

def generate_captcha_dataset(num_samples=NUM_SAMPLES, save_dir=DATA_DIR):

    # Создаём папку если не существует
    save_dir.mkdir(parents=True, exist_ok=True)

    # Инициализируем генератор CAPTCHA
    # Указываем размер изображения
    generator = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT)

    print(f"Генерация {num_samples} изображений CAPTCHA...")

    for i in range(num_samples):
        # Генерируем случайный текст из нашего алфавита
        captcha_text = "".join(
            np.random.choice(list(CHARACTERS), size=CAPTCHA_LENGTH)
        )

        # Генерируем изображение
        image_path = save_dir / f"{captcha_text}.png"
        generator.write(captcha_text, str(image_path))

        # Прогресс каждые 100 изображений
        if (i + 1) % 100 == 0:
            print(f"  Создано: {i + 1}/{num_samples}")

    print(f"Готово! Изображения сохранены в: {save_dir}")

def load_dataset(data_dir=DATA_DIR):

    # Получаем все PNG файлы из папки
    image_paths = sorted(list(data_dir.glob("*.png")))

    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"Изображения не найдены в {data_dir}. "
            f"Запустите generate_captcha_dataset() сначала."
        )

    # Извлекаем метки из имён файлов
    # Файл 'ab3k9.png' → метка 'ab3k9'
    labels = [img.stem for img in image_paths]

    print(f"Загружено {len(image_paths)} изображений")
    print(f"Пример: {image_paths[0].name} → '{labels[0]}'")

    return image_paths, labels


def encode_label(label, char_to_idx):
    return np.array([char_to_idx[ch] for ch in label])


def preprocess_image(image_path):

    # Загружаем как grayscale (1 канал вместо 3)
    img = keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    # Преобразуем в массив
    img = keras.utils.img_to_array(img)

    # Нормализация: делим на 255
    img = img / 255.0

    # Транспонируем оси (высота, ширина, 1) → (ширина, высота, 1)
    # Теперь каждая "строка" массива = вертикальная колонка изображения
    img = np.transpose(img, axes=(1, 0, 2))

    return img

def prepare_dataset(data_dir=DATA_DIR):
    """
    Полный пайплайн подготовки данных:
    загрузка → кодирование → предобработка → разбивка

    Returns:
        x_train, x_val: изображения для обучения и валидации
        y_train, y_val: метки для обучения и валидации
        char_to_idx: словарь символ → индекс
        idx_to_char: словарь индекс → символ
    """
    from sklearn.model_selection import train_test_split

    # Создаём словари для перевода символов ↔ числа
    char_to_idx = {ch: idx for idx, ch in enumerate(CHARACTERS)}
    idx_to_char = {idx: ch for idx, ch in enumerate(CHARACTERS)}

    # Загружаем пути и метки
    image_paths, labels = load_dataset(data_dir)

    # Предобрабатываем все изображения
    print("Предобработка изображений...")
    images = np.array([preprocess_image(p) for p in image_paths])

    # Кодируем все метки в числа
    encoded_labels = np.array([encode_label(lb, char_to_idx) for lb in labels])

    # Разбиваем на обучающую (80%) и валидационную (20%) выборки
    x_train, x_val, y_train, y_val = train_test_split(
        images,
        encoded_labels,
        test_size=0.2,
        random_state=42    # Фиксируем случайность для воспроизводимости
    )

    print(f"Обучающая выборка:    {len(x_train)} изображений")
    print(f"Валидационная выборка: {len(x_val)} изображений")

    return x_train, x_val, y_train, y_val, char_to_idx, idx_to_char


def visualize_samples(data_dir=DATA_DIR, num_samples=8):
    """
    Отображает несколько примеров из датасета.
    Полезно для визуальной проверки что данные выглядят корректно.

    Args:
        data_dir: папка с изображениями
        num_samples: количество примеров для показа
    """
    image_paths, labels = load_dataset(data_dir)

    # Выбираем случайные примеры
    indices = np.random.choice(len(image_paths), num_samples, replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = keras.utils.load_img(image_paths[idx], color_mode="grayscale")
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Метка: '{labels[idx]}'", fontsize=11)
        axes[i].axis("off")

    plt.suptitle("Примеры CAPTCHA из датасета", fontsize=14)
    plt.tight_layout()

    # Сохраняем график
    save_path = OUTPUT_DIR / "plots" / "dataset_samples.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"График сохранён: {save_path}")


if __name__ == "__main__":
    generate_captcha_dataset(num_samples=NUM_SAMPLES)
    visualize_samples()