import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import keras
from keras import layers

# ---------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ---------------------------------------------------------------
# Скачай датасет перед запуском (выполни в терминале):
#
# Windows PowerShell:
#   Invoke-WebRequest -Uri "https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip" -OutFile "captcha_images_v2.zip"
#   Expand-Archive captcha_images_v2.zip -DestinationPath .
#
# Mac / Linux:
#   curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
#   unzip captcha_images_v2.zip

data_dir = Path("./captcha_images_v2/")

images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = sorted(list(set(char for label in labels for char in label)))

print("Количество изображений: ", len(images))
print("Количество меток:       ", len(labels))
print("Уникальных символов:    ", len(characters))
print("Символы:                ", characters)

# ---------------------------------------------------------------
# 2. ПАРАМЕТРЫ
# ---------------------------------------------------------------
batch_size = 16
img_width  = 200
img_height = 50
max_length = max(len(label) for label in labels)

# ---------------------------------------------------------------
# 3. КОДИРОВЩИКИ СИМВОЛОВ
# ---------------------------------------------------------------
# StringLookup создаёт словарь:
#   индекс 0  →  [UNK]   ← используется как CTC blank символ
#   индекс 1  →  '2'
#   индекс 2  →  '3'
#   ...
#   индекс 19 →  'y'
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
vocab_size = len(char_to_num.get_vocabulary())  # 20 (включая [UNK])

# ---------------------------------------------------------------
# 4. РАЗДЕЛЕНИЕ ДАННЫХ
# ---------------------------------------------------------------
def split_data(images, labels, train_size=0.9, shuffle=True):
    size    = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    n_train = int(size * train_size)
    x_train = images[indices[:n_train]]
    y_train = labels[indices[:n_train]]
    x_valid = images[indices[n_train:]]
    y_valid = labels[indices[n_train:]]
    return x_train, x_valid, y_train, y_valid

x_train, x_valid, y_train, y_valid = split_data(
    np.array(images), np.array(labels)
)

# ---------------------------------------------------------------
# 5. ПРЕДОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ
# ---------------------------------------------------------------
def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)                    # grayscale
    img = tf.image.convert_image_dtype(img, tf.float32)         # [0,255]→[0,1]
    img = tf.image.resize(img, [img_height, img_width])         # нужный размер
    img = tf.transpose(img, perm=[1, 0, 2])                     # [H,W,C]→[W,H,C]
    label = char_to_num(
        tf.strings.unicode_split(label, input_encoding="UTF-8")
    )
    return {"image": img, "label": label}

# ---------------------------------------------------------------
# 6. tf.data.Dataset
# ---------------------------------------------------------------
def make_dataset(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

train_dataset      = make_dataset(x_train, y_train)
validation_dataset = make_dataset(x_valid, y_valid)

# Для обучения добавляем фиктивный y=zeros
# (настоящий loss уже вычислен внутри Lambda слоя модели)
def add_dummy_y(batch):
    return batch, tf.zeros([tf.shape(batch["image"])[0], 1])

train_dataset_ctc      = train_dataset.map(add_dummy_y)
validation_dataset_ctc = validation_dataset.map(add_dummy_y)

# ---------------------------------------------------------------
# 7. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ
# ---------------------------------------------------------------
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    imgs = batch["image"]
    lbls = batch["label"]
    for i in range(16):
        img   = (imgs[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(
            num_to_char(lbls[i])
        ).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=100)
plt.close()
print("Сохранено: sample_images.png")

# ---------------------------------------------------------------
# 8. ПОСТРОЕНИЕ МОДЕЛИ
# ---------------------------------------------------------------
def build_model():
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels_in = layers.Input(name="label", shape=(None,), dtype="float32")

    # --- CNN блок 1 ---
    x = layers.Conv2D(
        32, (3, 3), activation="relu",
        kernel_initializer="he_normal", padding="same", name="Conv1"
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)    # 200x50 → 100x25

    # --- CNN блок 2 ---
    x = layers.Conv2D(
        64, (3, 3), activation="relu",
        kernel_initializer="he_normal", padding="same", name="Conv2"
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)    # 100x25 → 50x12

    # --- CNN → RNN переход ---
    # Reshape: (50, 12, 64) → (50, 768)
    # 50 — временные шаги, 768 — признаки на каждом шаге
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # --- RNN ---
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # --- Выходной слой ---
    # ВАЖНО: activation=None (линейный выход = сырые логиты)
    # tf.nn.ctc_loss ожидает логиты, а НЕ вероятности (softmax)!
    # vocab_size=20: индекс 0=[UNK]=blank, индексы 1..19=символы
    x = layers.Dense(vocab_size, activation=None, name="dense2")(x)

    # --- Prediction model: только изображение → логиты ---
    prediction_model = keras.models.Model(
        inputs=input_img, outputs=x, name="prediction_model"
    )

    # --- CTC Loss через Lambda слой ---
    def ctc_loss_func(args):
        y_true, y_pred = args
        batch_len     = tf.shape(y_pred)[0]
        input_len     = tf.shape(y_pred)[1]
        label_len     = tf.shape(y_true)[1]
        input_lengths = tf.fill([batch_len], input_len)
        label_lengths = tf.fill([batch_len], label_len)
        loss = tf.nn.ctc_loss(
            labels            = tf.cast(y_true, dtype=tf.int32),
            logits            = y_pred,    # сырые логиты — правильно!
            label_length      = label_lengths,
            logit_length      = input_lengths,
            logits_time_major = False,
            blank_index       = 0,         # индекс 0 = [UNK] = CTC blank
        )
        return tf.reshape(loss, (-1, 1))

    ctc_out = layers.Lambda(ctc_loss_func, name="ctc_loss")([labels_in, x])

    # --- Training model: [изображение + метка] → loss ---
    training_model = keras.models.Model(
        inputs  = [input_img, labels_in],
        outputs = ctc_out,
        name    = "ocr_model_v1",
    )
    # loss lambda: y_pred уже и есть ctc loss, фиктивный y_true игнорируется
    training_model.compile(
        optimizer = keras.optimizers.Adam(),
        loss      = lambda y_true, y_pred: y_pred,
    )
    return training_model, prediction_model


model, prediction_model = build_model()
model.summary()

# ---------------------------------------------------------------
# 9. ОБУЧЕНИЕ
# ---------------------------------------------------------------
epochs                  = 100
early_stopping_patience = 10

early_stopping = keras.callbacks.EarlyStopping(
    monitor              = "val_loss",
    patience             = early_stopping_patience,
    restore_best_weights = True,
)

history = model.fit(
    train_dataset_ctc,
    validation_data = validation_dataset_ctc,
    epochs          = epochs,
    callbacks       = [early_stopping],
)

# ---------------------------------------------------------------
# 10. ДЕКОДИРОВАНИЕ ПРЕДСКАЗАНИЙ
# ---------------------------------------------------------------
def decode_batch_predictions(logits_batch):
    """
    Принимает сырые логиты (batch, time, vocab).
    1. log_softmax → лог-вероятности
    2. Транспонирует в (time, batch, vocab) для ctc_greedy_decoder
    3. Декодирует greedy методом
    4. Фильтрует blank (индекс 0) и конвертирует в текст
    """
    log_probs   = tf.nn.log_softmax(logits_batch, axis=-1)
    log_probs_t = tf.transpose(log_probs, perm=[1, 0, 2])
    input_len   = tf.fill([tf.shape(logits_batch)[0]], tf.shape(logits_batch)[1])

    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs          = log_probs_t,
        sequence_length = tf.cast(input_len, tf.int32),
    )
    # sparse → dense, blank позиции заполняются нулями
    dense = tf.sparse.to_dense(decoded[0], default_value=0)

    output_text = []
    for row in dense.numpy():
        row  = row[row > 0]       # фильтруем blank (индекс 0)
        row  = row[:max_length]   # ограничиваем длину
        text = tf.strings.reduce_join(
            num_to_char(row)
        ).numpy().decode("utf-8")
        output_text.append(text)
    return output_text

# ---------------------------------------------------------------
# 11. ПРОВЕРКА НА ВАЛИДАЦИОННЫХ ДАННЫХ
# ---------------------------------------------------------------
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    logits     = prediction_model.predict(batch_images, verbose=0)
    pred_texts = decode_batch_predictions(logits)

    orig_texts = []
    for lbl in batch_labels:
        text = tf.strings.reduce_join(
            num_to_char(lbl)
        ).numpy().decode("utf-8")
        orig_texts.append(text)

    # Сохраняем картинку с результатами
    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(min(16, len(pred_texts))):
        img   = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img   = img.T
        orig  = orig_texts[i] if i < len(orig_texts) else ""
        pred  = pred_texts[i]
        color = "green" if orig == pred else "red"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(
            f"Верно: {orig}\nПред:  {pred}", fontsize=8, color=color
        )
        ax[i // 4, i % 4].axis("off")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=100)
    plt.close()
    print("Сохранено: predictions.png")

    correct = sum(1 for p, o in zip(pred_texts, orig_texts) if p == o)
    total   = len(pred_texts)
    print(f"\nТочность на батче: {correct}/{total} = {correct / total * 100:.1f}%")
    print("-" * 40)
    for orig, pred in zip(orig_texts, pred_texts):
        mark = "✓" if orig == pred else "✗"
        print(f"  {mark}  Верно: {orig:<8}  Предсказано: {pred}")