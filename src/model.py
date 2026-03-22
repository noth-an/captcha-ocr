import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Импортируем константы из нашего dataset.py
from dataset import (
    CHARACTERS,
    CAPTCHA_LENGTH,
    IMG_WIDTH,
    IMG_HEIGHT
)

# Количество уникальных символов в алфавите
NUM_CLASSES = len(CHARACTERS)

class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        # Используем встроенную функцию CTC из Keras
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Определяем размеры
        batch_size   = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        # Приводим ВСЕ значения к одному типу int32
        batch_size   = tf.cast(batch_size,   dtype=tf.int32)
        input_length = tf.cast(input_length, dtype=tf.int32)
        label_length = tf.cast(label_length, dtype=tf.int32)

        input_length = input_length * tf.ones(shape=(batch_size, 1), dtype=tf.int32)
        label_length = label_length * tf.ones(shape=(batch_size, 1), dtype=tf.int32)

        # Метки тоже приводим к int32
        y_true = tf.cast(y_true, dtype=tf.int32)

        # Вычисляем CTC Loss
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def build_model():

    # Вход для изображения
    image_input = layers.Input(
        shape=(IMG_WIDTH, IMG_HEIGHT, 1),
        name="image"
    )

    # Вход для меток (нужен только при обучении для CTC Loss)
    label_input = layers.Input(
        shape=(CAPTCHA_LENGTH,),
        name="label",
        dtype="float32"
    )

    x = image_input

    # Первый свёрточный блок
    # Conv2D: 32 фильтра, ядро 3×3
    # Каждый фильтр учится находить определённый визуальный признак
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv1"
    )(x)
    # MaxPooling уменьшает размер вдвое: 50→25 по высоте
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Второй свёрточный блок
    # 64 фильтра — больше фильтров = более сложные признаки
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="conv2"
    )(x)
    # MaxPooling: 25→12 по высоте
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # После двух пулингов размер: (IMG_WIDTH, 12, 64)
    # Нам нужно: (IMG_WIDTH, 12 * 64) = (200, 768)
    # То есть для каждой из 200 колонок — вектор из 768 признаков

    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    # Уменьшаем размерность для эффективности
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    # Первый двунаправленный LSTM
    # return_sequences=True — возвращаем выход для каждого шага
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25),
        name="bilstm1"
    )(x)

    # Второй двунаправленный LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.25),
        name="bilstm2"
    )(x)

    # Dense слой: для каждой позиции предсказываем вероятность
    # каждого символа из алфавита + 1 специальный символ (blank для CTC)
    x = layers.Dense(
        NUM_CLASSES + 1,
        activation="softmax",
        name="output"
    )(x)

    output = CTCLayer(name="ctc_loss")(label_input, x)

    # Модель для ОБУЧЕНИЯ — принимает изображение И метку
    model = keras.Model(
        inputs=[image_input, label_input],
        outputs=output,
        name="captcha_ocr_model"
    )

    # Модель для ПРЕДСКАЗАНИЙ — принимает только изображение
    # CTC Loss не нужен при инференсе
    prediction_model = keras.Model(
        inputs=image_input,
        outputs=x,
        name="captcha_prediction_model"
    )

    return model, prediction_model

def decode_prediction(pred, idx_to_char):

    # Используем встроенный CTC декодер TensorFlow
    input_len = tf.ones(tf.shape(pred)[0]) * tf.shape(pred)[1]
    input_len = tf.cast(input_len, dtype=tf.int32)

    # Жадное декодирование
    decoded = keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=True
    )[0][0]

    decoded = tf.cast(decoded, dtype=tf.int32).numpy()

    results = []
    for sequence in decoded:
        # Убираем значение -1 (padding после декодирования)
        text = "".join([
            idx_to_char[idx]
            for idx in sequence
            if idx != -1 and idx in idx_to_char
        ])
        results.append(text)

    return results


def print_model_summary(model):
    """Выводит структуру модели в читаемом виде."""
    model.summary(line_length=80)

if __name__ == "__main__":
    model, prediction_model = build_model()

    # Компилируем модель
    # Оптимизатор Adam — стандартный выбор для большинства задач
    model.compile(optimizer=keras.optimizers.Adam())

    print("\n=== Модель для обучения ===")
    print_model_summary(model)

    print("\n=== Модель для предсказаний ===")
    print_model_summary(prediction_model)

    print("\nМодель успешно построена!")