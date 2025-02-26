from rembg import remove
from PIL import Image

def change_stable_diffusion_background(input_image_path, background_image_path, output_image_path):
    """
    Изменяет фон изображения, сгенерированного Stable Diffusion, на новое фоновое изображение.

    Args:
        input_image_path (str): Путь к исходному изображению (сгенерированному Stable Diffusion).
        background_image_path (str): Путь к изображению нового фона.
        output_image_path (str): Путь для сохранения изображения с измененным фоном.
    """
    try:
        # 1. Удаление фона из исходного изображения с помощью rembg
        with open(input_image_path, 'rb') as i:
            output_bytes = remove(i.read())

        foreground_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA") # Открываем как RGBA для прозрачности

        # 2. Загрузка фонового изображения
        background_image = Image.open(background_image_path).convert("RGB") # Убедимся, что фон RGB

        # 3. Изменение размера фонового изображения, чтобы соответствовать размеру переднего плана
        background_image = background_image.resize(foreground_image.size)

        # 4. Создание композитного изображения: фон + передний план (без фона)
        composite_image = Image.alpha_composite(background_image.convert("RGBA"), foreground_image) # Композиция RGBA

        # 5. Сохранение результирующего изображения
        composite_image.convert("RGB").save(output_image_path) # Сохраняем как RGB, так как фон непрозрачный
        print(f"Фон успешно изменен и изображение сохранено в: {output_image_path}")

    except FileNotFoundError:
        print("Ошибка: Один или оба файла изображения не найдены.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

import io # Импортируем io для работы с байтами в памяти


# Пример использования:
if __name__ == "__main__":
    input_image = "stable_diffusion_image.png" # Замените на путь к вашему сгенерированному изображению
    new_background = "new_background.jpg"   # Замените на путь к изображению нового фона
    output_image = "output_with_new_background.png" # Путь для сохранения результата

    # Создадим примеры файлов для демонстрации (замените на свои пути!)
    # Вам нужно будет заменить эти пути на ваши собственные изображения.
    # Для примера, создадим пустые файлы-заглушки, чтобы код запустился без ошибок:
    with open(input_image, 'w') as f: pass # Создаем пустой файл для примера
    with open(new_background, 'w') as f: pass # Создаем пустой файл для примера


    change_stable_diffusion_background(input_image, new_background, output_image)