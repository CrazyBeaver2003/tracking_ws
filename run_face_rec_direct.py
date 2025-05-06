#!/usr/bin/env python3

import os
import sys
import time
import traceback
from pathlib import Path

# Явно добавляем все необходимые пути
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'src'))
sys.path.append(str(current_dir / 'src/cv_tracker'))
sys.path.append(str(current_dir / 'src/cv_tracker/scripts'))

# Непосредственный импорт модуля
script_path = current_dir / 'src/cv_tracker/scripts/cv_tracker/face_recognizer_direct.py'

try:
    # Проверяем наличие файла
    if not script_path.exists():
        print(f"ОШИБКА: Файл {script_path} не существует!")
        sys.exit(1)
        
    # Прямой импорт модуля из файла
    import importlib.util
    spec = importlib.util.spec_from_file_location("face_recognizer_direct", script_path)
    face_rec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(face_rec)
    
    # Создаем экземпляр FaceRecognizer
    recognizer = face_rec.FaceRecognizer()
    
    print("\nРаспознаватель запущен и работает в фоновом режиме.")
    print("Нажмите Ctrl+C для завершения.")
    
    # Ожидаем завершения
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\nПолучен сигнал прерывания, завершение работы...")
    if 'recognizer' in locals():
        recognizer.stop()
    
except Exception as e:
    print(f"Критическая ошибка: {e}")
    traceback.print_exc() 