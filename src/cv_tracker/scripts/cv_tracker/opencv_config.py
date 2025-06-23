#!/usr/bin/env python3
"""
OpenCV configuration for Orange Pi (CPU only, no GPU)
Настройки OpenCV для Orange Pi без GPU
"""

import cv2
import os
import rospy

# Пытаемся импортировать rospy, но не падаем если его нет
try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False
    print("Warning: rospy not available, using print instead of rospy.log*")

def log_info(message):
    """Логирование с поддержкой или без rospy"""
    if HAS_ROSPY:
        rospy.loginfo(message)
    else:
        print(f"INFO: {message}")

def log_error(message):
    """Логирование ошибок с поддержкой или без rospy"""
    if HAS_ROSPY:
        rospy.logerr(message)
    else:
        print(f"ERROR: {message}")

def log_debug(message):
    """Отладочное логирование"""
    if HAS_ROSPY:
        rospy.logdebug(message)
    else:
        print(f"DEBUG: {message}")

def configure_opencv():
    """Настройка OpenCV для избежания проблем с Mali GPU на ARM устройствах"""
    try:
        # АГРЕССИВНОЕ отключение всех GPU ускорений
        cv2.ocl.setUseOpenCL(False)
        
        # Отключаем все возможные GPU backends
        os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_0'] = '0'
        os.environ['OPENCV_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2_1'] = '0'
        os.environ['OPENCV_DNN_OPENCL'] = '0'
        
        # Критически важные настройки для Mali GPU
        os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
        os.environ['OPENCV_OCL_RUNTIME'] = ''
        os.environ['OPENCL_VENDOR_PATH'] = '/dev/null'
        
        # Отключаем TBB для избежания проблем многопоточности
        os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
        
        # Принудительно используем только CPU
        os.environ['OPENCV_DNN_BACKEND'] = 'opencv'
        os.environ['OPENCV_DNN_TARGET'] = 'cpu'
        
        # Ограничиваем использование памяти
        os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '89478485'  # Ограничение на размер изображений
        
        # Настройки для многопоточности - используем только 1 поток для избежания проблем
        cv2.setNumThreads(1)  # Строго 1 поток для максимальной стабильности
        
        # Дополнительная проверка - если OpenCL все еще доступен, принудительно отключаем
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(False)
            # Попытка полностью деинициализировать OpenCL контекст
            try:
                cv2.ocl.finish()
            except:
                pass
        
        print("OpenCV configured for ARM device compatibility")
        
        # Логируем информацию о сборке OpenCV
        print(f"OpenCV version: {cv2.__version__}")
        print(f"OpenCL support: {cv2.ocl.haveOpenCL()}")
        print(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
        print(f"Number of threads: {cv2.getNumThreads()}")
        
    except Exception as e:
        print(f"Error configuring OpenCV: {e}")

def get_safe_capture_properties():
    """Возвращает безопасные свойства для VideoCapture на Orange Pi"""
    return {
        cv2.CAP_PROP_BUFFERSIZE: 1,      # Минимальный буфер
        cv2.CAP_PROP_FPS: 15,            # Умеренный FPS
        cv2.CAP_PROP_FRAME_WIDTH: 640,   # Стандартное разрешение
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        # Не используем FOURCC принудительно, пусть камера выберет лучший формат
    }

def create_safe_video_capture(device_id=0):
    """Создает VideoCapture с безопасными настройками для Orange Pi"""
    try:
        log_info(f"Attempting to open camera {device_id} for Orange Pi")
        
        # Для Orange Pi лучше всего работает V4L2
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            log_info("V4L2 failed, trying default backend")
            cap = cv2.VideoCapture(device_id)
        
        if cap.isOpened():
            log_info("Camera opened successfully")
            
            # Применяем безопасные свойства по одному
            safe_props = get_safe_capture_properties()
            for prop, value in safe_props.items():
                try:
                    result = cap.set(prop, value)
                    if result:
                        actual_value = cap.get(prop)
                        log_debug(f"Set property {prop} to {actual_value}")
                    else:
                        log_debug(f"Failed to set property {prop}")
                except Exception as e:
                    log_debug(f"Could not set property {prop}: {e}")
            
            # Проверяем финальные параметры
            try:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                log_info(f"Camera configured: {width}x{height} @ {fps} fps")
            except Exception as e:
                log_debug(f"Could not read camera properties: {e}")
            
            return cap
        else:
            log_error("Could not open camera")
            return None
        
    except Exception as e:
        log_error(f"Error in create_safe_video_capture: {e}")
        return None

def create_safe_tracker():
    """Создание безопасного трекера для ARM устройств"""
    try:
        # Попробуем создать CSRT трекер (лучшая точность)
        tracker = cv2.legacy.TrackerCSRT_create()
        print("Created CSRT tracker")
        return tracker
    except AttributeError:
        try:
            # Альтернатива для новых версий OpenCV
            tracker = cv2.TrackerCSRT_create()
            print("Created CSRT tracker (new API)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create CSRT tracker: {e}")
    
    try:
        # Fallback к KCF трекеру
        tracker = cv2.legacy.TrackerKCF_create()
        print("Created KCF tracker (fallback)")
        return tracker
    except AttributeError:
        try:
            tracker = cv2.TrackerKCF_create()
            print("Created KCF tracker (new API, fallback)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create KCF tracker: {e}")
    
    try:
        # Последний fallback к MOSSE трекеру (самый быстрый)
        tracker = cv2.legacy.TrackerMOSSE_create()
        print("Created MOSSE tracker (last fallback)")
        return tracker
    except AttributeError:
        try:
            tracker = cv2.TrackerMOSSE_create()
            print("Created MOSSE tracker (new API, last fallback)")
            return tracker
        except:
            pass
    except Exception as e:
        print(f"Failed to create MOSSE tracker: {e}")
    
    print("Failed to create any tracker!")
    return None

def get_available_trackers():
    """Получение списка доступных трекеров"""
    trackers = []
    
    # Проверяем CSRT
    try:
        cv2.legacy.TrackerCSRT_create()
        trackers.append("CSRT")
    except:
        try:
            cv2.TrackerCSRT_create()
            trackers.append("CSRT")
        except:
            pass
    
    # Проверяем KCF
    try:
        cv2.legacy.TrackerKCF_create()
        trackers.append("KCF")
    except:
        try:
            cv2.TrackerKCF_create()
            trackers.append("KCF")
        except:
            pass
    
    # Проверяем MOSSE
    try:
        cv2.legacy.TrackerMOSSE_create()
        trackers.append("MOSSE")
    except:
        try:
            cv2.TrackerMOSSE_create()
            trackers.append("MOSSE")
        except:
            pass
    
    return trackers

# Настраиваем OpenCV при импорте модуля
if __name__ != '__main__':
    configure_opencv() 

if __name__ == "__main__":
    # Тестирование конфигурации
    configure_opencv()
    available = get_available_trackers()
    print(f"Available trackers: {available}")
    
    tracker = create_safe_tracker()
    if tracker:
        print("✓ Tracker creation successful")
    else:
        print("✗ Tracker creation failed") 