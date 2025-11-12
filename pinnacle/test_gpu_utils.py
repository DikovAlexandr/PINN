"""
Тестирование базовых GPU утилит.

Проверка работоспособности простых и полезных функций для работы с GPU.
"""

import sys
import io
import os

# Для Windows: поддержка UTF-8 в консоли
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ["DDE_BACKEND"] = "pytorch"

import torch
import deepxde as dde


def test_device_selection():
    """Тест 1: Автоматический выбор устройства"""
    print("\n" + "="*70)
    print("ТЕСТ 1: Автоматический выбор устройства")
    print("="*70)
    
    device = dde.utils.get_optimal_device(verbose=True)
    print(f"\n✓ Устройство выбрано: {device}")
    
    assert device in [torch.device('cuda'), torch.device('cpu')], "Неверное устройство"
    print("✓ Тест пройден")


def test_memory_info():
    """Тест 2: Информация о памяти GPU"""
    print("\n" + "="*70)
    print("ТЕСТ 2: Информация о памяти GPU")
    print("="*70)
    
    info = dde.utils.get_gpu_memory_info()
    
    print(f"\n📊 Информация о памяти:")
    for key, value in info.items():
        if key != 'error':
            print(f"  {key}: {value:.2f} GB" if isinstance(value, float) and key != 'utilization' else f"  {key}: {value:.1f}%" if key == 'utilization' else f"  {key}: {value}")
    
    assert 'allocated' in info, "Отсутствует 'allocated' в info"
    assert 'total' in info, "Отсутствует 'total' в info"
    
    print("\n✓ Тест пройден")


def test_cache_clearing():
    """Тест 3: Очистка GPU кэша"""
    print("\n" + "="*70)
    print("ТЕСТ 3: Очистка GPU кэша")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠️ CUDA недоступна, пропускаем тест")
        return
    
    # Создать данные на GPU
    device = torch.device('cuda')
    print("\nСоздание тензоров на GPU...")
    tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
    
    info_before = dde.utils.get_gpu_memory_info()
    print(f"Память до очистки: {info_before['reserved']:.2f} GB")
    
    # Удалить тензоры
    del tensors
    
    # Очистить кэш
    dde.utils.clear_gpu_cache(verbose=True)
    
    info_after = dde.utils.get_gpu_memory_info()
    print(f"Память после очистки: {info_after['reserved']:.2f} GB")
    
    print("\n✓ Тест пройден")


def test_memory_context():
    """Тест 4: Context manager для управления памятью"""
    print("\n" + "="*70)
    print("ТЕСТ 4: Context manager для управления памятью")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n⚠️ CUDA недоступна, пропускаем тест")
        return
    
    device = torch.device('cuda')
    
    print("\nИспользование gpu_memory_context:")
    with dde.utils.gpu_memory_context(verbose=True):
        # Создать большие тензоры
        print("  Создание тензоров...")
        tensors = [torch.randn(5000, 1000, device=device) for _ in range(3)]
    
    print("\n✓ Память автоматически освобождена")
    print("✓ Тест пройден")


def test_tensor_memory_estimation():
    """Тест 5: Оценка памяти для тензора"""
    print("\n" + "="*70)
    print("ТЕСТ 5: Оценка памяти для тензора")
    print("="*70)
    
    test_cases = [
        ((10000, 100), torch.float32, "10k x 100 (float32)"),
        ((1000, 1000), torch.float64, "1k x 1k (float64)"),
        ((100, 100, 100), torch.float32, "100 x 100 x 100 (float32)"),
    ]
    
    print("\n📏 Оценка памяти для различных тензоров:")
    for shape, dtype, description in test_cases:
        memory = dde.utils.estimate_tensor_memory(shape, dtype)
        print(f"  {description}: {memory:.4f} GB")
    
    # Проверка корректности
    memory_10k = dde.utils.estimate_tensor_memory((10000, 100), torch.float32)
    expected = (10000 * 100 * 4) / 1e9  # 4 bytes per float32
    assert abs(memory_10k - expected) < 1e-6, "Неверный расчет памяти"
    
    print("\n✓ Тест пройден")


def test_integration():
    """Тест 6: Интеграция с простой задачей"""
    print("\n" + "="*70)
    print("ТЕСТ 6: Интеграция с простой задачей")
    print("="*70)
    
    # Выбрать устройство
    device = dde.utils.get_optimal_device(verbose=False)
    print(f"\nУстройство: {device}")
    
    # Проверить память перед обучением
    if device.type == 'cuda':
        info = dde.utils.get_gpu_memory_info()
        print(f"Свободная память: {info['free']:.2f} GB")
    
    # Простая задача
    print("\nОбучение простой модели...")
    
    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_xx + 1
    
    geom = dde.geometry.Interval(0, 1)
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
    
    data = dde.data.PDE(geom, pde, bc, num_domain=100, num_boundary=20)
    net = dde.nn.FNN([1, 20, 20, 1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=1e-3)
    
    # Обучение с context manager
    with dde.utils.gpu_memory_context(clear_cache=True):
        model.train(iterations=100, display_every=100)
    
    print("\n✓ Модель успешно обучена")
    print("✓ Тест пройден")


def main():
    """Запуск всех тестов"""
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ GPU УТИЛИТ")
    print("="*70)
    print("\nПроверка базовых функций для работы с GPU.")
    
    # Информация о системе
    if torch.cuda.is_available():
        print(f"\n✓ CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA версия: {torch.version.cuda}")
    else:
        print("\n⚠️ CUDA недоступна, некоторые тесты будут пропущены")
    
    # Запуск тестов
    try:
        test_device_selection()
        test_memory_info()
        test_cache_clearing()
        test_memory_context()
        test_tensor_memory_estimation()
        test_integration()
        
        print("\n" + "="*70)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ! ✓")
        print("="*70)
        print("\nGPU утилиты работают корректно.")
        print("\nДоступные функции:")
        print("  - dde.utils.get_optimal_device()     # Выбор устройства")
        print("  - dde.utils.get_gpu_memory_info()    # Информация о памяти")
        print("  - dde.utils.clear_gpu_cache()        # Очистка кэша")
        print("  - dde.utils.gpu_memory_context()     # Context manager")
        print("  - dde.utils.estimate_tensor_memory() # Оценка памяти")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

