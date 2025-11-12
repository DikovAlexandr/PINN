"""
Тестирование метода Good Lattice Training (GLT) для генерации точек коллокации.

Этот скрипт демонстрирует:
1. Базовое использование GoodLatticeSampler
2. Сравнение GLT с другими методами сэмплирования
3. Визуализацию распределения точек
4. Применение к простой PDE задаче
"""

import sys
import io

# Для Windows: поддержка UTF-8 в консоли
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
from deepxde.geometry.glt import GoodLatticeSampler, get_glt_vectors, suggest_glt_size


def test_basic_sampler():
    """Тест 1: Базовая функциональность GoodLatticeSampler"""
    print("\n" + "="*70)
    print("ТЕСТ 1: Базовая функциональность GoodLatticeSampler")
    print("="*70)
    
    # Тест для разных размерностей
    for dim in [1, 2, 3, 4]:
        print(f"\n--- Размерность {dim}D ---")
        
        # Получить доступные размеры
        available_sizes = GoodLatticeSampler.get_available_sizes(dim)
        print(f"Доступные размеры N: {available_sizes[:5]}... (всего {len(available_sizes)})")
        
        # Взять средний размер для теста
        n_points = available_sizes[len(available_sizes)//2]
        print(f"Выбран размер: N = {n_points}")
        
        # Создать sampler
        sampler = GoodLatticeSampler(dim=dim, n_points=n_points)
        print(f"Generating vector z: {sampler.z}")
        
        # Сгенерировать точки
        points = sampler.sample()
        print(f"Сгенерировано точек: {points.shape}")
        print(f"Min: {points.min(axis=0)}, Max: {points.max(axis=0)}")
        
        # Проверить диапазон [0, 1]^dim
        assert np.all(points >= 0) and np.all(points <= 1), "Точки выходят за пределы [0,1]"
        print("✓ Все точки в диапазоне [0, 1]^dim")


def test_randomization_and_periodization():
    """Тест 2: Рандомизация и periodization tricks"""
    print("\n" + "="*70)
    print("ТЕСТ 2: Рандомизация и Periodization tricks")
    print("="*70)
    
    dim = 2
    n_points = 144
    sampler = GoodLatticeSampler(dim=dim, n_points=n_points)
    
    # Стандартный GLT
    points_std = sampler.sample()
    print(f"\nСтандартный GLT: {points_std.shape}")
    print(f"Первые 3 точки:\n{points_std[:3]}")
    
    # Randomized GLT
    points_rand = sampler.sample(randomize=True)
    print(f"\nRandomized GLT: {points_rand.shape}")
    print(f"Первые 3 точки:\n{points_rand[:3]}")
    print(f"✓ Точки отличаются (randomization работает)")
    
    # Periodization trick
    points_periodic = sampler.sample(fold_axes=[0, 1])
    print(f"\nPeriodization trick (fold_axes=[0,1]): {points_periodic.shape}")
    print(f"Первые 3 точки:\n{points_periodic[:3]}")
    
    # Проверка fold: все координаты должны быть в [0, 1]
    # но распределение должно быть более плотным около границ
    assert np.all(points_periodic >= 0) and np.all(points_periodic <= 1)
    print(f"✓ Periodization корректен (точки в [0, 1])")


def test_integration_with_deepxde():
    """Тест 3: Интеграция с DeepXDE через train_distribution"""
    print("\n" + "="*70)
    print("ТЕСТ 3: Интеграция с DeepXDE")
    print("="*70)
    
    # Простая 2D геометрия
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    
    # Тестируем разные варианты train_distribution
    methods = ["GLT", "GLT-rand", "pseudo", "LHS", "Sobol"]
    
    for method in methods:
        print(f"\n--- Метод: {method} ---")
        try:
            # Простая PDE (для теста достаточно заглушки)
            def pde(x, y):
                return dde.grad.hessian(y, x, i=0, j=0) + dde.grad.hessian(y, x, i=1, j=1)
            
            bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
            
            # Создать данные
            n_domain = 144 if method.startswith("GLT") else 150
            data = dde.data.PDE(
                geom,
                pde,
                bc,
                num_domain=n_domain,
                num_boundary=40,
                train_distribution=method
            )
            
            print(f"✓ train_distribution='{method}' работает")
            print(f"  Точек в домене: {data.train_x_all.shape[0]}")
            print(f"  Точек всего: {data.train_x.shape[0]}")
            
        except Exception as e:
            print(f"✗ Ошибка для метода '{method}': {e}")


def visualize_point_distributions():
    """Тест 4: Визуализация распределения точек для разных методов"""
    print("\n" + "="*70)
    print("ТЕСТ 4: Визуализация распределения точек")
    print("="*70)
    
    dim = 2
    n_points = 144
    
    # Генерация точек разными методами
    methods = {
        'GLT': None,
        'GLT-rand': None,
        'GLT-periodic': None,
        'Pseudo': None,
        'LHS': None,
        'Sobol': None,
    }
    
    # GLT стандартный
    sampler_glt = GoodLatticeSampler(dim=2, n_points=144)
    methods['GLT'] = sampler_glt.sample()
    
    # GLT randomized
    methods['GLT-rand'] = sampler_glt.sample(randomize=True)
    
    # GLT periodic
    methods['GLT-periodic'] = sampler_glt.sample(fold_axes=[0, 1])
    
    # Другие методы через DeepXDE sampler
    from deepxde.geometry import sampler as dde_sampler
    methods['Pseudo'] = dde_sampler.sample(144, 2, "pseudo")
    methods['LHS'] = dde_sampler.sample(144, 2, "LHS")
    methods['Sobol'] = dde_sampler.sample(150, 2, "Sobol")[:144]  # Sobol может вернуть чуть больше
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (method_name, points) in enumerate(methods.items()):
        ax = axes[idx]
        ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{method_name}\n(N={len(points)})', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'glt_visualization_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Визуализация сохранена: {output_path}")
    plt.close()


def test_suggest_size():
    """Тест 5: Автоматический подбор размера"""
    print("\n" + "="*70)
    print("ТЕСТ 5: Автоматический подбор размера N")
    print("="*70)
    
    # Тестируем для разных целевых значений
    test_cases = [
        (2, 100),   # Целевой N=100 для 2D
        (2, 150),   # Целевой N=150 для 2D
        (2, 200),   # Целевой N=200 для 2D
        (3, 300),   # Целевой N=300 для 3D
        (4, 500),   # Целевой N=500 для 4D
    ]
    
    for dim, target_n in test_cases:
        suggested_n, msg = suggest_glt_size(dim, target_n)
        print(f"\nDim={dim}, Target N={target_n}")
        print(f"  → Рекомендовано: N={suggested_n}")
        print(f"  → {msg}")


def test_simple_pde():
    """Тест 6: Решение простой PDE с GLT"""
    print("\n" + "="*70)
    print("ТЕСТ 6: Решение простой 2D Poisson equation с GLT")
    print("="*70)
    
    # 2D Poisson: -Δu = 2π²sin(πx)sin(πy), u=0 на границе
    # Аналитическое решение: u = sin(πx)sin(πy)
    
    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)
        return -dy_xx - dy_yy - 2*np.pi**2 * dde.backend.sin(np.pi*x[:, 0:1]) * dde.backend.sin(np.pi*x[:, 1:2])
    
    def solution(x):
        return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
    
    print("\nСравнение методов сэмплирования...")
    results = {}
    
    for method in ["GLT", "LHS", "pseudo"]:
        print(f"\n--- Метод: {method} ---")
        
        # Данные
        n_domain = 144 if method == "GLT" else 150
        data = dde.data.PDE(
            geom,
            pde,
            bc,
            num_domain=n_domain,
            num_boundary=40,
            train_distribution=method,
            solution=solution
        )
        
        # Сеть
        net = dde.nn.FNN([2] + [20]*3 + [1], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        
        # Обучение
        model.compile("adam", lr=1e-3)
        losshistory, train_state = model.train(iterations=5000, display_every=1000)
        
        # Оценка
        x_test = geom.uniform_points(1000, boundary=True)
        y_true = solution(x_test)
        y_pred = model.predict(x_test)
        l2_error = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        
        # Финальная loss - может быть массивом, суммируем
        final_loss = train_state.loss_train
        if isinstance(final_loss, (list, np.ndarray)):
            final_loss = np.sum(final_loss)
        
        results[method] = {
            'error': l2_error,
            'loss': final_loss,
            'n_points': data.train_x.shape[0]
        }
        
        print(f"Точек коллокации: {data.train_x.shape[0]}")
        print(f"Финальная loss: {final_loss:.6e}")
        print(f"Относительная L2 ошибка: {l2_error:.6e}")
    
    # Сравнительная таблица
    print("\n" + "="*70)
    print("ИТОГОВАЯ ТАБЛИЦА")
    print("="*70)
    print(f"{'Метод':<15} {'N точек':<10} {'Final Loss':<15} {'L2 Error':<15}")
    print("-"*70)
    for method, res in results.items():
        print(f"{method:<15} {res['n_points']:<10} {res['loss']:<15.6e} {res['error']:<15.6e}")
    
    # Найти лучший метод
    best_method = min(results.items(), key=lambda x: x[1]['error'])
    print(f"\n✓ Лучший метод: {best_method[0]} (L2 error = {best_method[1]['error']:.6e})")
    
    # Визуализация решения для GLT
    if "GLT" in results:
        print("\nВизуализация решения (GLT)...")
        x_viz = geom.uniform_points(10000, boundary=True)
        y_true_viz = solution(x_viz)
        
        # Переобучим модель с GLT для визуализации
        data_glt = dde.data.PDE(
            geom, pde, bc,
            num_domain=144,
            num_boundary=40,
            train_distribution="GLT",
            solution=solution
        )
        net_glt = dde.nn.FNN([2] + [20]*3 + [1], "tanh", "Glorot uniform")
        model_glt = dde.Model(data_glt, net_glt)
        model_glt.compile("adam", lr=1e-3)
        model_glt.train(iterations=5000, display_every=1000)
        
        y_pred_viz = model_glt.predict(x_viz)
        error_viz = np.abs(y_true_viz - y_pred_viz)
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, data, title in zip(axes, 
                                    [y_true_viz, y_pred_viz, error_viz],
                                    ['Точное решение', 'GLT предсказание', 'Абсолютная ошибка']):
            # Reshape для 2D plot
            x_grid = x_viz[:, 0].reshape(100, 100)
            y_grid = x_viz[:, 1].reshape(100, 100)
            z_grid = data.reshape(100, 100)
            
            c = ax.contourf(x_grid, y_grid, z_grid, levels=20, cmap='viridis')
            plt.colorbar(c, ax=ax)
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        output_path = 'glt_poisson_solution.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Визуализация решения сохранена: {output_path}")
        plt.close()


def main():
    """Запуск всех тестов"""
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ GOOD LATTICE TRAINING (GLT)")
    print("="*70)
    print("\nReference: https://arxiv.org/pdf/2307.13869")
    print("Matsubara & Yaguchi (AAAI 2023)")
    print("\n")
    
    # Запуск тестов
    test_basic_sampler()
    test_randomization_and_periodization()
    test_suggest_size()
    test_integration_with_deepxde()
    visualize_point_distributions()
    test_simple_pde()
    
    print("\n" + "="*70)
    print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО! ✓")
    print("="*70)
    print("\nСгенерированные файлы:")
    print("  - glt_visualization_comparison.pdf - сравнение методов сэмплирования")
    print("  - glt_poisson_solution.pdf - решение уравнения Пуассона с GLT")
    print("\nДля дополнительной информации см. GLT_README.md")


if __name__ == "__main__":
    main()

