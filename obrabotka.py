import os
import numpy as np
import matplotlib.pyplot as plt


# НАСТРОЙКИ 

DATA_PATH = "../data/"    # папка с файлами данных
PLOTS_PATH = "../plots/"  # папка, куда сохранять графики

# Файлы и давления калибровки давления (п1.4)
CAL_ZERO_FILE = "calibration/00 Pa.txt"
CAL_PRESS_FILE = "calibration/45 Pa.txt"
P_ZERO_PA = 0.0   # давление при выключенном вентиляторе, Па
P_MANOMETER_PA = 45.0 # давление по цифровому манометру, Па

# Данные калибровки перемещения (п1.4):
CAL_STEPS = 500
CAL_DISTANCE_MM = 18.0

# Физические параметры установки
RHO = 1.2                 # плотность воздуха, кг/м^3
NOZZLE_DIAMETER_MM = 15.0    # диаметр сопла, мм
ADC_GLITCH_LIMIT = 500000     # порог отсечения сбойных отсчётов АЦП
EDGE_POINTS = 5          # число точек на краях для поиска нуля давления
MIN_PRESSURE_THRESHOLD_PA = 3.0   # минимальный порог обнуления давления, Па

# Расстояния сечений от сопла, мм (п1.5)
DISTANCES_MM = [0, 10, 20, 30, 40, 50, 60, 70]

# Способ центровки: "com" (центр масс) или "fwhm" (середина по полувысоте)
CENTER_METHOD = "com"

# Необязательные ручные поправки центровки в мм (если захотите подправить вручную)
MANUAL_SHIFTS_MM = {d: 0.0 for d in DISTANCES_MM}

# Совместимость с NumPy 1.x (trapz) и NumPy 2.x (trapezoid)
TRAPZ = getattr(np, "trapezoid", np.trapz)



# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def load_adc_file(filepath):
    """Читает файл с данными (2 столбца: step, adc; либо 1 столбец: adc)
    и отбрасывает сбойные отсчёты АЦП."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        adc = data
        steps = np.arange(len(adc), dtype=float)
    else:
        steps, adc = data[:, 0], data[:, 1]

    mask = adc < ADC_GLITCH_LIMIT
    return steps[mask], adc[mask]


def calibrate_pressure():
    """п1.4: Калибровка давления P = k * N + b и сохранение графика."""
    _, adc_zero = load_adc_file(os.path.join(DATA_PATH, CAL_ZERO_FILE))
    _, adc_press = load_adc_file(os.path.join(DATA_PATH, CAL_PRESS_FILE))

    mean_zero = np.mean(adc_zero)
    mean_press = np.mean(adc_press)

    adc_points = np.array([mean_zero, mean_press])
    p_points = np.array([P_ZERO_PA, P_MANOMETER_PA])

    # Линейная аппроксимация P = k*N + b
    k, b = np.polyfit(adc_points, p_points, 1)

    # График калибровки давления
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    adc_line = np.linspace(adc_points.min() * 0.98, adc_points.max() * 1.02, 200)
    p_line = k * adc_line + b

    ax.plot(
        adc_line, p_line,
        color="#1f77b4", linewidth=2,
        label=f"P = {k:.4f}·N + ({b:.2f}) [Па]"
    )
    ax.scatter(
        adc_points, p_points,
        color="#d62728", s=60, zorder=5,
        label="Экспериментальные точки"
    )

    ax.set_title("Калибровка датчика давления")
    ax.set_xlabel("Показания АЦП, отсчёты")
    ax.set_ylabel("Давление P, Па")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, "pressure-calibration.png"))
    plt.close(fig)

    return k, b


def calibrate_distance():
    """п1.4: Калибровка перемещения (метры на шаг) и сохранение графика."""
    steps_points = np.array([0.0, float(CAL_STEPS)])
    dist_mm_points = np.array([0.0, float(CAL_DISTANCE_MM)])

    slope_mm, intercept_mm = np.polyfit(steps_points, dist_mm_points, 1)
    step_to_m = slope_mm * 1e-3

    # График калибровки перемещения
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    steps_line = np.linspace(0, CAL_STEPS * 1.05, 200)
    dist_line = slope_mm * steps_line + intercept_mm

    ax.plot(
        steps_line, dist_line,
        color="#1f77b4", linewidth=2,
        label=f"x = {slope_mm:.5f}·step [мм] ({step_to_m:.2e} м/шаг)"
    )
    ax.scatter(
        steps_points, dist_mm_points,
        color="#d62728", s=60, zorder=5,
        label="Экспериментальные точки"
    )

    ax.set_title("Калибровка перемещения шагового двигателя")
    ax.set_xlabel("Количество шагов двигателя")
    ax.set_ylabel("Перемещение трубки Пито, мм")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, "distance-calibration.png"))
    plt.close(fig)

    return step_to_m


def calculate_velocity(filepath, k_pressure, step_to_m):
    """п1.5: Пересчёт отсчётов АЦП и шагов в координату x (м) и скорость u (м/с)."""
    steps, adc = load_adc_file(filepath)

    # 1. Ноль давления для конкретного профиля — прямая между медианами краёв
    n = EDGE_POINTS
    left = np.median(adc[:n])
    right = np.median(adc[-n:])
    zero = np.interp(
        np.arange(len(adc)),
        [n / 2.0, len(adc) - n / 2.0],
        [left, right]
    )

    # 2. Перевод в избыточное давление (Па)
    p = k_pressure * (adc - zero)

    # 3. Порог шума по краям (считаем отклонения относительно своего уровня края)
    edge_residuals = np.concatenate([adc[:n] - left, adc[-n:] - right])
    noise_threshold = max(3.0 * k_pressure * np.std(edge_residuals), MIN_PRESSURE_THRESHOLD_PA)

    # Обнуляем отрицательные и малые шумовые давления, чтобы не было nan
    p[p < noise_threshold] = 0.0

    # 4. Уравнение Бернулли: p = rho * u^2 / 2 => u = sqrt(2 * p / rho)
    u = np.sqrt(2.0 * p / RHO)

    # 5. Координата в метрах
    x = steps * step_to_m

    # Гарантируем сортировку по возрастанию x
    order = np.argsort(x)
    return x[order], u[order]


def find_center_and_fwhm(x, u, method="com"):
    """п1.5 (+1 балл): Автоматическое нахождение центра струи и ширины на полувысоте."""
    u_max = np.max(u)
    if u_max <= 0:
        return 0.0, 0.0

    # Точки выше половины максимума (нужны и для fwhm-центровки, и для проверки диаметра сопла)
    above_half = np.where(u >= 0.5 * u_max)[0]
    x_left_half = x[above_half[0]]
    x_right_half = x[above_half[-1]]
    fwhm_m = x_right_half - x_left_half

    if method == "fwhm":
        x_center = 0.5 * (x_left_half + x_right_half)
    else:
        # Центр масс профиля скорости
        x_center = np.sum(x * u) / np.sum(u)

    return x_center, fwhm_m


def calculate_outgo(x, u):
    """п1.6: Массовый расход в сечении (кг/с) методом трапеций.
    Усреднение по двум половинам: 0.5 * integral(rho * 2 * pi * |x| * u dx)."""
    r = np.abs(x)
    integrand = RHO * 2.0 * np.pi * r * u
    q_kg_s = 0.5 * TRAPZ(integrand, x)
    return q_kg_s



# ОСНОВНОЙ БЛОК ОБРАБОТКИ

def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)

    # --- Шаг 1. Калибровки (п1.4) ---
    k_pressure, b_pressure = calibrate_pressure()
    step_to_m = calibrate_distance()

    print("=== КАЛИБРОВКИ (п1.4) ===")
    print(f"k = {k_pressure:.4f} Па/отсчёт (ориентир ~0.045 Па/отсчёт)")
    print(f"1 шаг = {step_to_m:.2e} м ({step_to_m * 1e3:.4f} мм/шаг, ориентир ~0.036 мм)")
    print()

    # --- Шаг 2. Профили скорости, центровка и расчёт расхода (п1.5 и п1.6) ---
    profiles = {}
    outgoes_g_s = []

    print("=== РЕЗУЛЬТАТЫ ПО СЕЧЕНИЯМ (п1.5–п1.6) ===")
    for d in DISTANCES_MM:
        filename = f"{d:02d} mm.txt"
        filepath = os.path.join(DATA_PATH, filename)

        x_raw, u = calculate_velocity(filepath, k_pressure, step_to_m)
        x_center, fwhm_m = find_center_and_fwhm(x_raw, u, method=CENTER_METHOD)

        # Сдвиг в центр + возможная ручная поправка
        x_centered = x_raw - x_center - MANUAL_SHIFTS_MM[d] * 1e-3

        # Массовый расход (кг/с и г/с)
        q_kg_s = calculate_outgo(x_centered, u)
        q_g_s = q_kg_s * 1e3

        profiles[d] = {
            "x": x_centered,
            "u": u,
            "q_kg_s": q_kg_s,
            "q_g_s": q_g_s,
            "fwhm_mm": fwhm_m * 1e3,
            "u_max": np.max(u),
        }
        outgoes_g_s.append(q_g_s)

        print(
            f"z = {d:2d} мм | u_max = {np.max(u):5.2f} м/с | "
            f"FWHM = {fwhm_m * 1e3:5.2f} мм | Q = {q_g_s:5.2f} г/с"
        )

    # --- Шаг 3. Проверка калибровки по диаметру сопла и оценка Q0 = V * S * rho ---
    if 0 in profiles:
        p0 = profiles[0]
        # Скорость в ядре струи (берём среднее по плоскому плато в пределах +-4 мм от оси)
        core_mask = np.abs(p0["x"] * 1e3) <= 4.0
        v_core = np.mean(p0["u"][core_mask]) if np.any(core_mask) else p0["u_max"]

        d_m = NOZZLE_DIAMETER_MM * 1e-3
        s_nozzle = np.pi * (d_m ** 2) / 4.0
        q_est_g_s = (v_core * s_nozzle * RHO) * 1e3

        diam_err_pct = abs(p0["fwhm_mm"] - NOZZLE_DIAMETER_MM) / NOZZLE_DIAMETER_MM * 100
        outgo_err_pct = abs(p0["q_g_s"] - q_est_g_s) / q_est_g_s * 100

        print("\n=== ПРОВЕРКА КАЛИБРОВКИ НА СРЕЗЕ СОПЛА (00 мм) ===")
        print(f"Диаметр сопла: d = {NOZZLE_DIAMETER_MM:.1f} мм")
        print(f"Ширина профиля (FWHM) на 00 мм: {p0['fwhm_mm']:.2f} мм (отклонение {diam_err_pct:.1f}%)")
        print(f"Скорость в ядре струи V_core = {v_core:.2f} м/с (u_max = {p0['u_max']:.2f} м/с)")
        print(f"Площадь сопла S = {s_nozzle:.3e} м^2")
        print(f"Теоретическая оценка Q0 = V·S·rho = {q_est_g_s:.2f} г/с")
        print(f"Экспериментальный расход Q(00 мм) = {p0['q_g_s']:.2f} г/с (расхождение {outgo_err_pct:.1f}%)")

    # --- Шаг 4. График скоростей без расходов: velocity.png (п1.5) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for d in DISTANCES_MM:
        ax.plot(
            profiles[d]["x"] * 1e3,
            profiles[d]["u"],
            linewidth=1.6,
            label=f"{d:02d} мм"
        )

    ax.set_title("Профили скорости в сечениях затопленной струи")
    ax.set_xlabel("Положение трубки Пито относительно центра струи, мм")
    ax.set_ylabel("Скорость потока воздуха, м/с")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend(title="Расстояние от сопла")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, "velocity.png"))
    plt.close(fig)

    # --- Шаг 5. График скоростей с расходами в легенде: velocity-outgo.png (п1.6) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for d in DISTANCES_MM:
        ax.plot(
            profiles[d]["x"] * 1e3,
            profiles[d]["u"],
            linewidth=1.6,
            label=f"Q ({d:02d} мм) = {profiles[d]['q_g_s']:.2f} [г/с]"
        )

    ax.set_title("Профили скорости и массовый расход в сечениях затопленной струи")
    ax.set_xlabel("Положение трубки Пито относительно центра струи, мм")
    ax.set_ylabel("Скорость потока воздуха, м/с")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, "velocity-outgo.png"))
    plt.close(fig)

    # --- Шаг 6. График зависимости расхода от расстояния: outgo-distance.png (п1.6) ---
    dist_arr = np.array(DISTANCES_MM, dtype=float)
    outgo_arr = np.array(outgoes_g_s, dtype=float)

    slope_q, intercept_q = np.polyfit(dist_arr, outgo_arr, 1)
    dist_line = np.linspace(dist_arr.min(), dist_arr.max(), 200)
    outgo_line = slope_q * dist_line + intercept_q

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.scatter(
        dist_arr, outgo_arr,
        color="#d62728", s=55, zorder=5,
        label="Экспериментальные значения расхода"
    )
    ax.plot(
        dist_line, outgo_line,
        color="#1f77b4", linewidth=2, linestyle="--",
        label=f"Аппроксимация: Q(z) = {slope_q:.3f}·z + {intercept_q:.2f} [г/с]"
    )

    ax.set_title("Зависимость массового расхода затопленной струи от расстояния до сопла")
    ax.set_xlabel("Расстояние от среза сопла z, мм")
    ax.set_ylabel("Массовый расход Q, г/с")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, "outgo-distance.png"))
    plt.close(fig)

    print(f"\nГотово! Все 5 графиков сохранены в папку: {os.path.abspath(PLOTS_PATH)}")


if __name__ == "__main__":
    main()