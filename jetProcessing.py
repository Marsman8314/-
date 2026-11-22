import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data")
    PLOTS_PATH = os.path.join(BASE_DIR, "..", "plots")
    SHOW_PLOTS = False
else:
    BASE_DIR = os.getcwd()
    DATA_PATH = BASE_DIR
    PLOTS_PATH = os.path.join(BASE_DIR, "plots")
    SHOW_PLOTS = True

# Файлы калибровки давления
CAL_FILES = [
    "profie_without_vozduhoduvka_cal_0mm.txt",
    "profie_vozduhoduvka_cal_0mm.txt",
    "profie_vozduhoduvka2_cal_0mm.txt",
    "profie_vozduhoduvka3_cal_0mm.txt",
]

# True — строить прямую по 40, 100 и 140 Па (точка 0 Па = 208000 сильно выбивается)
# False — включить в прямую все 4 точки
EXCLUDE_ZERO_CAL = True

# Файлы сечений (если 80 мм не нужно — поставь # перед строкой 80)
SECTION_FILES = {
    0:  "proflie_0mmnew.txt",
    10: "proflie_10mm.txt",
    20: "proflie_20mm.txt",
    30: "proflie_30mm.txt",
    40: "proflie_40mm.txt",
    50: "proflie_50mm.txt",
    60: "proflie_60mm.txt",
    70: "proflie_70mm.txt",
    80: "proflie_80mm.txt",
}

# Данные калибровки перемещения (ЗАМЕНИ на ваши реальные шаги и миллиметры от экспериментатора!)
CAL_STEPS = 500
CAL_DISTANCE_MM = 18.0

# Физические параметры установки
RHO = 1.2                         # плотность воздуха, кг/м^3
NOZZLE_DIAMETER_MM = 15.0         # диаметр сопла, мм
ADC_GLITCH_LIMIT = 500000         # порог отсечения сбойных отсчётов АЦП
EDGE_POINTS = 5                   # число точек на краях для поиска нуля давления
MIN_PRESSURE_THRESHOLD_PA = 3.0   # минимальный порог обнуления давления, Па

# Способ центровки: "com" (центр масс) или "fwhm" (середина по полувысоте)
CENTER_METHOD = "com"

# Необязательные ручные поправки центровки в мм (по умолчанию 0)
MANUAL_SHIFTS_MM = {d: 0.0 for d in SECTION_FILES}

# Совместимость с NumPy 1.x (trapz) и NumPy 2.x (trapezoid)
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# =============================================================================
# ПОИСК И ЗАГРУЗКА ФАЙЛОВ
# =============================================================================
def find_file(filename, raise_error=True):
    """Ищет файл в стандартных папках и рекурсивно в рабочей директории."""
    candidates = [
        os.path.join(DATA_PATH, filename),
        os.path.join(DATA_PATH, "calibration", filename),
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, "data", "calibration", filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Поиск по всем вложенным папкам (на случай если файлы загружены в подпапку)
    for root, _, filenames in os.walk(BASE_DIR):
        if filename in filenames:
            return os.path.join(root, filename)

    if raise_error:
        raise FileNotFoundError(f"Файл '{filename}' не найден.")
    return None


def ensure_files_in_colab():
    """В Google Colab открывает кнопку загрузки, если не хватает каких-либо .txt файлов."""
    if not IN_COLAB:
        return

    required = list(CAL_FILES) + list(SECTION_FILES.values())
    while True:
        missing = [f for f in required if find_file(f, raise_error=False) is None]
        if not missing:
            break
        print(f"Нужно загрузить ещё {len(missing)} файл(ов):")
        for f in missing:
            print("  -", f)
        print("Нажми 'Выберите файлы' ниже и выдели нужные .txt файлы (можно сразу все через Ctrl+A):")
        uploaded = colab_files.upload()
        if not uploaded:
            break


def save_and_show(fig, filename):
    """Сохраняет график в plots/ и показывает его в ячейке Colab."""
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_PATH, filename))
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def load_cal_point(filepath):
    """Читает калибровочный файл формата 'step adc pressure'."""
    data = np.atleast_2d(np.loadtxt(filepath))
    adc = data[:, 1]
    adc = adc[adc < ADC_GLITCH_LIMIT]
    pressure = float(np.mean(data[:, 2]))
    return float(np.mean(adc)), pressure


def load_profile_file(filepath):
    """Читает файл сечения формата 'step adc' и отбрасывает сбойные отсчёты АЦП."""
    data = np.atleast_2d(np.loadtxt(filepath))
    steps, adc = data[:, 0], data[:, 1]
    mask = adc < ADC_GLITCH_LIMIT
    return steps[mask], adc[mask]


# =============================================================================
# РАСЧЁТЫ И ГРАФИКИ
# =============================================================================
def calibrate_pressure():
    """п1.4: Калибровка давления P = k * N + b и сохранение pressure-calibration.png."""
    all_adc, all_p = [], []
    for fname in CAL_FILES:
        adc_val, p_val = load_cal_point(find_file(fname))
        all_adc.append(adc_val)
        all_p.append(p_val)

    all_adc = np.array(all_adc)
    all_p = np.array(all_p)

    fit_mask = (all_p > 0) if EXCLUDE_ZERO_CAL else np.ones_like(all_p, dtype=bool)
    adc_fit, p_fit = all_adc[fit_mask], all_p[fit_mask]

    k, b = np.polyfit(adc_fit, p_fit, 1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    adc_line = np.linspace(adc_fit.min() - 1000, adc_fit.max() + 500, 200)
    p_line = k * adc_line + b

    ax.plot(
        adc_line, p_line,
        color="#1f77b4", linewidth=2,
        label=f"P = {k:.4f}·N + ({b:.1f}) [Па]"
    )
    ax.scatter(
        adc_fit, p_fit,
        color="#d62728", s=65, zorder=5,
        label="Экспериментальные точки"
    )

    if EXCLUDE_ZERO_CAL and np.any(~fit_mask):
        ax.scatter(
            all_adc[~fit_mask], all_p[~fit_mask],
            color="gray", marker="x", s=75, zorder=5,
            label="Исключённая точка (0 Па: дрейф нуля)"
        )

    ax.set_title("Калибровка датчика давления")
    ax.set_xlabel("Показания АЦП, отсчёты")
    ax.set_ylabel("Давление P, Па")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    save_and_show(fig, "pressure-calibration.png")

    return k, b


def calibrate_distance():
    """п1.4: Калибровка перемещения (метры на шаг) и сохранение distance-calibration.png."""
    steps_points = np.array([0.0, float(CAL_STEPS)])
    dist_mm_points = np.array([0.0, float(CAL_DISTANCE_MM)])

    slope_mm, intercept_mm = np.polyfit(steps_points, dist_mm_points, 1)
    step_to_m = slope_mm * 1e-3

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    steps_line = np.linspace(0, CAL_STEPS * 1.05, 200)
    dist_line = slope_mm * steps_line + intercept_mm

    ax.plot(
        steps_line, dist_line,
        color="#1f77b4", linewidth=2,
        label=f"x = {slope_mm:.5f}·step [мм] ({step_to_m:.2e} м/шаг)"
    )
    ax.scatter(
        steps_points, dist_mm_points,
        color="#d62728", s=65, zorder=5,
        label="Экспериментальные точки"
    )

    ax.set_title("Калибровка перемещения шагового двигателя")
    ax.set_xlabel("Количество шагов двигателя")
    ax.set_ylabel("Перемещение трубки Пито, мм")
    ax.grid(True, which="major", linestyle="-", alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.legend()
    save_and_show(fig, "distance-calibration.png")

    return step_to_m


def calculate_velocity(filepath, k_pressure, step_to_m):
    """п1.5: Пересчёт отсчётов АЦП и шагов в координату x (м) и скорость u (м/с)."""
    steps, adc = load_profile_file(filepath)

    n = EDGE_POINTS
    left = np.median(adc[:n])
    right = np.median(adc[-n:])
    zero = np.interp(
        np.arange(len(adc)),
        [n / 2.0, len(adc) - n / 2.0],
        [left, right]
    )

    p = k_pressure * (adc - zero)

    edge_residuals = np.concatenate([adc[:n] - left, adc[-n:] - right])
    noise_threshold = max(3.0 * k_pressure * np.std(edge_residuals), MIN_PRESSURE_THRESHOLD_PA)
    p[p < noise_threshold] = 0.0

    u = np.sqrt(2.0 * p / RHO)
    x = steps * step_to_m

    order = np.argsort(x)
    return x[order], u[order]


def find_center_and_fwhm(x, u, method="com"):
    """п1.5 (+1 балл): Автоматический поиск центра струи и ширины на полувысоте."""
    u_max = np.max(u)
    if u_max <= 0:
        return 0.0, 0.0

    half = 0.5 * u_max
    idx = np.where(u >= half)[0]
    i_l, i_r = idx[0], idx[-1]

    if i_l > 0 and u[i_l] != u[i_l - 1]:
        x_l = np.interp(half, [u[i_l - 1], u[i_l]], [x[i_l - 1], x[i_l]])
    else:
        x_l = x[i_l]

    if i_r < len(u) - 1 and u[i_r] != u[i_r + 1]:
        x_r = np.interp(half, [u[i_r + 1], u[i_r]], [x[i_r + 1], x[i_r]])
    else:
        x_r = x[i_r]

    fwhm_m = x_r - x_l

    if method == "fwhm":
        x_center = 0.5 * (x_l + x_r)
    else:
        x_center = np.sum(x * u) / np.sum(u)

    return x_center, fwhm_m


def calculate_outgo(x, u):
    """п1.6: Массовый расход в сечении (кг/с), усреднённый по обеим половинам."""
    integrand = RHO * 2.0 * np.pi * np.abs(x) * u
    return 0.5 * TRAPZ(integrand, x)


# =============================================================================
# ОСНОВНОЙ БЛОК ОБРАБОТКИ
# =============================================================================
def main():
    os.makedirs(PLOTS_PATH, exist_ok=True)
    ensure_files_in_colab()

    # --- Шаг 1. Калибровки (п1.4) ---
    k_pressure, b_pressure = calibrate_pressure()
    step_to_m = calibrate_distance()

    print("=== КАЛИБРОВКИ (п1.4) ===")
    print(f"k = {k_pressure:.4f} Па/отсчёт, b = {b_pressure:.1f} Па")
    print(f"1 шаг = {step_to_m:.2e} м ({step_to_m * 1e3:.4f} мм/шаг)")
    print()

    # --- Шаг 2. Профили скорости, центровка и расчёт расхода (п1.5 и п1.6) ---
    distances = sorted(SECTION_FILES.keys())
    profiles = {}
    outgoes_g_s = []

    print("=== РЕЗУЛЬТАТЫ ПО СЕЧЕНИЯМ (п1.5–п1.6) ===")
    for d in distances:
        filepath = find_file(SECTION_FILES[d])

        x_raw, u = calculate_velocity(filepath, k_pressure, step_to_m)
        x_center, fwhm_m = find_center_and_fwhm(x_raw, u, method=CENTER_METHOD)

        x_centered = x_raw - x_center - MANUAL_SHIFTS_MM.get(d, 0.0) * 1e-3

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
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for d in distances:
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
    save_and_show(fig, "velocity.png")

    # --- Шаг 5. График скоростей с расходами в легенде: velocity-outgo.png (п1.6) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for d in distances:
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
    save_and_show(fig, "velocity-outgo.png")

    # --- Шаг 6. График зависимости расхода от расстояния: outgo-distance.png (п1.6) ---
    dist_arr = np.array(distances, dtype=float)
    outgo_arr = np.array(outgoes_g_s, dtype=float)

    slope_q, intercept_q = np.polyfit(dist_arr, outgo_arr, 1)
    dist_line = np.linspace(dist_arr.min(), dist_arr.max(), 200)
    outgo_line = slope_q * dist_line + intercept_q

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
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
    save_and_show(fig, "outgo-distance.png")

    print(f"\nГотово! Все 5 графиков сохранены в папку: {os.path.abspath(PLOTS_PATH)}")


if __name__ == "__main__":
    main()