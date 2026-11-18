import sys
import time
import numpy as np
import jetFunctions as j

# --- Параметры профиля (меняйте перед каждым запуском) ---
filename = sys.argv[1] if len(sys.argv) > 1 else "profile_0mm.txt"  # говорящее имя
count = 60           # точек в профиле
stepsBetween = 28    # шагов между точками: 28 шагов = 1 мм
samples = 10         # отсчётов в одной точке, с паузой 0.05 с

rows = []

try:
    # Начало: двигатель и АЦП включают один раз
    j.initSpiAdc()
    j.initStepMotorGpio()

    for i in range(count):
        # Одна точка: прочитать samples отсчётов с паузой 0.05 с
        values = []
        for k in range(samples):
            values.append(j.getAdc())
            time.sleep(0.05)

        # Медиана не боится сбойных отсчётов АЦП
        rows.append((i * stepsBetween, np.median(values)))

        # Сдвиг и пауза, чтобы датчик отпустил давление
        j.stepForward(stepsBetween)
        time.sleep(0.2)

    # Возврат трубки в начало
    j.stepBackward(count * stepsBetween)

    # Сохранение: два столбца, по строке на точку
    np.savetxt(filename, np.array(rows),
               fmt=["%d", "%.1f"], header="step adc")
    print("Профиль сохранён в", filename)

finally:
    # Конец: выполнится даже при ошибке или Ctrl+C
    j.deinitSpiAdc()
    j.deinitStepMotorGpio()