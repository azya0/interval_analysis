import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sympy import symbols, diff, solve, sin, cos, Abs
from scipy.optimize import fsolve

# -----------------------------
# 1. Функции и их производные
# -----------------------------
def f1(x):
    return x ** 3 - 3 * x ** 2 + 2

def f2(x):
    return math.e ** x - 2 * x - 1

def f1_derivative(x):
    return 3 * x ** 2 - 6 * x

def f2_derivative(x):
    return math.e ** x - 2

def f2_second_derivative(x):
    return math.e ** x

# -----------------------------
# Обоснование интервала для f2
# -----------------------------
print("Обоснование интервала X=[-1, 2] для f2:")
print("Экстремумы: решаем f2' = e^x - 2 = 0.")
guesses = np.linspace(-1, 1, 20)
crit = [fsolve(f2_derivative, g)[0] for g in guesses]
crit = np.unique(np.round(crit,3))
print("Приближенные критич. точки:", crit)
print("Точек перегиба нет: f''(x) = e^x > 0 для всех x")
print("[-1, 2] включает все критич. точки и перегибы, с глоб. max/min внутри.")

# -----------------------------
# 2. Точная область значений (улучшено)
# -----------------------------
def exact_range(func, func_deriv, a, b, num_points=100000):
    xs = np.linspace(a, b, num_points)
    ys = func(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    guesses = np.linspace(a, b, 20)
    crit_points = [fsolve(func_deriv, g)[0] for g in guesses]
    crit_points = np.unique(np.round(crit_points, 3))
    crit_vals = [func(c) for c in crit_points if a <= c <= b]
    crit_vals += [func(a), func(b)]
    min_y = min(min_y, min(crit_vals))
    max_y = max(max_y, max(crit_vals))
    return [min_y, max_y]

ran_f1 = exact_range(f1, f1_derivative, 0, 3)
ran_f2 = exact_range(f2, f2_derivative, -1, 2)
print("ran(f1, [0,3]):", ran_f1)
print("ran(f2, [-1, 2]):", ran_f2)

# -----------------------------
# 3. Интервальные операции
# -----------------------------
def interval_add(I1, I2):
    return [I1[0] + I2[0], I1[1] + I2[1]]
def interval_sub(I1, I2):
    return [I1[0] - I2[1], I1[1] - I2[0]]
def interval_mul(I1, I2):
    vals = [I1[0]*I2[0], I1[0]*I2[1], I1[1]*I2[0], I1[1]*I2[1]]
    return [min(vals), max(vals)]
def interval_pow(I, n):
    if n == 1: return I
    xs = [I[0]**n, I[1]**n]
    if n % 2 == 0 and I[0] < 0 < I[1]:
        return [0, max(xs)]
    return [min(xs), max(xs)]
def interval_sin(I):
    if I[1] - I[0] >= 2 * np.pi:
        return [-1, 1]
    xs = np.linspace(I[0], I[1], 1000)
    sin_vals = np.sin(xs)
    return [np.min(sin_vals), np.max(sin_vals)]

# -----------------------------
# 4. Методы интервальных оценок
# -----------------------------
def natural_extension_f1(I):
    x3 = interval_pow(I, 3)
    x2 = interval_pow(I, 2)
    three_x2 = interval_mul([3, 3], x2)
    return interval_add(interval_sub(x3, three_x2), [2, 2])

def natural_extension_f2(I):
    x5 = interval_pow(I, 5)
    five_x = interval_mul([5, 5], I)
    poly = interval_sub(x5, five_x)
    sinI = interval_sin(I)
    return interval_add(poly, sinI)

def horner_f1(I):
    x_minus_3 = interval_sub(I, [3, 3])
    t2 = interval_mul(x_minus_3, I)
    t3 = interval_mul(t2, I)
    return interval_add(t3, [2, 2])

def horner_f2(I):
    t1 = interval_pow(I, 2)  # x^2
    t2 = interval_pow(t1, 2)  # x^4
    t3 = interval_sub(t2, [5, 5])  # x^4 -5
    t4 = interval_mul(t3, I)  # x*(x^4 -5)
    sinI = interval_sin(I)
    return interval_add(t4, sinI)

def differential_centered(func, func_deriv, I, c):
    f_c = func(c)
    dI = [I[0] - c, I[1] - c]
    second_deriv = f2_second_derivative if func == f2 else (lambda x: 6*x -6)
    deriv_range = exact_range(func_deriv, second_deriv, I[0], I[1])
    prod = interval_mul(deriv_range, dI)
    return interval_add([f_c, f_c], prod)

def slope_centered(func, I, c):
    f_c = func(c)
    dI = [I[0] - c, I[1] - c]
    xs = np.linspace(I[0], I[1], 10000)  # increased points for precision
    xs = xs[xs != c]
    slopes = (func(xs) - f_c) / (xs - c)
    slope_interval = [np.min(slopes), np.max(slopes)]
    prod = interval_mul(slope_interval, dI)
    return interval_add([f_c, f_c], prod)

def interval_mid(I):
    return (I[0] + I[1]) / 2
def interval_rad(I):
    return (I[1] - I[0]) / 2
def cut(val, lo=-1, hi=1):
    return max(lo, min(hi, val))
def interval_intersection(I1, I2):
    left = max(I1[0], I2[0])
    right = min(I1[1], I2[1])
    if left <= right:
        return [left, right]
    else:
        return None

def bicentered_mv(func, func_deriv, I):
    second_deriv = f2_second_derivative if func == f2 else (lambda x: 6*x -6)
    deriv_interval = exact_range(func_deriv, second_deriv, I[0], I[1])
    mid_df = interval_mid(deriv_interval)
    rad_df = interval_rad(deriv_interval)
    p = cut(mid_df / rad_df) if rad_df != 0 else 0
    mid_x = interval_mid(I)
    rad_x = interval_rad(I)
    c_low = mid_x - p * rad_x
    c_high = mid_x + p * rad_x
    F_low = differential_centered(func, func_deriv, I, c_low)
    F_high = differential_centered(func, func_deriv, I, c_high)
    inter = interval_intersection(F_low, F_high)
    return inter if inter else F_low

def bicentered_scf(func, func_deriv, I):
    second_deriv = f2_second_derivative if func == f2 else (lambda x: 6*x -6)
    deriv_interval = exact_range(func_deriv, second_deriv, I[0], I[1])
    mid_df = interval_mid(deriv_interval)
    rad_df = interval_rad(deriv_interval)
    p = cut(mid_df / rad_df) if rad_df != 0 else 0
    mid_x = interval_mid(I)
    rad_x = interval_rad(I)
    c_low = mid_x - p * rad_x
    c_high = mid_x + p * rad_x
    F_low = slope_centered(func, I, c_low)
    F_high = slope_centered(func, I, c_high)
    inter = interval_intersection(F_low, F_high)
    return inter if inter else F_low

# -----------------------------
# 5. Липшиц и сравнение
# -----------------------------
def lipschitz_constant(func_deriv, I):
    if func_deriv == f1_derivative:
        return 9.0  # аналитически max |3x^2-6x| на [0,3] =9 при x=3
    else:
        xs = np.linspace(I[0], I[1], 1000)
        vals = np.abs(func_deriv(xs))
        return np.max(vals)

def hausdorff_dist(est, ran):
    return max(ran[0] - est[0], est[1] - ran[1])

def compare_with_lipschitz(I_est, ran_exact, L, X):
    rad_est = interval_rad(I_est)
    rad_true = interval_rad(ran_exact)
    haus = hausdorff_dist(I_est, ran_exact)
    bound = L * interval_rad(X)
    return [I_est, rad_est, rad_true, haus, bound]

# -----------------------------
# 6. Применение методов
# -----------------------------
def apply_with_centers(func, func_deriv, I, ran, L, X, name):
    nat = natural_extension_f1(I) if name == 'f1' else natural_extension_f2(I)
    horn = horner_f1(I) if name == 'f1' else horner_f2(I)
    centers = [I[0], interval_mid(I), I[1]]
    diff_ests = [differential_centered(func, func_deriv, I, c) for c in centers]
    slope_ests = [slope_centered(func, I, c) for c in centers]
    best_diff = diff_ests[1]  # mid
    best_slope = slope_ests[1]  # mid
    print(f"Для {name}: Diff centers {centers}: rads {[interval_rad(e) for e in diff_ests]} -> best {interval_rad(best_diff)}")
    print(f"Для {name}: Slope centers {centers}: rads {[interval_rad(e) for e in slope_ests]} -> best {interval_rad(best_slope)}")
    bic_mv = bicentered_mv(func, func_deriv, I)
    bic_scf = bicentered_scf(func, func_deriv, I)
    return nat, horn, best_diff, best_slope, bic_mv, bic_scf

F1_nat, F1_horner, F1_diff, F1_slope, F1_bic_mv, F1_bic_scf = apply_with_centers(f1, f1_derivative, [0,3], ran_f1, 0, [0,3], 'f1')
F2_nat, F2_horner, F2_diff, F2_slope, F2_bic_mv, F2_bic_scf = apply_with_centers(f2, f2_derivative, [-1, 2], ran_f2, 0, [-1, 2], 'f2')
L1 = lipschitz_constant(f1_derivative, [0, 3])
L2 = lipschitz_constant(f2_derivative, [-1.0, 2.0])
print("L1 (f1):", L1, "(аналитически max |f'| =9)")
print("L2 (f2):", L2)

# -----------------------------
# 7. Таблицы результатов
# -----------------------------
data_f1 = []
for nm, est in [("Естественное", F1_nat), ("Горнер (B.2)", F1_horner), ("Дифф. центр (best)", F1_diff),
                ("Наклонная центр (best)", F1_slope), ("Бицентрированная MV", F1_bic_mv), ("Бицентрированная SCF", F1_bic_scf)]:
    row = compare_with_lipschitz(est, ran_f1, L1, [0, 3])
    data_f1.append([nm] + row)
df_f1 = pd.DataFrame(data_f1, columns=["Метод", "Интервал", "rad_est", "rad_true", "Хаусдорф dist", "Bound Липшица"])

data_f2 = []
for nm, est in [("Естественное", F2_nat), ("Горнер (B.2)", F2_horner), ("Дифф. центр (best)", F2_diff),
                ("Наклонная центр (best)", F2_slope), ("Бицентрированная MV", F2_bic_mv), ("Бицентрированная SCF", F2_bic_scf)]:
    row = compare_with_lipschitz(est, ran_f2, L2, [-1.0, 2.0])
    data_f2.append([nm] + row)
df_f2 = pd.DataFrame(data_f2, columns=["Метод", "Интервал", "rad_est", "rad_true", "Хаусдорф dist", "Bound Липшица"])

print("\n=== f1(x) = x^3 - 3x^2 + 2 ===")
print(df_f1)
print("\n=== f2(x) = e^x - 2x - 2 ===")
print(df_f2)

# -----------------------------
# 8. Графики
# -----------------------------

xs = np.linspace(0, 3, 400)
plt.plot(xs, f1(xs), label="f1(x)")

xs = np.linspace(-1, 2, 400)
plt.plot(xs, f2(xs), label="f2(x)")

plt.legend()
plt.title("Графики функции f1 и f2")
plt.grid(True)
plt.show()
