import numpy as np


def interval_div(u_min: float, u_max: float, v_min: float, v_max: float) -> tuple[float, float]:
    # Вычисляет интервальное деление [u_min, u_max] / [v_min, v_max]
    
    d = [u_min / v_min, u_min / v_max, u_max / v_min, u_max / v_max]
    
    return min(d), max(d)


def row_intervals(a: float, b: float, delta: float, is_tomo: bool = True) -> tuple[float, float]:
    # Создает интервалы для элементов матрицы в зависимости от режима
    
    U = (a - delta, a + delta)
    V = (b - delta, b + delta) if is_tomo else (b, b)
    
    return U, V


def lambda_interval(matrix: np.ndarray[np.ndarray[float]], delta: float, is_tomo: bool) -> tuple[float, float] | None:
    a, b = matrix[:, 0], matrix[:, 1]

    _lambda_l, _lambda_r = -np.inf, np.inf
    
    for ai, bi in zip(a, b):
        U, V = row_intervals(ai, bi, delta, is_tomo)

        if V[0] <= 0:
            return None
        
        lo, hi = interval_div(*U, *V)
        _lambda_l, _lambda_r = max(_lambda_l, lo), min(_lambda_r, hi)
        
        if _lambda_l > _lambda_r:
            return None
        
    return _lambda_l, _lambda_r


def construct_matrix(matrix: np.ndarray[np.ndarray[float]], delta: float, _lambda: float, is_tomo: bool, tol: float = 1e-12) -> np.ndarray[float] | None:
    result = []
    
    for ai, bi in matrix:
        U, V = row_intervals(ai, bi, delta, is_tomo)
        
        WV = (
            min(_lambda * V[0], _lambda * V[1]),
            max(_lambda * V[0], _lambda * V[1])
        )

        lo, hi = max(U[0], WV[0]), min(U[1], WV[1])
        
        if lo > hi + tol:
            return None
        
        u = 0.5 * (lo + hi)
        v = min(max(u / _lambda if abs(_lambda) > tol else 0, V[0]), V[1])
        
        result.append([u, v])
    
    return np.array(result)


def delta_star(matrix: np.ndarray[np.ndarray[float]], delta_init: float = 0.05, delta_max: float = 1.0, delta_tol: float = 1e-4, is_tomo: bool = True):
    _lambda_rang = lambda_interval(matrix, 0, is_tomo)
    
    if _lambda_rang: 
        _lambda = _lambda_rang[0]
        return 0, construct_matrix(matrix, 0, _lambda, is_tomo), _lambda

    l, r = 0, delta_init
    
    while not lambda_interval(matrix, r, is_tomo) and r < delta_max:
        l, r = r, r * 2
    
    if r >= delta_max:
        return None, None, None

    while r - l > delta_tol:
        m = 0.5*(l + r)
        
        if lambda_interval(matrix, m, is_tomo):
            r = m
        else:
            l = m

    _lambda_rang = lambda_interval(matrix, r, is_tomo)
    
    if not _lambda_rang:
        return None, None, None
    
    _lambda = _lambda_rang[0]
    result = construct_matrix(matrix, r, _lambda, is_tomo)
    
    if result is None and len(_lambda_rang) > 1:
        _lambda = _lambda_rang[1]
        result = construct_matrix(matrix, r, _lambda, is_tomo)
    
    return r, result, _lambda



def main(matrix: np.ndarray[np.ndarray[float]]):
    for mode in [True, False]:
        print(f"{"TOMO" if mode else "REGRESS"}")

        delta, result, _lambda = delta_star(matrix, delta_init=0.05, delta_tol=1e-5, is_tomo=mode)
        
        print(f"{delta=:.6f}, {_lambda=:.6f}")
        
        print(result)


if __name__ == "__main__":
    _input_matrix = np.array([[0.95, 1.0], [1.05, 1.0], [1.10, 1.0]], float)
    
    main(_input_matrix)
