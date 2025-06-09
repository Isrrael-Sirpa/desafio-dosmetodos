import numpy as np
import pandas as pd

def heun(f, t0, y0, h, n_steps):
    t = t0 + np.arange(n_steps + 1) * h
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * (k1 + k2) / 2
    return t, y

def rk4(f, t0, y0, h, n_steps):
    t = t0 + np.arange(n_steps + 1) * h
    y = np.zeros(n_steps + 1)
    y[0] = y0
    for i in range(n_steps):
        ti, yi = t[i], y[i]
        k1 = f(ti, yi)
        k2 = f(ti + h / 2, yi + h * k1 / 2)
        k3 = f(ti + h / 2, yi + h * k2 / 2)
        k4 = f(ti + h, yi + h * k3)
        y[i + 1] = yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y

exercises = [
    {
        'name': 'Crecimiento logístico',
        'f': lambda t, y: 0.000095 * y * (5000 - y),
        't0': 0, 'y0': 100, 'T': 20, 'h_coarse': 0.5, 'h_ref': 0.01
    },
    {
        'name': 'Crecimiento tumoral',
        'f': lambda t, y: 0.8 * y * (1 - (y / 60) ** 0.25),
        't0': 0, 'y0': 1, 'T': 30, 'h_coarse': 0.5, 'h_ref': 0.01
    },
    {
        'name': 'Velocidad con fricción',
        'f': lambda t, y: (-5 * 9.81 + 0.05 * y**2) / 5,
        't0': 0, 'y0': 0, 'T': 15, 'h_coarse': 0.1, 'h_ref': 0.01
    },
]

rows = []
for ex in exercises:
    n_coarse = int(ex['T'] / ex['h_coarse'])
    n_ref = int(ex['T'] / ex['h_ref'])
    _, y_ref = rk4(ex['f'], ex['t0'], ex['y0'], ex['h_ref'], n_ref)
    factor = int(ex['h_coarse'] / ex['h_ref'])
    y_ref_sample = y_ref[::factor]
    
    _, y_h = heun(ex['f'], ex['t0'], ex['y0'], ex['h_coarse'], n_coarse)
    _, y_r = rk4(ex['f'], ex['t0'], ex['y0'], ex['h_coarse'], n_coarse)
    
    err_h = np.max(np.abs(y_h - y_ref_sample))
    err_r = np.max(np.abs(y_r - y_ref_sample))
    eval_h = 2 * n_coarse
    eval_r = 4 * n_coarse
    
    rows.append({
        'Ejercicio': ex['name'],
        'Paso h': ex['h_coarse'],
        'Error Máx Heun': err_h,
        'Error Máx RK4': err_r,
        'Eval f (Heun)': eval_h,
        'Eval f (RK4)': eval_r
    })

df = pd.DataFrame(rows)
print("Comparación de métodos (Parte 3)")
print(df)