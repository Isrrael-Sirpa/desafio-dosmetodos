import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def heun(f, t0, y0, h, n_steps):
    t = t0 + np.arange(n_steps+1)*h
    y = np.zeros(n_steps+1)
    y[0] = y0
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        y_pred = y[i] + h*k1
        k2 = f(t[i+1], y_pred)
        y[i+1] = y[i] + h*(k1 + k2)/2
    return t, y

def rk4(f, t0, y0, h, n_steps):
    t = t0 + np.arange(n_steps+1)*h
    y = np.zeros(n_steps+1)
    y[0] = y0
    for i in range(n_steps):
        ti, yi = t[i], y[i]
        k1 = f(ti, yi)
        k2 = f(ti + h/2, yi + h*k1/2)
        k3 = f(ti + h/2, yi + h*k2/2)
        k4 = f(ti + h,   yi + h*k3)
        y[i+1] = yi + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return t, y

# Ejercicio 1: Crecimiento logístico
f1 = lambda t, N: 0.000095 * N * (5000 - N)
t1_h, N1_h = heun(f1, 0, 100, 0.5, int(20/0.5))
t1_r, N1_r = rk4 (f1, 0, 100, 0.5, int(20/0.5))
df1 = pd.DataFrame({ 't': t1_h, 'Heun': N1_h, 'RK4': N1_r })

# Ejercicio 2: Crecimiento tumoral
f2 = lambda t, A: 0.8 * A * (1 - (A/60)**0.25)
t2_h, A2_h = heun(f2, 0, 1, 0.5, int(30/0.5))
t2_r, A2_r = rk4 (f2, 0, 1, 0.5, int(30/0.5))
df2 = pd.DataFrame({ 't': t2_h, 'Heun': A2_h, 'RK4': A2_r })

# Ejercicio 3: Velocidad con fricción
f3 = lambda t, v: (-5*9.81 + 0.05*v**2) / 5
t3_h, v3_h = heun(f3, 0, 0, 0.1, int(15/0.1))
t3_r, v3_r = rk4 (f3, 0, 0, 0.1, int(15/0.1))
df3 = pd.DataFrame({ 't': t3_h, 'Heun': v3_h, 'RK4': v3_r })

print("=== Ejercicio 1: Crecimiento logístico  ===")
print(df1.head(10).to_string(index=False))
print("\n=== Ejercicio 2: Crecimiento tumoral  ===")
print(df2.head(10).to_string(index=False))
print("\n=== Ejercicio 3: Velocidad con fricción  ===")
print(df3.head(10).to_string(index=False))

for df, title, fname in [
    (df1, 'Crecimiento logístico', 'logistico.png'),
    (df2, 'Crecimiento tumoral',   'tumoral.png'),
    (df3, 'Velocidad con fricción','friccion.png')
]:
    plt.figure()
    plt.plot(df['t'], df['Heun'], 'o--', label='Heun')
    plt.plot(df['t'], df['RK4'],  '-',  label='RK4')
    plt.title(title)
    plt.xlabel('t')
    plt.legend()
    plt.savefig(fname)
    plt.close()
    print(f"Gráfico guardado como {fname}")
