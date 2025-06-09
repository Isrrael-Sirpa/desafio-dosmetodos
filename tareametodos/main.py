import numpy as np
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

def f1(t, N):
    k, Nm = 0.000095, 5000
    return k * N * (Nm - N)

t0, N0, T1, h1 = 0, 100.0, 20, 0.5
n1 = int((T1-t0)/h1)
t1_h, N1_h = heun(f1, t0, N0, h1, n1)
t1_r, N1_r = rk4 (f1, t0, N0, h1, n1)

def f2(t, A):
    α, k, ν = 0.8, 60, 0.25
    return α * A * (1 - (A/k)**ν)

t0, A0, T2, h2 = 0, 1.0, 30, 0.5
n2 = int((T2-t0)/h2)
t2_h, A2_h = heun(f2, t0, A0, h2, n2)
t2_r, A2_r = rk4 (f2, t0, A0, h2, n2)

def f3(t, v):
    m, g, kf = 5, 9.81, 0.05
    return (-m*g + kf*v**2) / m

t0, v0, T3, h3 = 0, 0.0, 15, 0.1
n3 = int((T3-t0)/h3)
t3_h, v3_h = heun(f3, t0, v0, h3, n3)
t3_r, v3_r = rk4 (f3, t0, v0, h3, n3)

plt.figure(figsize=(9, 7))

for idx, (t_h, y_h, t_r, y_r, title) in enumerate([
    (t1_h, N1_h, t1_r, N1_r, 'Logístico'),
    (t2_h, A2_h, t2_r, A2_r, 'Tumoral'),
    (t3_h, v3_h, t3_r, v3_r, 'Fricción')
]):
    plt.subplot(3,1,idx+1)
    plt.plot(t_h, y_h, 'o--', label='Heun')
    plt.plot(t_r, y_r, '-',  label='RK4')
    plt.title(title)
    plt.legend()

plt.tight_layout()
plt.show()
