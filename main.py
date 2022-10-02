import random

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import qqqq
import reg

N = 1000
tr = 0.8
val = 0.1
random.seed()

def num2str(num, w):
    k = 1
    s = str(round(w[0], 2))
    for i in num:
        if i == 0:
            s += ' + ' + str(round(w[k], 2)) + ' * sin(x)'
        elif i == 1:
            s += ' + ' + str(round(w[k], 2)) + ' * cos(x)'
        elif i == 2:
            s += ' + ' + str(round(w[k], 2)) + ' * exp(x)'
        elif i >= 3 and i <= 12:
            s += ' + ' + str(round(w[k], 2)) + ' * x^' + str(i-2)
        else:
            s += ' + ' + str(round(w[k], 2)) + ' * x^' + str((i-11)*10)
        k += 1
        if k == 5:
            s+=" + ..."
            break
    return s


x = np.linspace(0, 1, N)
def z(x):
    return 20 * np.sin(6 * np.pi * x) + 100 * np.exp(x)
def t(x):
    return z(x) + np.random.randn(len(x)) * 10



def fffff(ii, x):
    return np.array([1] + [bas_fun[j](x) for j in top[ii][1]]).dot(top[ii][2])


ind_prm = np.random.permutation(x)
train_ind = ind_prm[:int(tr*N)]
valid_ind = ind_prm[int(tr*N):int((val+tr)*N)]
test_ind = ind_prm[int((val+tr)*N):]

x_train, x_valid, x_test = train_ind, valid_ind, test_ind
z_train, z_valid, z_test = z(x_train), z(x_valid), z(x_test)
t_train, t_valid, t_test = t(x_train), t(x_valid), t(x_test)

# bas_fun = [lambda x:np.sin(x), lambda x:np.cos(x), lambda x:np.log(x + 10**(-7)), lambda x:np.exp(x), lambda x:x**0.5, lambda x:x, lambda x:x**2, lambda x:x**3]
bas_fun = [lambda x:np.sin(x), lambda x:np.cos(x), lambda x:np.exp(x)] + [lambda x:x**i for i in range(1, 10)] + [lambda x:x**(i*10) for i in range(1, 11)]
lam = [0.000001, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 50, 100]#0.000001 чтобы убрать матрицы определитель которых 0

aAa = qqqq.Posl()
sSs = reg.Reg(x_train, t_train, len(x_train))
ind = []
top = []
while len(ind) < 31:
    d = aAa.kk(8, 17, len(bas_fun))
    if ind.count(d) == 0:
        ind.append(d)
for i in ind:
    lamba = aAa.kk(5, 5, len(lam))
    for ll in lamba:
        m = len(i)
        qweewq = [bas_fun[j] for j in i]
        w = sSs.fddkd(qweewq, m, lam[ll])
        e = sSs.err(w, t_valid, [bas_fun[j] for j in i], m, x_valid)
        if len(top) < 10:
            top.append([e, i, w, lam[ll]])
        else:
            top.append([e, i, w, lam[ll]])
            top.sort()
            top.pop(10)

fig = go.Figure()
fig.add_trace(go.Scatter(x=[num2str(i[1], i[2]) for i in top], y=[i[0] for i in top], mode='markers', name="t"))
erorr = np.array([sSs.err(top[i][2], t_test, [bas_fun[j] for j in top[i][1]], len(top[i][1]), x_test) for i in range(10)])
llam = np.array([i[3] for i in top])
customdata = np.stack((erorr, llam), axis=-1)
fig.update_traces(customdata=customdata, hovertemplate="y=%{x}<br>Error_val=%{y}<br>Error_test=%{customdata[0]}<br>lambda=%{customdata[1]}"+'<extra></extra>')
fig.update_layout(xaxis_title="function",
                  yaxis_title="error_valid",)
fig.show()
fig.write_html("top.html")


xxx = [num2str(i[1], i[2]) for i in top]
ff = go.Figure()
ff.add_trace(go.Scatter(x=x, y=z(x)))
ff.add_trace(go.Scatter(x=x, y=t(x), mode='markers'))
ff.add_trace(go.Scatter(x=x, y=fffff(0, x), name=xxx[0]))
ff.add_trace(go.Scatter(x=x, y=fffff(1, x), name=xxx[1]))
ff.add_trace(go.Scatter(x=x, y=fffff(2, x), name=xxx[2]))
ff.add_trace(go.Scatter(x=x, y=fffff(3, x), name=xxx[3]))
ff.add_trace(go.Scatter(x=x, y=fffff(4, x), name=xxx[4]))
ff.add_trace(go.Scatter(x=x, y=fffff(5, x), name=xxx[5]))
ff.add_trace(go.Scatter(x=x, y=fffff(6, x), name=xxx[6]))
ff.add_trace(go.Scatter(x=x, y=fffff(7, x), name=xxx[7]))
ff.add_trace(go.Scatter(x=x, y=fffff(8, x), name=xxx[8]))
ff.add_trace(go.Scatter(x=x, y=fffff(9, x), name=xxx[9]))
ff.show()
ff.write_html("fun.html")
