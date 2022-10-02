import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class Reg(object):

    def __init__(self, x, t, N):
        self.N = N
        self.x = x
        self.t = t

    def fddkd(self, f, m, lamb):
        ddd = 0
        ddd += ddd
        f = [lambda x:x**0] + f
        F = np.array([f[j](self.x) for j in range(m + 1)]).T
        I = np.zeros((m+1, m+1))
        I[np.diag_indices(m+1)] = 1
        I[0][0] = 0
        w = ((np.linalg.inv(F.T.dot(F) + lamb * I)).dot(F.T)).dot(self.t)
        return w

    def err(self, w, t, f, m, xx):
        e = 0
        f = [lambda x: x**0] + f
        F = np.array([f[j](xx) for j in range(m + 1)]).T
        e = np.sum((t - F.dot(w.T)) ** 2) / 2
        return e
