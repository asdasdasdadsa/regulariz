import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class Reg(object):

    def __init__(self, x, t, N):
        self.N = N
        self.x = x
        self.t = t

    def fddkd(self, f, m):
        ddd = 0
        ddd += ddd
        f = [lambda x:1] + f
        F = np.array([[f[j](self.x[i]) for j in range(m + 1)] for i in range(self.N)])
        # F = np.array([f[j](self.x) for j in range(m + 1)]).T
        w = ((np.linalg.inv(F.T.dot(F))).dot(F.T)).dot(self.t)
        self.F = F
        return w

    def err(self, w, t, f, m):
        e = 0
        f = [lambda x: 1] + f
        for i in range(len(t)):
            fff = np.array([f[j](self.x[i]) for j in range(m + 1)])
            e += (t[i] - w.T.dot(fff)) ** 2
        e /= 2
        # e = np.sum((t - self.F.dot(w)) ** 2) / 2
        return e
