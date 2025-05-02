import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt


def wiener_proc(dt=0.1, x=0, n=1000):

    w = np.zeros(n+1)
    t = np.linspace(x, n, n+1)

    w[1:n+1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))

    return t, w


def plot_proc(t, w):
    plt.plot(t, w)
    plt.xlabel('Time(t)')
    plt.ylabel('Winer-process W(t)')
    plt.title('Winer-process')
    plt.show()


if __name__ == '__main__':

    time, data = wiener_proc()
    plot_proc(time, data)
