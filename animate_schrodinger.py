"""
Solve and animate the Schrodinger equation

First presented at http://jakevdp.github.com/blog/2012/09/05/quantum-python/

Authors:
- Jake Vanderplas <vanderplas@astro.washington.edu>
- Anthony Michalek (Real and Imaginary Parts Separated, Consistent Number of data points for both)
License: BSD
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from schrodinger import Schrodinger
from classical import ClassicalApprox


######################################################################
# Helper functions for gaussian wave-packets
def gauss_x(x, a, x0, p0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum p0
    """
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * p0))


def gauss_p(p, a, x0, p0):
    """
    analytical fourier transform of gauss_x(x), above
    """
    return ((a / np.sqrt(np.pi)) ** 0.5
            * np.exp(-0.5 * (a * (p - p0)) ** 2 - 1j * (p - p0) * x0))


######################################################################
# Utility functions for running the animation
def theta(x):
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y


def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))


######################################################################
# Create the animation

dt = 0.01
N_steps = 50
t_max = 120
frames = int(t_max / float(N_steps * dt))

# specify constants
hbar = 1.0   # planck's constant
m = 1.9      # particle mass

xmin = -100.
xmax = 100.
pmin = -5.
pmax = 5.
deltax = xmax - xmin
deltap = pmax - pmin

# specify range in x coordinate
N = int((deltax * deltap) / (2 * np.pi) + 2)
dx = deltax / N
x = dx * (np.arange(N) - 0.5 * N)

# specify potential
V0 = 1.5
L = hbar / np.sqrt(2 * m * V0)
a = 3 * L
x0 = -60 * L
V_x = square_barrier(x, a, V0)
V_x[x > xmax - .5 * deltax * 0.01] = 1E6
V_x[x < xmin + .5 * deltax * 0.01] = 1E6

# specify initial momentum and quantities derived from it
p0 = np.sqrt(2 * m * 0.2 * V0)
dp2 = p0 * p0 * 1. / 80
d = hbar / np.sqrt(2 * dp2)

p0 = p0 / hbar
v0 = p0 / m
psi_x0 = gauss_x(x, d, x0, p0)

# define the Classical Approximation object which calculates the approximation

C = ClassicalApprox(x_init=x0,
                    p_init=p0,
                    x=x,
                    V_x=V_x,
                    m=m,
                    xmax=xmax,
                    xmin=xmin,
                    pmax=pmax,
                    pmin=pmin,
                    hbar=hbar)


# define the Schrodinger object which performs the calculations about the quantum simulation
S = Schrodinger(x=x,
                psi_x0=psi_x0,
                V_x=V_x,
                hbar=hbar,
                m=m)

######################################################################
# Set up plot
fig = plt.figure(num="pySchrodinger by Jake Vanderplas and modified by Anthony Michalek")

# plotting limits
xlim = (xmin, xmax)
plim = (pmin, pmax)

# top axes show the x-space dat
ymin = -1
ymax = 1
ax1 = fig.add_subplot(211, xlim=xlim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_x_r_line, = ax1.plot([], [], c='b', label=r'$real(\psi(x))$')
psi_x_i_line, = ax1.plot([], [], c='g', label=r'$imag(\psi(x))$')
psi_x_line, = ax1.plot([], [], c='k', label=r'$|\psi(x)|^2$')
V_x_line, = ax1.plot([], [], c='k', label=r'$V(x)$')
center_line = ax1.axvline(0, c='k', ls=':', label=r"$x_{classical}$")

title = ax1.set_title("")
ax1.legend(prop=dict(size=12))
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$\psi(x)$')

# bottom axes show the p-space data
ymin = -1
ymax = 1

ax2 = fig.add_subplot(212, xlim=plim,
                      ylim=(ymin - 0.2 * (ymax - ymin),
                            ymax + 0.2 * (ymax - ymin)))
psi_p_r_line, = ax2.plot([], [], c='b', label=r'$real(\psi(p))$')
psi_p_i_line, = ax2.plot([], [], c='g', label=r'$imag(\psi(p))$')
psi_p_line, = ax2.plot([], [], c='k', label=r'$|\psi(p)|^2$')

pc_line1 = ax2.axvline(p0, c='k', ls=':', label=r'$\pm p_{classical}$')
ax2.legend(prop=dict(size=12))
ax2.set_xlabel('$p$')
ax2.set_ylabel(r'$\psi(p)$')

V_x_line.set_data(S.x, S.V_x)


######################################################################
# Functions to Animate the plot
def init():
    psi_x_line.set_data([], [])
    V_x_line.set_data([], [])
    center_line.set_data([], [])
    pc_line1.set_data([], [])

    psi_p_line.set_data([], [])
    title.set_text("")
    return psi_x_line, V_x_line, center_line, psi_p_line, pc_line1, title


def animate(i):
    S.time_step(dt, N_steps)
    C.time_step(dt, N_steps)
    # print(S.wf_norm(S.psi_x, S.dx))
    # print(S.wf_norm(S.psi_p, S.dp))
    psi_x_line.set_data(S.x, abs(S.psi_x) ** 2)
    psi_x_r_line.set_data(S.x, S.psi_x.real)
    psi_x_i_line.set_data(S.x, S.psi_x.imag)
    V_x_line.set_data(S.x, S.V_x)
    center_line.set_data(2 * [C.x], [0, 1])
    pc_line1.set_data(2 * [C.p], [0, 1])

    psi_p_line.set_data(S.p, abs(S.psi_p) ** 2)
    psi_p_r_line.set_data(S.p, S.psi_p.real)
    psi_p_i_line.set_data(S.p, S.psi_p.imag)
    # title.set_text("t = %.2f" % S.t)
    return (psi_x_r_line, psi_x_i_line, psi_x_line, V_x_line, center_line, psi_p_r_line,
            psi_p_i_line, psi_p_line, pc_line1, title)


# call the animator.
# blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=15, blit=True)

# uncomment the following line to save the video in mp4 format.  This
# requires either mencoder or ffmpeg to be installed on your system
# anim.save('schrodinger_barrier.htm', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
