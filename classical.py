"""
Calculates the Expectation Value

Authors:
- Anthony Michalek <ianthisawesomee@gmail.com>

License: BSD style
Please feel free to use and modify this, but keep the above information.
"""

import numpy as np


class ClassicalApprox:
    def __init__(self, x_init, p_init, x, V_x, xmax, xmin, pmax, pmin, hbar, t_0=0.0, m=1):
        """
        Parameters
        -----------
        x_init : float
            The initial x value
        p_init: float
            The initial momentum value
        x : array_like, float
            Length-N array of evenly spaced spatial coordinates
        V_x : array_like, float
            Length-N array giving the potential at each x
        m : float
            Particle mass (default = 1)
        t_0 : float
            Initial time (default = 0)
        """

        self.hbar = hbar
        self.pmin = pmin
        self.pmax = pmax
        self.xmin = xmin
        self.xmax = xmax
        self.xa, self.V_x = map(np.asarray, (x, V_x))
        self.dx = self.xa[1] - self.xa[0]
        self.F_x = np.diff(a=self.V_x, append=[0.]) / self.dx
        self.m = m
        self.t = t_0
        self.x = x_init
        self.N = len(x)
        self.p = p_init

    def time_step(self, dt, Nsteps=1):
        assert Nsteps >= 0
        Nstep = Nsteps
        for num_iter in xrange(Nsteps):
            x_intermediate = self.x
            p_intermediate = self.p
            # print("old")
            # print(self.x)
            # print(self.p)
            # print(self.xmax)
            # print(self.xmin)
            # print(self.dx)
            # if (self.x - self.xmax) != 0:
            # print(abs(self.x - self.xmax) / self.dx)
            F_values = self.F_x[abs(self.xa - x_intermediate) < self.dx]
            if len(F_values) != 0:
                F_avg = sum(F_values) / len(F_values)
            else:
                F_avg = 0.0
            if F_avg != 0:
                print(F_avg)
            if (abs(x_intermediate - self.xmax) < (2. * self.dx)) or (F_avg < -100.):
                print("BONK R")
                self.p = -abs(self.p)
            if (abs(x_intermediate - self.xmin) < (2. * self.dx)) or (F_avg > 100.):
                print("BONK L")
                self.p = abs(self.p)
            if (abs(x_intermediate - self.xmax) >= (2. * self.dx)
                    and (abs(x_intermediate - self.xmin) >= (2. * self.dx))
                    and (-100 <= F_avg <= 100.)):
                self.p -= dt * F_avg
                if p_intermediate > self.pmax:
                    p_intermediate = self.pmax
                if p_intermediate < self.pmin:
                    p_intermediate = self.pmin
                if self.p > self.pmax:
                    self.p = self.pmax
                if self.p < self.pmin:
                    self.p = self.pmin
            self.x += dt * p_intermediate / self.m
            self.t += dt
            Nstep -= 1
            # print("new")
            # print(self.x)
            # print(self.p)
