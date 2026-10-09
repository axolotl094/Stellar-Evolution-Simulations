"""
Quantum Tunneling Simulator  (1D time-dependent Schrodinger equation)
=====================================================================
A Gaussian wave packet with fixed energy E hits a rectangular potential barrier.
Two sliders control the barrier:
    * Barrier height     V0   (energy units)
    * Barrier thickness  a    (length units)

Method: split-step Fourier (FFT) solution of  i dpsi/dt = [-1/2 d2/dx2 + V(x)] psi
Units : hbar = m = 1     ->   E = k0^2 / 2

Run:   python quantum_tunneling_sim.py       (needs numpy + matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------ simulation setup
L = 400.0                 # simulation box  [-L/2, L/2]
N = 4096                  # grid points
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
dx = x[1] - x[0]
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

K0 = 1.0                  # packet wavenumber  -> E = 0.5
E = 0.5 * K0 ** 2
SIGMA = 6.0               # packet width
X0 = -50.0                # starting position
DT = 0.05
STEPS_PER_FRAME = 4
T_END = 130.0             # stop time (packet fully past the barrier)

# soft absorbing edges so nothing wraps around the box
mask = np.ones(N)
edge = np.abs(x) > 170
mask[edge] = np.cos(0.5 * np.pi * (np.abs(x[edge]) - 170) / (L / 2 - 170)) ** 0.25

KINETIC = np.exp(-1j * 0.5 * k ** 2 * DT)


def barrier(V0, a):
    """Rectangular barrier of height V0 and thickness a centred at x = 0."""
    return np.where(np.abs(x) <= a / 2, V0, 0.0)


def transmission_theory(V0, a):
    """Exact plane-wave transmission coefficient at energy E."""
    if V0 < 1e-12:
        return 1.0
    if E < V0:
        kappa = np.sqrt(2 * (V0 - E))
        return 1.0 / (1.0 + V0 ** 2 * np.sinh(kappa * a) ** 2 / (4 * E * (V0 - E)))
    if abs(E - V0) < 1e-9:
        return 1.0 / (1.0 + 0.5 * E * a ** 2)
    kp = np.sqrt(2 * (E - V0))
    return 1.0 / (1.0 + V0 ** 2 * np.sin(kp * a) ** 2 / (4 * E * (E - V0)))


class Tunneling:
    def __init__(self, V0, a):
        self.V0, self.a = V0, a
        self.V = barrier(V0, a)
        self.half_pot = np.exp(-1j * self.V * DT / 2)
        psi = np.exp(-(x - X0) ** 2 / (4 * SIGMA ** 2)) * np.exp(1j * K0 * x)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        self.psi = psi
        self.t = 0.0

    def step(self):
        self.psi *= self.half_pot
        self.psi = np.fft.ifft(KINETIC * np.fft.fft(self.psi))
        self.psi *= self.half_pot * mask
        self.t += DT

    def probabilities(self):
        p = np.abs(self.psi) ** 2
        half = self.a / 2
        refl = p[x < -half].sum() * dx
        trans = p[x > half].sum() * dx
        inside = p[np.abs(x) <= half].sum() * dx
        return refl, inside, trans


# ------------------------------------------------------------------------- GUI
def main():
    fig = plt.figure(figsize=(12, 7.5), facecolor="white")
    fig.suptitle("Quantum Tunneling  -  wave packet vs. potential barrier",
                 fontsize=14, fontweight="bold")
    ax = fig.add_axes([0.08, 0.36, 0.88, 0.52])
    ax_v = ax.twinx()

    ax_sh = fig.add_axes([0.20, 0.17, 0.55, 0.03])
    ax_sa = fig.add_axes([0.20, 0.11, 0.55, 0.03])
    s_h = Slider(ax_sh, "Barrier height", 0.05, 2.0, valinit=0.8, color="#e8833a")
    s_a = Slider(ax_sa, "Barrier thickness", 0.2, 12.0, valinit=3.0, color="#3a8ee8")
    ax_btn = fig.add_axes([0.83, 0.11, 0.11, 0.09])
    btn = Button(ax_btn, "Restart")

    # wave function plot
    amp = np.sqrt(1.0 / (np.sqrt(2 * np.pi) * SIGMA))   # peak amplitude of psi
    ax.set_xlim(-100, 100)
    ax.set_ylim(-amp * 1.3, amp * 1.3)
    ax.set_xlabel("position x")
    ax.set_ylabel(r"wave function $\psi(x)$", color="navy")
    ax.axhline(0, color="gray", lw=0.6)
    (line_re,) = ax.plot([], [], color="royalblue", lw=1.6, label=r"Re $\psi$")
    (line_im,) = ax.plot([], [], color="darkorange", lw=1.6, label=r"Im $\psi$")
    (line_ab,) = ax.plot([], [], color="black", lw=1.2, ls="--", label=r"$|\psi|$")
    ax.legend(loc="upper right", fontsize=9)

    # barrier + energy plot (right axis)
    ax_v.set_ylim(0, 2.2)
    ax_v.set_ylabel("energy", color="firebrick")
    ax_v.axhline(E, color="green", ls="--", lw=1.5)
    ax_v.text(-98, E + 0.04, f"particle energy E = {E:.2f}", color="green", fontsize=9)
    barrier_fill = [None]

    info = fig.text(0.08, 0.27, "", fontsize=11, family="monospace", va="top")

    sim = {"s": None}

    def restart(_=None):
        V0, a = s_h.val, s_a.val
        sim["s"] = Tunneling(V0, a)
        if barrier_fill[0] is not None:
            barrier_fill[0].remove()
        barrier_fill[0] = ax_v.fill_between(x, 0, sim["s"].V, color="firebrick",
                                            alpha=0.35, step="mid")

    s_h.on_changed(restart)
    s_a.on_changed(restart)
    btn.on_clicked(restart)
    restart()

    def update(_):
        s = sim["s"]
        if s.t < T_END:
            for _ in range(STEPS_PER_FRAME):
                s.step()
        line_re.set_data(x, s.psi.real)
        line_im.set_data(x, s.psi.imag)
        line_ab.set_data(x, np.abs(s.psi))

        refl, inside, trans = s.probabilities()
        T_th = transmission_theory(s.V0, s.a)
        regime = "TUNNELING (E < V0)" if E < s.V0 else "over the barrier (E > V0)"
        state = "running" if s.t < T_END else "finished"
        info.set_text(
            f"t = {s.t:6.1f} ({state})   V0 = {s.V0:.2f}   thickness = {s.a:.2f}   {regime}\n"
            f"Reflected   = {refl * 100:6.2f} %\n"
            f"In barrier  = {inside * 100:6.2f} %\n"
            f"Transmitted = {trans * 100:6.2f} %      "
            f"(theory, plane wave: {T_th * 100:6.2f} %)")
        return line_re, line_im, line_ab

    global anim
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
