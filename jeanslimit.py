"""
Jeans Mass / Gas Cloud Collapse Simulator  (3D, SPH + self-gravity)
====================================================================
Two sliders control the cloud:
    * Cloud mass       (Msun, log scale)
    * Number density  (particles / cm^3, log scale)

Physics
-------
Jeans mass  M_J = (pi^(5/2) / 6) * c_s^3 / (G^(3/2) * rho^(1/2))
    M > M_J  ->  gravity beats thermal pressure  -> cloud collapses (star formation)
    M < M_J  ->  pressure wins                   -> cloud puffs up / disperses

The cloud is a uniform, initially-static sphere of N gas particles evolved with
    - softened Newtonian self-gravity
    - isothermal SPH pressure  (P = c_s^2 * rho)
    - Monaghan artificial viscosity

Units: length = pc, mass = Msun, velocity = km/s  (time unit = 0.978 Myr)

Run:   python jeans_collapse_sim.py          (needs numpy + matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------- constants
G = 4.30091e-3            # pc (km/s)^2 / Msun
K_B = 1.380649e-23        # J/K
M_H = 1.6735575e-27       # kg
MU = 2.33                 # mean molecular weight of molecular gas (H2 + He)
MSUN_G = 1.989e33         # g
PC_CM = 3.0857e18         # cm
TEMP = 10.0               # gas temperature in K  (typical molecular cloud)
N_PART = 350              # number of SPH particles
ETA = 1.3                 # smoothing-length factor
MYR_PER_TIME = 0.9778     # 1 (pc / km/s) in Myr


def sound_speed(T=TEMP):
    """Isothermal sound speed in km/s."""
    return np.sqrt(K_B * T / (MU * M_H)) / 1e3


def n_to_rho(n_cm3):
    """Number density [cm^-3] -> mass density [Msun/pc^3]."""
    rho_g_cm3 = n_cm3 * MU * M_H * 1e3
    return rho_g_cm3 * PC_CM ** 3 / MSUN_G


def jeans_mass(n_cm3, T=TEMP):
    """Jeans mass in Msun for number density n [cm^-3]."""
    cs = sound_speed(T)
    rho = n_to_rho(n_cm3)
    return np.pi ** 2.5 / 6.0 * cs ** 3 / (G ** 1.5 * np.sqrt(rho))


# ------------------------------------------------------------- SPH simulation
class GasCloud:
    def __init__(self, mass, n_cm3, npart=N_PART, T=TEMP, seed=1):
        rng = np.random.default_rng(seed)
        self.N = npart
        self.cs = sound_speed(T)
        self.M = mass
        self.rho0 = n_to_rho(n_cm3)
        self.R0 = (3 * mass / (4 * np.pi * self.rho0)) ** (1 / 3)
        self.m = mass / npart
        self.tff = np.sqrt(3 * np.pi / (32 * G * self.rho0))
        self.MJ = jeans_mass(n_cm3, T)

        # uniform sphere
        r = self.R0 * rng.random(npart) ** (1 / 3)
        v = rng.normal(size=(npart, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        self.pos = v * r[:, None]
        # tiny random thermal-like motion (5% of c_s) to seed perturbations
        self.vel = rng.normal(scale=0.05 * self.cs, size=(npart, 3))
        self.rho = np.full(npart, self.rho0)
        self.h = ETA * (self.m / self.rho) ** (1 / 3)
        self.t = 0.0
        self.acc = self.accel()

    # cubic spline kernel (3D)
    @staticmethod
    def W(r, h):
        q = r / h
        s = 1 / np.pi / h ** 3
        return s * np.where(q < 1, 1 - 1.5 * q ** 2 + 0.75 * q ** 3,
                            np.where(q < 2, 0.25 * (2 - q) ** 3, 0.0))

    @staticmethod
    def dW(r, h):
        q = r / h
        s = 1 / np.pi / h ** 4
        return s * np.where(q < 1, -3 * q + 2.25 * q ** 2,
                            np.where(q < 2, -0.75 * (2 - q) ** 2, 0.0))

    def accel(self):
        dx = self.pos[:, None, :] - self.pos[None, :, :]       # r_i - r_j
        r2 = np.sum(dx ** 2, axis=2)
        r = np.sqrt(r2) + 1e-12
        np.fill_diagonal(r, 1e-12)

        # density with per-particle smoothing length
        self.rho = np.maximum(self.m * self.W(r, self.h[:, None]).sum(axis=1), 1e-30)
        self.h = ETA * (self.m / self.rho) ** (1 / 3)

        hij = 0.5 * (self.h[:, None] + self.h[None, :])
        P = self.cs ** 2 * self.rho
        pterm = P / self.rho ** 2
        dwr = self.dW(r, hij)
        np.fill_diagonal(dwr, 0.0)

        # artificial viscosity
        dv = self.vel[:, None, :] - self.vel[None, :, :]
        vr = np.sum(dv * dx, axis=2)
        mu = np.where(vr < 0, hij * vr / (r2 + 0.01 * hij ** 2), 0.0)
        rho_bar = 0.5 * (self.rho[:, None] + self.rho[None, :])
        Pi = (-1.0 * self.cs * mu + 2.0 * mu ** 2) / rho_bar

        coef = self.m * (pterm[:, None] + pterm[None, :] + Pi) * dwr / r
        a_press = -np.sum(coef[:, :, None] * dx, axis=1)

        # softened self-gravity
        eps2 = (0.5 * hij) ** 2
        inv = 1.0 / (r2 + eps2) ** 1.5
        np.fill_diagonal(inv, 0.0)
        a_grav = -G * self.m * np.sum(inv[:, :, None] * dx, axis=1)
        return a_press + a_grav

    def step(self):
        vmax = np.max(np.linalg.norm(self.vel, axis=1))
        dt = min(0.02 * self.tff, 0.25 * self.h.min() / (self.cs + vmax))
        dt = max(dt, 2e-4 * self.tff)
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        self.acc = self.accel()
        self.vel += 0.5 * dt * self.acc
        self.t += dt

    # diagnostics
    def half_mass_radius(self):
        com = self.pos.mean(axis=0)
        return np.median(np.linalg.norm(self.pos - com, axis=1))

    def density_contrast(self):
        return self.rho.max() / self.rho0


# --------------------------------------------------------------------- GUI
def main():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Jeans Mass & Gas-Cloud Collapse  (3D SPH, T = %.0f K)" % TEMP,
                 fontsize=14, fontweight="bold")

    ax3d = fig.add_axes([0.03, 0.22, 0.50, 0.68], projection="3d")
    ax_j = fig.add_axes([0.60, 0.55, 0.36, 0.34])
    ax_r = fig.add_axes([0.60, 0.25, 0.36, 0.22])

    # ---- sliders (log10 scale) ----
    ax_sm = fig.add_axes([0.15, 0.10, 0.55, 0.03])
    ax_sn = fig.add_axes([0.15, 0.05, 0.55, 0.03])
    s_mass = Slider(ax_sm, "Cloud mass", -1, 3, valinit=np.log10(5.0), color="#e8833a")
    s_dens = Slider(ax_sn, "Density", 1, 6, valinit=np.log10(1e4), color="#3a8ee8")
    ax_btn = fig.add_axes([0.80, 0.06, 0.12, 0.06])
    btn = Button(ax_btn, "Restart")

    def set_labels():
        s_mass.valtext.set_text("%.2f Msun" % 10 ** s_mass.val)
        s_dens.valtext.set_text("%.1e cm$^{-3}$" % 10 ** s_dens.val)

    set_labels()

    state = {"cloud": None, "hist_t": [], "hist_r": [], "hist_rho": []}

    # ---- Jeans-mass diagram (static curve + moving point) ----
    ngrid = np.logspace(1, 6, 200)
    ax_j.loglog(ngrid, jeans_mass(ngrid), "k-", lw=2, label="Jeans mass")
    ax_j.fill_between(ngrid, jeans_mass(ngrid), 1e4, color="tomato", alpha=0.18)
    ax_j.fill_between(ngrid, 1e-2, jeans_mass(ngrid), color="skyblue", alpha=0.18)
    ax_j.text(2e1, 3e2, "COLLAPSE\n(M > M_J)", color="firebrick", fontsize=9)
    ax_j.text(1.5e4, 0.03, "STABLE\n(M < M_J)", color="navy", fontsize=9)
    ax_j.set_xlim(1e1, 1e6)
    ax_j.set_ylim(1e-2, 1e4)
    ax_j.set_xlabel("density n  [cm$^{-3}$]")
    ax_j.set_ylabel("mass  [Msun]")
    ax_j.set_title("Jeans mass diagram", fontsize=10)
    (pt,) = ax_j.plot([], [], "o", ms=11, mfc="gold", mec="k", zorder=5)

    # ---- time-evolution panel ----
    ax_r.set_xlabel("time  [Myr]")
    ax_r.set_ylabel("R / R$_0$")
    ax_r.set_yscale("log")
    ax_r.grid(alpha=0.3)
    (line_r,) = ax_r.plot([], [], "b-", label="half-mass radius")
    ax_r.legend(loc="upper left", fontsize=8)

    # ---- 3D scatter ----
    sc = ax3d.scatter([], [], [], c=[], cmap="turbo", s=16, vmin=0, vmax=2.5,
                      depthshade=False, edgecolors="k", linewidths=0.3)
    info = fig.text(0.03, 0.93, "", fontsize=10, family="monospace", va="top")
    ax3d.set_facecolor("white")
    ax3d.xaxis.set_pane_color((0, 0, 0, 0))
    ax3d.yaxis.set_pane_color((0, 0, 0, 0))
    ax3d.zaxis.set_pane_color((0, 0, 0, 0))

    def restart(_=None):
        M = 10 ** s_mass.val
        n = 10 ** s_dens.val
        state["cloud"] = GasCloud(M, n)
        state["hist_t"], state["hist_r"], state["hist_rho"] = [], [], []
        c = state["cloud"]
        lim = 1.6 * c.R0
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)
        ax3d.set_xlabel("x [pc]")
        ax3d.set_ylabel("y [pc]")
        ax3d.set_zlabel("z [pc]")
        pt.set_data([n], [M])
        ax_r.set_xlim(0, 3 * c.tff * MYR_PER_TIME)
        ax_r.set_ylim(0.01, 30)
        line_r.set_data([], [])

    def on_slider(_):
        set_labels()
        restart()

    s_mass.on_changed(on_slider)
    s_dens.on_changed(on_slider)
    btn.on_clicked(restart)
    restart()

    def update(_frame):
        c = state["cloud"]
        for _ in range(3):
            c.step()
        x, y, z = c.pos.T
        sc._offsets3d = (x, y, z)
        sc.set_array(np.log10(c.rho / c.rho0 + 1e-9) + 0.3)

        rr = c.half_mass_radius() / c.R0
        state["hist_t"].append(c.t * MYR_PER_TIME)
        state["hist_r"].append(max(rr, 1e-3))
        line_r.set_data(state["hist_t"], state["hist_r"])
        if state["hist_t"][-1] > ax_r.get_xlim()[1]:
            ax_r.set_xlim(0, state["hist_t"][-1] * 1.3)

        ratio = c.M / c.MJ
        contrast = c.density_contrast()
        if ratio > 1 and contrast > 30:
            status = "PROTOSTELLAR CORE FORMING"
        elif ratio > 1:
            status = "COLLAPSING"
        elif rr > 1.3:
            status = "DISPERSING (pressure wins)"
        else:
            status = "STABLE / OSCILLATING"
        info.set_text(
            f"M = {c.M:8.2f} Msun   M_J = {c.MJ:8.2f} Msun   M/M_J = {ratio:6.2f}\n"
            f"R0 = {c.R0:6.3f} pc   t_ff = {c.tff * MYR_PER_TIME:6.3f} Myr   "
            f"t = {c.t * MYR_PER_TIME:6.3f} Myr\n"
            f"peak density = {contrast:7.1f} x initial   -> {status}")
        return sc, line_r

    global anim  # keep a reference so it is not garbage collected
    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
