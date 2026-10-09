"""
Type II (core-collapse) Supernova Simulator  ->  leaves a NEUTRON STAR
=====================================================================
A massive star (8-25 Msun) burns fuel in "onion" shells until its iron core
can no longer support itself.  The animation follows the whole story:

  1. PRE-SUPERNOVA  onion-skin star: H, He, C, Ne/O, Si burning ... Fe core
  2. COLLAPSE       Fe core falls in at up to ~0.25c (log-radius zoom, 3000 km -> 37 km)
  3. BOUNCE/STALL   core stiffens at nuclear density, a shock is born, then
                    STALLS at ~170 km (loses energy dissociating iron)
                    while neutrinos stream out and heat the "gain layer"
  4. REVIVAL        neutrino heating re-energises the shock -> explosion.
                    The proto-neutron star cools and shrinks to R ~ 12 km
  5. SHOCK CROSSING the shock ploughs through the star (asymmetric),
                    Fe-group + Ni-56 are made, then SHOCK BREAKOUT
  6. EXPANSION      layered, mixed ejecta expand; the neutron star gets a kick;
                    photosphere glow follows the plateau (H recombination)
                    -> drop -> Co-56 radioactive tail

Right-hand panels
  * shock radius & proto-neutron-star radius vs time after bounce
  * neutrino luminosity vs time (neutronization burst, accretion, cooling)

Sliders:  progenitor mass | explosion energy | Ni-56 mass

Run:  python type_ii_supernova_sim.py       (needs numpy + matplotlib)

Notes: the animation is schematic (radii are log-scaled / not to scale, the
core dynamics use analytic fits, not a hydrodynamics code).  Numbers are
chosen to match typical values from the literature.
"""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_rgba

# =========================================================== constants / setup
MSUN = 1.989e33
LSUN = 3.828e33
RSUN_CM = 6.957e10
EPS_NI, EPS_CO = 3.9e10, 6.78e9          # erg/s/g
TAU_NI, TAU_CO = 8.8, 111.3               # days

BG_STAR = "#05060f"                       # <- background of the two star views
_lum = np.dot(to_rgb(BG_STAR), [0.299, 0.587, 0.114])
TXT = "white" if _lum < 0.5 else "black"

# ---- animation timeline (frames) ----
F_PRE, F_COL, F_STALL, F_REV, F_SHK, F_BRK, F_EXP = 96, 70, 80, 90, 75, 18, 210
B1 = F_PRE
B2 = B1 + F_COL
B3 = B2 + F_STALL
B4 = B3 + F_REV
B5 = B4 + F_SHK
B6 = B5 + F_BRK
B7 = B6 + F_EXP
DAYS_PER_FRAME = 1.0
INTERVAL_MS = 30

# ---- core-collapse fit parameters ----
T_COL = 0.30          # s of collapse shown before bounce
R_IC0 = 800.0         # km, inner core radius at start of collapse
R_B = 37.0            # km, inner core / PNS radius at bounce
DT_GRID = 0.002
N_CORE = 480
B_LIM = 3.75          # axis limit of core panel (log-radius units)

T_GRID = np.linspace(0.0, 220.0, 2201)     # days (light curve)

# ---- star layers (display radius fractions) ----
LAYER_NAMES = ["Fe core  (-> neutron star)", "Si / S", "O / Ne / Mg", "C", "He", "H envelope"]
LAYER_COL = ["#b9bdd0", "#f2d24b", "#6fcf6f", "#4aa3e0", "#ff9f43", "#e0483e"]
NI_COL = "#ff4fd8"
EDGES = np.array([0.07, 0.15, 0.27, 0.42, 0.60, 1.00])
N_EJ = 4500
PHI1, PHI2 = 0.9, 2.3          # direction of the asymmetry (radians)
MIX_AMP = 5.0                  # strength of Rayleigh-Taylor style Ni mixing


# ============================================================ small helpers
def ssm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def rho_disp(r_km):
    """log-radius mapping used in the core panel (3 km -> 0, 3000 km -> 3)."""
    return np.log10(np.maximum(r_km, 3.0) / 3.0)


def bb_color(T):
    """Approximate RGB colour of a blackbody of temperature T [K]."""
    t = float(np.clip(T, 1000, 40000)) / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (99.4708025861 * np.log(t) - 161.1195681661 if t <= 66
         else 288.1221695283 * (t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255
    elif t <= 19:
        b = 0
    else:
        b = 138.5177312231 * np.log(t - 10) - 305.0447927307
    return tuple(np.clip([r, g, b], 0, 255) / 255.0)


def fmt_dur(yr):
    if yr >= 100:
        return f"{yr:.1e} yr"
    if yr >= 1:
        return f"{yr:.1f} yr"
    d = yr * 365.25
    if d >= 1:
        return f"{d:.1f} d"
    return f"{d * 24:.1f} h"


# ================================================================ physics model
def make_model(M, E51, m_ni):
    P = SimpleNamespace()
    P.M, P.E51, P.m_ni = M, E51, m_ni
    P.m_he = 0.31 * M - 0.65                       # helium-core mass
    P.m_fin = M * (1 - 0.01 * (M - 8))              # after wind mass loss
    P.m_fe = 1.30 + 0.012 * (M - 8)                 # iron-core mass
    P.m_ns = 1.25 + 0.025 * (M - 8)                 # neutron-star (gravitational) mass
    P.m_ej = P.m_fin - P.m_ns
    P.r_rsun = 300 + 30 * (M - 8)                   # red-supergiant radius
    P.v_ej = np.sqrt(2 * E51 * 1e51 / (P.m_ej * MSUN)) / 1e5     # km/s
    P.e_nu = 2.6e53 * (P.m_ns / 1.4) ** 2           # erg, total neutrino energy
    P.r_st = 170.0 * np.sqrt(P.m_ns / 1.4)          # km, stalled-shock radius
    P.t_rev = 0.45 / np.sqrt(E51) * (P.m_ns / 1.4)  # s, shock revival time
    P.v_f = 12000.0 * np.sqrt(E51)                  # km/s, shock speed after revival
    P.t_bo = P.r_rsun * RSUN_CM / (1.5 * P.v_ej * 1e5)   # s, breakout time
    P.t_end = P.t_rev + 1.6
    P.v_kick = 400.0 * np.sqrt(E51)                 # km/s

    # --- Type II-P light curve (Popov 1993 scalings + Ni-56 tail) ---
    t = T_GRID
    Lp = 1.64e42 * (P.r_rsun / 500) ** (2 / 3) * E51 ** (5 / 6) * (P.m_ej / 10) ** -0.5
    tp = 99.0 * (P.m_ej / 10) ** 0.5 * (P.r_rsun / 500) ** (1 / 6) * E51 ** (-1 / 6)
    rise = 1 - np.exp(-t / 1.2)
    shape = Lp * (1.08 - 0.22 * t / tp) + 0.8 * Lp * np.exp(-t / 2.5)   # cooling peak + plateau
    drop = 1.0 / (1.0 + np.exp((t - tp) / 4.0))
    ni = (m_ni * MSUN * (EPS_NI * np.exp(-t / TAU_NI) +
          EPS_CO * TAU_CO / (TAU_CO - TAU_NI) * (np.exp(-t / TAU_CO) - np.exp(-t / TAU_NI))))
    t0 = 120.0 * (P.m_ej / 10) / np.sqrt(E51)                              # gamma-ray trapping time
    trap = 1 - np.exp(-(t0 / np.maximum(t, 1e-3)) ** 2)
    P.L = np.maximum(shape * rise * drop + ni * trap * (1 - drop), 1e30)
    P.Lp, P.tp = Lp, tp
    return P


def r_pns(t):
    """Inner-core / proto-neutron-star radius [km]."""
    t = np.asarray(t, float)
    tau = np.clip((t + T_COL) / T_COL, 0, 1)
    pre = R_IC0 * (R_B / R_IC0) ** (tau ** 2)
    post = 12.0 + (R_B - 12.0) * np.exp(-np.maximum(t, 0) / 0.4)
    return np.where(t < 0, pre, post)


def r_shock(t, P):
    """Shock radius [km] (0 before bounce): launch, stall (+SASI wobble), revival."""
    t = np.asarray(t, float)
    tp = np.maximum(t, 0.0)
    stall = P.r_st * (1 + 0.12 * np.minimum(tp, P.t_rev) / P.t_rev)
    launch = stall - (stall - R_B) * np.exp(-tp / 0.03)
    wob = 1 + 0.03 * np.sin(2 * np.pi * tp / 0.06) * (1 - np.exp(-tp / 0.05)) * \
        np.exp(-np.maximum(tp - P.t_rev, 0) / 0.1)
    s = np.maximum(tp - P.t_rev, 0)
    out = launch * wob + P.v_f * (s - 0.15 * (1 - np.exp(-s / 0.15)))
    return np.where(t < 0, 0.0, out)


def lnu(t, P):
    """Neutrino luminosity [erg/s]."""
    t = np.maximum(np.asarray(t, float), 1e-4)
    burst = 3.5e53 * np.exp(-((t - 0.004) / 0.003) ** 2)          # neutronization burst
    acc = 6e52 * (P.m_ns / 1.4) * np.exp(-t / 0.35)               # accretion phase
    cool = P.e_nu / 5.0 * np.exp(-t / 5.0)                        # Kelvin-Helmholtz cooling
    return burst + acc + cool


def t_phot(t, tp):
    """Photospheric temperature [K] of a Type II-P supernova."""
    base = 5800 - 400 * min(t / tp, 1.0) + 25000 * np.exp(-t / 2.0)
    return base * (1 - 0.40 / (1 + np.exp(-(t - tp) / 6.0)))


# ------------------------------------------------------------ particle sets
def build_core(P, seed=5):
    """Infalling shells for the core panel, pre-computed on a time grid."""
    rng = np.random.default_rng(seed)
    N = N_CORE
    r0 = np.exp(rng.uniform(np.log(700), np.log(4200), N))
    th = rng.uniform(0, 2 * np.pi, N)
    tff = T_COL * (r0 / 1300.0) ** 1.5
    tg = np.arange(-T_COL, P.t_end, DT_GRID)
    Nt = len(tg)
    tau = np.clip((tg[None, :] + T_COL) / tff[:, None], 0, 1)
    pos = r0[:, None] * (1 - tau) ** (2 / 3)
    Rp, Rs = r_pns(tg), r_shock(tg, P)
    pos = np.maximum(pos, Rp[None, :])
    hit = pos <= Rp[None, :] * 1.001
    thit = np.where(hit.any(1), hit.argmax(1), Nt)
    cross = (pos <= Rs[None, :]) & (tg[None, :] >= 0)
    tc = np.where(cross.any(1), cross.argmax(1), Nt)
    f_acc = 0.35 + 0.5 * np.clip((P.m_ns - 1.25) / 0.45, 0, 1)
    acc = rng.random(N) < f_acc
    acc &= tg[np.minimum(tc, Nt - 1)] < P.t_rev
    base = np.array([to_rgba("#b9bdd0") if r < 1300 else
                     to_rgba("#f2d24b") if r < 2400 else to_rgba("#6fcf6f") for r in r0])
    return dict(tg=tg, r0=r0, th=th, pos=pos, tc=tc, thit=thit, acc=acc,
                dwell=rng.uniform(0.04, 0.30, N), w=rng.uniform(0.45, 0.95, N),
                ph=rng.uniform(0, 2 * np.pi, N), base=base)


def build_ejecta(P, seed=11):
    """2-D cross-section of the star for the big view."""
    rng = np.random.default_rng(seed)
    N = N_EJ
    r = np.sqrt(rng.random(N))
    th = rng.uniform(0, 2 * np.pi, N)
    layer = np.minimum(np.searchsorted(EDGES, r), 5)
    f_ni = np.clip(P.m_ni / 0.15, 0.08, 1.0)
    r_cut = np.sqrt(EDGES[0] ** 2 + f_ni * (EDGES[1] ** 2 - EDGES[0] ** 2))
    ni = (layer == 1) & (r < r_cut)
    g = 1 + 0.12 * np.cos(th - PHI1) + 0.05 * np.cos(2 * (th - PHI2))
    ks = np.arange(3, 11)
    amp = rng.uniform(0.3, 1.0, len(ks)) / ks ** 0.3
    phs = rng.uniform(0, 2 * np.pi, len(ks))
    tt = np.linspace(0, 2 * np.pi, 720)
    Fmax = sum(a * np.cos(k * tt + p) for a, k, p in zip(amp, ks, phs)).max()
    Fp = sum(a * np.cos(k * th + p) for a, k, p in zip(amp, ks, phs)) / Fmax
    Pp = np.clip(Fp, 0, 1) ** 1.5
    W = np.select([ni, layer == 1, layer == 2], [1.0, 0.7, 0.35], 0.0)
    base = np.array([to_rgba(c) for c in LAYER_COL])[layer]
    return dict(r=r, th=th, pos0=np.c_[r * np.cos(th), r * np.sin(th)], layer=layer,
                ni=ni, parr=r / g, mixw=W * Pp, base=base)


# ============================================================================ GUI
def main():
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    fig.suptitle("Type II Core-Collapse Supernova  -  a massive star dies and leaves a neutron star",
                 fontsize=14, fontweight="bold")

    axA = fig.add_axes([0.02, 0.36, 0.34, 0.54], facecolor=BG_STAR)
    axB = fig.add_axes([0.395, 0.47, 0.33, 0.43], facecolor=BG_STAR)
    axS = fig.add_axes([0.435, 0.15, 0.26, 0.25])
    axN = fig.add_axes([0.775, 0.17, 0.21, 0.55])

    # ------------------------------------------------------------- star view
    axA.set_aspect("equal")
    axA.set_xticks([])
    axA.set_yticks([])
    circles = []
    for i in range(6):
        c = Circle((0, 0), EDGES[i], fc=LAYER_COL[i], ec="none", zorder=2 + (5 - i))
        axA.add_patch(c)
        circles.append(c)
    hl = Circle((0, 0), 0.5, fill=False, ec="white", lw=2.0, zorder=12)
    axA.add_patch(hl)
    glow = Circle((0, 0), 1.0, fc="white", ec="none", alpha=0.0, zorder=1)
    axA.add_patch(glow)
    flash = Circle((0, 0), 0.1, fc="white", ec="none", alpha=0.0, zorder=14)
    axA.add_patch(flash)
    partsA = axA.scatter([], [], s=5, zorder=6, edgecolors="none")
    shock_line, = axA.plot([], [], color="white", lw=1.6, zorder=13)
    nsA = axA.scatter([0], [0], s=28, c="#7fe7ff", zorder=15, edgecolors="white", linewidths=0.6)
    beamsA = [axA.plot([], [], color="#7fe7ff", lw=1.5, zorder=15)[0] for _ in range(2)]
    nuA = axA.scatter([], [], s=6, c="#66f0ff", zorder=10, edgecolors="none")
    bannerA = axA.text(0.5, 0.965, "", transform=axA.transAxes, ha="center", va="top",
                       color=TXT, fontsize=12, fontweight="bold", zorder=20)
    timeA = axA.text(0.03, 0.03, "", transform=axA.transAxes, ha="left", va="bottom",
                     color=TXT, fontsize=9.5, family="monospace", zorder=20)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=n, ms=6)
               for n, c in zip(LAYER_NAMES, LAYER_COL)]
    handles.insert(2, Line2D([], [], marker="o", ls="", color=NI_COL, label="Ni-56 (made in explosion)", ms=6))
    leg = axA.legend(handles=handles, loc="lower right", fontsize=7.3, framealpha=0.55,
                     facecolor=BG_STAR, edgecolor="none")
    for tx in leg.get_texts():
        tx.set_color(TXT)
    axA.set_xlim(-1.35, 1.35)
    axA.set_ylim(-1.35, 1.35)

    # ------------------------------------------------------------- core view
    axB.set_aspect("equal")
    axB.set_xticks([])
    axB.set_yticks([])
    axB.set_xlim(-B_LIM, B_LIM)
    axB.set_ylim(-B_LIM, B_LIM)
    axB.set_title("Core zoom (log radius scale)", fontsize=10)
    for rk, lab in [(10, "10 km"), (30, "30"), (100, "100"), (300, "300"),
                    (1000, "1000"), (3000, "3000"), (10000, "10 000 km")]:
        axB.add_patch(Circle((0, 0), rho_disp(rk), fill=False, ec="#5a6070", lw=0.7, ls=":", zorder=2))
        axB.text(0.02, rho_disp(rk) + 0.03, lab, color="#8a90a0", fontsize=6.5, zorder=2)
    gain_out = Circle((0, 0), 1, fc="#ff8a1f", ec="none", alpha=0.20, zorder=1)
    gain_in = Circle((0, 0), 1, fc=BG_STAR, ec="none", zorder=1.5)
    axB.add_patch(gain_out)
    axB.add_patch(gain_in)
    pns = Circle((0, 0), 1, fc="gold", ec="white", lw=1.0, zorder=8)
    axB.add_patch(pns)
    partsB = axB.scatter([], [], s=7, zorder=6, edgecolors="none")
    nuB = axB.scatter([], [], s=4, c="#66f0ff", zorder=5, edgecolors="none")
    shock_B, = axB.plot([], [], color="#ffe066", lw=1.8, zorder=7)
    beamsB = [axB.plot([], [], color="#7fe7ff", lw=2.2, zorder=9, alpha=0.9)[0] for _ in range(2)]
    timeB = axB.text(0.03, 0.97, "", transform=axB.transAxes, color=TXT, va="top",
                     fontsize=10, family="monospace", zorder=20)
    capB = axB.text(0.5, 0.02, "", transform=axB.transAxes, color="#ffd27f", ha="center",
                    va="bottom", fontsize=8.6, zorder=20)
    nsB = axB.text(0.97, 0.97, "", transform=axB.transAxes, color="#7fe7ff", ha="right", va="top",
                   fontsize=9, family="monospace", zorder=20)

    # -------------------------------------------------- shock radius panel
    axS.set_yscale("log")
    axS.set_ylim(5, 5e4)
    axS.set_xlabel("time after core bounce  [s]", fontsize=9)
    axS.set_ylabel("radius [km]", fontsize=9)
    axS.set_title("Shock & proto-neutron-star radius", fontsize=10)
    axS.grid(alpha=0.3, which="both")
    axS.tick_params(labelsize=8)
    lnS_pns, = axS.plot([], [], color="royalblue", lw=2, label="inner core / PNS")
    lnS_sh, = axS.plot([], [], color="crimson", lw=2, label="shock front")
    vB = axS.axvline(0, color="gray", ls="--", lw=1)
    vR = axS.axvline(0.5, color="darkorange", ls="--", lw=1)
    txB = axS.text(0.01, 8, "bounce", fontsize=8, color="gray", rotation=90, va="bottom")
    txR = axS.text(0.5, 8, " revival", fontsize=8, color="darkorange", rotation=90, va="bottom")
    txSt = axS.text(0.5, 0.60, "STALL", transform=axS.transAxes, fontsize=8, color="crimson")
    dotS, = axS.plot([], [], "o", ms=9, mfc="gold", mec="k", zorder=5)
    axS.legend(loc="upper left", fontsize=8)

    # ------------------------------------------------------- neutrino panel
    axN.set_xscale("log")
    axN.set_yscale("log")
    axN.set_xlim(1e-3, 10)
    axN.set_ylim(1e50, 1e54)
    axN.set_xlabel("time after bounce  [s]", fontsize=9)
    axN.set_ylabel("neutrino luminosity [erg/s]", fontsize=9)
    axN.set_title("Neutrino luminosity", fontsize=10)
    axN.grid(alpha=0.3, which="both")
    axN.tick_params(labelsize=8)
    axN.text(0.0015, 2.5e53, "neutronization\nburst", fontsize=7.5, color="teal", va="top")
    axN.text(0.12, 1.5e53, "accretion", fontsize=7.5, color="teal")
    axN.text(1.6, 3e52, "cooling", fontsize=7.5, color="teal")
    lnN, = axN.plot([], [], color="teal", lw=2)
    dotN, = axN.plot([], [], "o", ms=9, mfc="gold", mec="k", zorder=5)
    enuT = axN.text(0.03, 0.05, "", transform=axN.transAxes, fontsize=8.5, color="teal")

    # --------------------------------------------------- text, sliders, buttons
    info = fig.text(0.02, 0.335, "", fontsize=8.4, family="monospace", va="top")
    s_m = Slider(fig.add_axes([0.11, 0.085, 0.24, 0.022]), "Progenitor mass", 8, 25,
                 valinit=15, color="#e8833a")
    s_e = Slider(fig.add_axes([0.11, 0.052, 0.24, 0.022]), "Explosion energy", 0.3, 3.0,
                 valinit=1.0, color="#3a8ee8")
    s_n = Slider(fig.add_axes([0.11, 0.019, 0.24, 0.022]), "Ni-56 mass", 0.005, 0.30,
                 valinit=0.07, color="#c04fd0")
    btn_r = Button(fig.add_axes([0.80, 0.05, 0.08, 0.055]), "Restart")
    btn_p = Button(fig.add_axes([0.89, 0.05, 0.08, 0.055]), "Pause")

    def set_labels():
        s_m.valtext.set_text("%.1f Msun" % s_m.val)
        s_e.valtext.set_text("%.2f foe" % s_e.val)
        s_n.valtext.set_text("%.3f Msun" % s_n.val)

    S = {"f": 0, "paused": False, "P": None, "core": None, "ej": None}
    rng_nu = np.random.default_rng(2)
    nu_ang = rng_nu.uniform(0, 2 * np.pi, 90)
    nu_ph = rng_nu.random(90)
    nuA_ang = rng_nu.uniform(0, 2 * np.pi, 70)
    nuA_ph = rng_nu.random(70)

    # ------------------------------------------------------------- rebuild
    def rebuild():
        P = make_model(s_m.val, s_e.val, s_n.val)
        S["P"] = P
        S["core"] = build_core(P)
        S["ej"] = build_ejecta(P)
        # shock panel
        ts = np.linspace(-T_COL, P.t_end, 800)
        lnS_pns.set_data(ts, r_pns(ts))
        lnS_sh.set_data(ts[ts >= 0], r_shock(ts[ts >= 0], P))
        axS.set_xlim(-T_COL, P.t_end)
        vR.set_xdata([P.t_rev, P.t_rev])
        txR.set_position((P.t_rev, 8))
        txSt.set_position((0.12 + 0.6 * P.t_rev / P.t_end * 0.5, 0.60))
        txB.set_position((0.005, 8))
        # neutrino panel
        tn = np.logspace(-3, 1, 400)
        lnN.set_data(tn, lnu(tn, P))
        enuT.set_text("E_nu = %.1e erg (~99%% of the energy)" % P.e_nu)

    def on_slider(_):
        set_labels()
        rebuild()
        S["f"] = B1

    def restart(_=None):
        rebuild()
        S["f"] = 0

    def toggle_pause(_):
        S["paused"] = not S["paused"]
        btn_p.label.set_text("Play" if S["paused"] else "Pause")

    s_m.on_changed(on_slider)
    s_e.on_changed(on_slider)
    s_n.on_changed(on_slider)
    btn_r.on_clicked(restart)
    btn_p.on_clicked(toggle_pause)
    set_labels()
    rebuild()

    theta_line = np.linspace(0, 2 * np.pi, 181)

    # ---------------------------------------------------------- frame logic
    def phase_of(f):
        if f < B1:
            return 0, f / F_PRE
        if f < B2:
            return 1, (f - B1) / F_COL
        if f < B3:
            return 2, (f - B2) / F_STALL
        if f < B4:
            return 3, (f - B3) / F_REV
        if f < B5:
            return 4, (f - B4) / F_SHK
        if f < B6:
            return 5, (f - B5) / F_BRK
        return 6, min((f - B6) / F_EXP, 1.0)

    def core_time(ph, p, P):
        if ph == 0:
            return -T_COL
        if ph == 1:
            return -T_COL + T_COL * p
        if ph == 2:
            return P.t_rev * p ** 1.5
        if ph == 3:
            return P.t_rev + 1.6 * p
        return P.t_end

    # ------------------------------------------------------------ star view
    def draw_A(ph, p, f, P):
        E = S["ej"]
        lines_time = ""
        for c in circles:
            c.set_visible(False)
        hl.set_visible(False)
        flash.set_alpha(0.0)
        glow.set_alpha(0.0)
        shock_line.set_data([], [])
        nuA.set_offsets(np.empty((0, 2)))
        partsA.set_visible(False)
        nsA.set_visible(ph >= 2)
        for b in beamsA:
            b.set_data([], [])
        lim = 1.35
        banner, banner_col = "", TXT

        if ph == 0:                                      # ---- onion star forms
            s = p * 6.0
            scale = np.interp(s, [0, 1, 2, 6], [0.30, 0.62, 1.0, 1.0])
            thr = [5.6, 4.95, 3.0, 2.0, 1.0, 0.0]
            for i, c in enumerate(circles):
                a = float(np.clip((s - thr[i]) / 0.35, 0, 1))
                c.set_visible(a > 0)
                c.set_alpha(a)
                c.set_radius(EDGES[i] * scale)
            k = min(int(s), 5)
            idx = [5, 4, 3, 2, 2, 1][k]
            if s > 5.7:
                idx = 0
            hl.set_visible(True)
            hl.set_radius(EDGES[idx] * scale * 0.98)
            hl.set_alpha(0.6 + 0.4 * np.sin(f * 0.5))
            names = ["HYDROGEN burning (main sequence)", "HELIUM burning (red giant)",
                     "CARBON burning", "NEON burning", "OXYGEN burning", "SILICON burning"]
            banner = names[k]
            dur = [d for _, d in stage_table(P)]
            u = s - k
            rem = dur[k] * (1 - u) + sum(dur[k + 1:]) + 0.3 / 3.156e7
            lines_time = "collapse in  T - " + fmt_dur(rem)
            nsA.set_visible(False)
        elif ph == 1:                                    # ---- collapse begins
            for i, c in enumerate(circles):
                c.set_visible(True)
                c.set_alpha(1.0)
                c.set_radius(EDGES[i])
            circles[0].set_radius(EDGES[0] * (1 - 0.8 * p ** 2))
            glow.set_alpha(0.25 + 0.25 * np.sin(f))
            glow.set_radius(EDGES[0] * 1.6)
            glow.set_facecolor("#ff5533")
            banner, banner_col = "IRON CORE COLLAPSES  (outer star doesn't know yet)", "#ff9a7a"
            lines_time = "core: %.0f ms before bounce" % (T_COL * 1000 * (1 - p))
        elif ph in (2, 3):
            for i, c in enumerate(circles):
                c.set_visible(i >= 1)
                c.set_alpha(1.0)
                c.set_radius(EDGES[i])
            tb = P.t_rev * p ** 1.5 if ph == 2 else P.t_rev + 1.6 * p
            if ph == 2 and p < 0.18:
                flash.set_alpha(0.9 * (1 - p / 0.18))
                flash.set_radius(0.08 + 0.35 * p / 0.18)
                flash.set_facecolor("white")
            banner = "CORE BOUNCE - NEUTRINO BURST" if ph == 2 else "SHOCK REVIVED -> EXPLOSION BEGINS"
            banner_col = "#7fe7ff" if ph == 2 else "#ffe066"
            lines_time = "t = +%.0f ms after bounce" % (tb * 1000)
            ph_nu = (nuA_ph + 0.035 * f) % 1.0
            rr = ph_nu * 1.3
            nuA.set_offsets(np.c_[rr * np.cos(nuA_ang), rr * np.sin(nuA_ang)])
        else:
            # ---- shock crossing / breakout / expansion: particle view
            partsA.set_visible(True)
            if ph == 4:
                Rs = p * 1.25
                w = ssm((Rs - E["parr"]) / 0.2)
                pos = E["pos0"] * (1 + 0.18 * w)[:, None]
                shocked = w > 0.35
                a_c = 0.85 - 0.57 * ssm(p / 0.2)
                for i, c in enumerate(circles):
                    c.set_visible(i >= 1)
                    c.set_alpha(a_c)
                    c.set_radius(EDGES[i])
                Rg = np.clip(Rs * (1 + 0.12 * np.cos(theta_line - PHI1) +
                                   0.05 * np.cos(2 * (theta_line - PHI2))), 0, 1.3)
                shock_line.set_data(Rg * np.cos(theta_line) * 1.0, Rg * np.sin(theta_line) * 1.0)
                shock_line.set_alpha(0.9 if p < 0.98 else 0.0)
                hrs = P.t_bo * (max(p, 1e-3) ** 1.2) / 3600.0
                banner, banner_col = "SHOCK WAVE CROSSES THE STAR", "#ffe066"
                lines_time = "t = +%.1f hours after bounce" % hrs
                alpha_p = float(ssm(p / 0.2))
                ph_nu = None
            elif ph == 5:
                pos = E["pos0"] * 1.18
                shocked = np.ones(N_EJ, bool)
                a_c = 0.28 * (1 - p)
                for i, c in enumerate(circles):
                    c.set_visible(i >= 1 and a_c > 0.01)
                    c.set_alpha(a_c)
                    c.set_radius(EDGES[i])
                flash.set_alpha(0.9 * (1 - p))
                flash.set_radius(1.15 + 0.6 * p)
                flash.set_facecolor("#bfe0ff")
                lim = 1.35 + 0.35 * p
                banner, banner_col = "SHOCK BREAKOUT  -  first light (UV / X-ray flash)", "#bfe0ff"
                lines_time = "t = +%.1f hours (breakout at %.1f h)" % (P.t_bo / 3600.0, P.t_bo / 3600.0)
                alpha_p = 1.0
            else:
                t = p * (F_EXP * DAYS_PER_FRAME) if False else (f - B6) * DAYS_PER_FRAME
                X = 1.18 + 5 * (1 - np.exp(-t / 6.0)) + 0.035 * t
                grow = 1 - np.exp(-t / 20.0)
                mult = 1 + MIX_AMP * grow * E["mixw"]
                pos = E["pos0"] * (X * mult)[:, None]
                shocked = np.ones(N_EJ, bool)
                lim = max(1.35, 1.32 * X)
                i = min(int(t / (T_GRID[1] - T_GRID[0])), len(T_GRID) - 1)
                L = P.L[i]
                T = t_phot(t, P.tp)
                r_ph = 0.95 * (1 - 0.75 * ssm((t - P.tp) / 30.0))
                glow.set_visible(True)
                glow.set_facecolor(bb_color(T))
                glow.set_radius(max(r_ph * X, 0.05))
                glow.set_alpha(float(np.clip(0.55 * np.sqrt(L / P.Lp), 0.06, 0.55)))
                lines_time = "t = +%.0f days" % t
                banner, banner_col = "EXPANDING EJECTA (cross-section)", TXT
                alpha_p = 1.0
                # neutron star + kick + pulsar beams
                kd = np.array([np.cos(PHI1), np.sin(PHI1)])
                off = -kd * 0.16 * X * (P.v_kick / 400.0) * min(t / 100.0, 1.6)
                nsA.set_offsets([off])
                ang = 0.45 * f
                for j, b in enumerate(beamsA):
                    a = ang + j * np.pi
                    b.set_data([off[0], off[0] + 0.09 * lim * np.cos(a)],
                               [off[1], off[1] + 0.09 * lim * np.sin(a)])
            # colours
            cols = E["base"].copy()
            cols[E["ni"] & shocked] = to_rgba(NI_COL)
            cols[E["layer"] == 0, 3] = 0.0                  # Fe core became the neutron star
            if ph == 4:
                cols[:, 3] *= alpha_p
            partsA.set_offsets(pos)
            partsA.set_facecolor(cols)
            if ph == 4:
                nsA.set_offsets([[0, 0]])
        if ph == 6:
            pass
        elif ph >= 2 and ph != 6:
            nsA.set_offsets([[0, 0]])
        axA.set_xlim(-lim, lim)
        axA.set_ylim(-lim, lim)
        bannerA.set_text(banner)
        bannerA.set_color(banner_col)
        timeA.set_text(lines_time)

    # ------------------------------------------------------------ core view
    def draw_B(t, ph, f, P):
        C = S["core"]
        tg = C["tg"]
        jf = int(np.clip(round((t - tg[0]) / DT_GRID), 0, len(tg) - 1))
        rp = float(r_pns(t))
        rs = float(r_shock(t, P))
        lim = B_LIM

        # ---- particles ----
        r = C["pos"][:, jf].copy()
        tc, thx = C["tc"], C["thit"]
        crossed = (tc <= jf) & (tc < thx)
        gone = (thx <= jf) & (thx <= tc)
        ang = C["th"].copy()
        cols = C["base"].copy()
        if crossed.any():
            tcross = tg[np.minimum(tc, len(tg) - 1)]
            dtc = np.maximum(t - tcross, 0)
            w_t = C["w"] + (1 - C["w"]) * np.exp(-dtc / 0.04)
            damp = np.exp(-np.maximum(t - P.t_rev, 0) / 0.3)
            rh = rs * w_t * (1 + 0.06 * np.sin(11 * t + C["ph"]) * damp)
            ang = ang + 0.10 * np.sin(7 * t + C["ph"]) * damp
            ds = t - (tcross + C["dwell"])
            sink = crossed & C["acc"] & (ds > 0)
            rh = np.where(sink, rp + (rh - rp) * np.exp(-np.maximum(ds, 0) / 0.06), rh)
            r = np.where(crossed, rh, r)
            cols[crossed] = to_rgba("#ff7a1a")
            gone = gone | (sink & (r < rp * 1.03))
        cols[gone, 3] = 0.0
        rr = rho_disp(r)
        partsB.set_offsets(np.c_[rr * np.cos(ang), rr * np.sin(ang)])
        partsB.set_facecolor(cols)

        # ---- PNS / inner core ----
        pns.set_radius(rho_disp(rp))
        if t < 0:
            pns.set_facecolor("#9a9fb8")
        else:
            c1, c2 = np.array(to_rgb("gold")), np.array(to_rgb("deepskyblue"))
            pns.set_facecolor(tuple(c1 + (c2 - c1) * float(ssm(t / 2.0))))

        # ---- gain layer + shock front ----
        show_sh = t > 0 and rho_disp(rs) < lim * 1.6
        if show_sh:
            rg = min(max(2.4 * rp, 65.0), rs * 0.85)
            gain_out.set_visible(True)
            gain_out.set_radius(rho_disp(rs))
            gain_in.set_visible(True)
            gain_in.set_radius(rho_disp(rg))
            A = 0.10 * np.exp(-max(t - P.t_rev, 0) / 0.3) + 0.03
            psi = 2 * np.pi * t / 0.12
            Rl = rs * (1 + A * np.cos(theta_line - psi))
            rl = rho_disp(Rl)
            shock_B.set_data(rl * np.cos(theta_line), rl * np.sin(theta_line))
        else:
            gain_out.set_visible(False)
            gain_in.set_visible(False)
            shock_B.set_data([], [])

        # ---- neutrinos streaming out ----
        if t > 0:
            L = float(lnu(t, P))
            a = float(np.clip((np.log10(L) - 50.5) / 3.0, 0.0, 1.0))
            rho_p = rho_disp(rp)
            rn = rho_p + ((nu_ph + 0.02 * f) % 1.0) * (lim * 1.05 - rho_p)
            nuB.set_offsets(np.c_[rn * np.cos(nu_ang), rn * np.sin(nu_ang)])
            nuB.set_alpha(0.75 * a)
        else:
            nuB.set_offsets(np.empty((0, 2)))

        # ---- pulsar beams (neutron star) ----
        if ph >= 4:
            rho_p = rho_disp(rp)
            for j, b in enumerate(beamsB):
                a = 0.35 * f + j * np.pi
                b.set_data([0, 1.5 * np.cos(a)], [0, 1.5 * np.sin(a)])
            nsB.set_text("NEUTRON STAR\nM = %.2f Msun\nR ~ %.0f km\nrho ~ 1e14-1e15 g/cm3" % (P.m_ns, rp))
        else:
            for b in beamsB:
                b.set_data([], [])
            nsB.set_text("")

        # ---- text ----
        if t < 0:
            tt = "t = %+.0f ms  (before bounce)" % (t * 1000)
        else:
            tt = "t = %+.0f ms  after bounce" % (t * 1000)
        timeB.set_text(tt)
        if ph == 0:
            cap = "Iron core (%.2f Msun): fusion cannot go further - no pressure source left" % P.m_fe
        elif t < 0:
            vmax = 0.25 * (0.4 + 0.6 * ssm((t + T_COL) / T_COL))
            cap = "COLLAPSE: electron capture + Fe photodisintegration -> infall ~ %.2f c" % vmax
        elif t < 0.03:
            cap = "BOUNCE at nuclear density (2.7e14 g/cm3): shock wave launched"
        elif t < P.t_rev:
            cap = "SHOCK STALLS (~%.0f km): loses energy breaking up nuclei; neutrinos heat the gain layer" % P.r_st
        elif t < P.t_rev + 0.35:
            cap = "NEUTRINO HEATING REVIVES THE SHOCK -> it runs away: EXPLOSION"
        elif ph <= 3:
            cap = "shock leaves the core carrying ~%.1f x 10^51 erg" % P.E51
        else:
            cap = "Proto-neutron star cools by neutrino emission and shrinks to a neutron star"
        capB.set_text(cap)

        # ---- side markers ----
        if t >= -T_COL and ph >= 1:
            y = rs if (t > 0 and rs > 0) else rp
            dotS.set_data([t], [y])
        else:
            dotS.set_data([], [])
        if t > 1e-3 and ph >= 2:
            dotN.set_data([t], [float(lnu(t, P))])
        else:
            dotN.set_data([], [])
        return rp, rs

    # --------------------------------------------------------------- info text
    def stage_table(P):
        k = 15.0 / P.M
        return [("H  burning (main sequence)", 1.0e7 * k ** 2.5),
                ("He burning", 1.0e6 * k ** 2.0),
                ("C  burning", 1.0e3 * k ** 1.5),
                ("Ne burning", 1.0 * k ** 1.2),
                ("O  burning", 0.5 * k),
                ("Si burning", k / 365.25)]

    def build_info(ph, p, f, t, rp, rs, P):
        head = [
            "M_ZAMS %.1f Msun | He core %.1f | Fe core %.2f Msun" % (P.M, P.m_he, P.m_fe),
            "R_star %.0f Rsun | ejecta %.1f Msun | NS %.2f Msun" % (P.r_rsun, P.m_ej, P.m_ns),
            "E_kin %.2f x10^51 erg | E_nu %.1e erg | Ni-56 %.3f Msun" % (P.E51, P.e_nu, P.m_ni),
            "v_ej ~ %.0f km/s | plateau lasts ~%.0f d" % (P.v_ej, P.tp),
            "-" * 55,
        ]
        body = []
        if ph == 0:
            k = min(int(p * 6), 5)
            for i, (n, d) in enumerate(stage_table(P)):
                body.append("%s %-27s %10s" % (">" if i == k else " ", n, fmt_dur(d)))
            body.append("  Fe core collapse           ~0.3 s")
        elif ph in (1, 2, 3):
            L = float(lnu(max(t, 1e-3), P))
            body += ["Core radius   : %8.0f km" % rp,
                     "Shock radius  : %8.0f km" % rs if rs > 0 else "Shock radius  :   (not yet formed)",
                     "Neutrino lum. : %8.1e erg/s" % L if t > 0 else "Neutrino lum. :   (starts at bounce)",
                     "Revival at    : +%.0f ms" % (P.t_rev * 1000),
                     "~99% of the energy leaves as neutrinos; only ~1% is the blast"]
        elif ph == 4:
            body += ["Shock crossing the star (asymmetric front)",
                     "Explosive burning makes Ni-56 in the inner Si layer",
                     "Fe core is gone: it became the neutron star",
                     "Breakout expected after ~%.1f hours" % (P.t_bo / 3600)]
        elif ph == 5:
            body += ["Shock reaches the surface: BREAKOUT flash",
                     "Neutrinos left hours earlier (as seen in SN 1987A)"]
        else:
            td = (f - B6) * DAYS_PER_FRAME
            i = min(int(td / (T_GRID[1] - T_GRID[0])), len(T_GRID) - 1)
            L = P.L[i]
            if td < 6:
                stage = "cooling envelope after breakout"
            elif td < P.tp - 10:
                stage = "PLATEAU: H recombination wave"
            elif td < P.tp + 20:
                stage = "DROP: recombination front reaches core"
            else:
                stage = "TAIL: Co-56 -> Fe-56 decay"
            body += ["t = %5.0f d   L = %.2e erg/s (%.1e Lsun)" % (td, L, L / LSUN),
                     "T_phot = %.0f K" % t_phot(td, P.tp),
                     "Stage: " + stage,
                     "NS kick (opposite the fast ejecta): ~%.0f km/s" % P.v_kick]
        return "\n".join(head + body)

    # ------------------------------------------------------------------ update
    def update(_):
        f = S["f"]
        P = S["P"]
        ph, p = phase_of(f)
        t = core_time(ph, p, P)
        rp, rs = draw_B(t, ph, f, P)
        draw_A(ph, p, f, P)
        info.set_text(build_info(ph, p, f, t, rp, rs, P))

        if not S["paused"] and f < B7 - 1:
            S["f"] += 1
        return partsA,

    S["update"] = update            # (exposed for testing)
    global anim
    anim = FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False, cache_frame_data=False)
    S["fig"] = fig
    main.S = S
    plt.show()


if __name__ == "__main__":
    main()