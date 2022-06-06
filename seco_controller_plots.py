# %%
import csv
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (
    MultipleLocator, AutoMinorLocator, AutoLocator, FormatStrFormatter)
from operator import methodcaller


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


mpl.use("pgf")
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "sans-serif",
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    "pgf.preamble": r"\usepackage{siunitx}",
}
mpl.rcParams.update(pgf_with_latex)
# plt.rcParams['figure.dpi'] = 300 # 200 e.g. is really fine, but slower
mpl.rcParams["figure.figsize"] = list(set_size(472.03123, fraction=0.9))
plt.style.use('seaborn-notebook')

# %%
# helper functions for transient state plot analysis


def annot_mp(x, y, r, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "$t_p = {:g}$ (s)\n$M_p = \\num{{{:.3e}}}$ (rad)".format(
        xmax, ymax-r)
    if not ax:
        ax = plt.gca()
    if ymax-r >= 0:
        ax.text(x=xmax-0.01, y=0.05, s=text,
                rotation=0, horizontalalignment='right')
        ax.axvline(x=xmax, color='k', linestyle='-.', linewidth=0.8)


def annot_ts(x, y, r, margin, ax=None):
    lbound = r*(1-margin)
    ubound = r*(1+margin)
    yrev = y[::-1]
    idx = np.argmax((yrev <= lbound) | (yrev >= ubound))
    xts = x[len(y) - idx - 1]
    text = "$t_s = {:g}$ (s)".format(xts)
    if not ax:
        ax = plt.gca()
    ax.text(x=xts+0.01, y=0.05, s=text, rotation=0)
    ax.axvline(x=xts, color='k', linestyle='-.', linewidth=0.8)
    ax.hlines(y=[lbound, ubound], xmin=xts, xmax=x[-1],
              color='k', linestyle='-.', linewidth=0.8)


# %%
# motor transfer function parameters
k = 2652.28
p = 64.986

# %%
# P transfer function
kp_arr = [0.5, 1, 5, 10, 20]
kp_fixed = 1

# Kp variation
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp in kp_arr:
    h_pos = sig.TransferFunction([kp*k], [1, p, kp*k])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$K_p = {kp}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(f"P: Motor closed-loop model step response, various $K_P$")
plt.savefig('step_response_p_kp-var.pdf', bbox_inches='tight')
fig


# %%
# P-D transfer function
kp_arr = [0.1, 1, 5, 10, 50, 100]
td_arr = [1, 5, 10, 50, 100]
kp_fixed = 1
td_fixed = 0.1

# Kp variation with fixed td
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp in kp_arr:
    h_pos = sig.TransferFunction(
        [kp*k], [1, p + kp*k*td_fixed, kp*k])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$K_p = {kp}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
#plt.xlim([0, 0.750])
plt.title(
    f"P-D: Motor closed-loop model step response, various $K_P$, fixed $\\tau _D = {td_fixed}$")
plt.savefig('step_response_p-d_kp-var.pdf', bbox_inches='tight')

# td variation with fixed Kp
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for td in td_arr:
    h_pos = sig.TransferFunction(
        [kp_fixed*k], [1, p + kp_fixed*k*td, kp_fixed*k])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$\\tau _D = {td}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
#plt.xlim([0, 0.750])
plt.title(
    f"P-D: Motor closed-loop model step response, fixed $K_P = {kp_fixed}$, various $\\tau _D$")
plt.savefig('step_response_p-d_td-var.pdf', bbox_inches='tight')

# ramp response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

h_pos = sig.TransferFunction(
    [kp_fixed*k], [1, p + kp_fixed*k*td, kp_fixed*k, 0])  # closed loop
tx = np.linspace(0, 500, 1000, endpoint=True)
# print(tx)
tout, ampl = sig.step2(h_pos, T=tx)
axs.plot(
    tout, ampl, label=f"$K_p = {kp_fixed}, \\tau_D = {td_fixed}$", linewidth=0.8)
axs.plot(tout, tout, label=f"Input signal", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"P-D: Motor closed-loop model ramp response, fixed $\\tau _D = {td_fixed}, K_P = {kp_fixed}$")
plt.savefig('ramp_response_p-d.pdf', bbox_inches='tight')

# ramp response error
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

axs.plot(tout, tout-ampl, label="error", linewidth=0.8, c="red")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"P-D: Motor closed-loop model ramp response error, fixed $\\tau _D = {td_fixed}, K_P = {kp_fixed}$")
plt.savefig('ramp_response_p-d-err.pdf', bbox_inches='tight')

# %%
# PD transfer function
td_arr = [-0.015, -0.01, 0.01, 0.015]
kp_fixed = 1

# td variation with fixed Kp
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for td in td_arr:
    h_pos = sig.TransferFunction(
        [kp_fixed*k*td, kp_fixed*k], [1, p + kp_fixed*k*td, kp_fixed*k])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$\\tau _D = {td}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PD: Motor closed-loop model step response, fixed $K_P = {kp_fixed}$, various $\\tau _D$")
plt.savefig('step_response_pd_td-var.pdf', bbox_inches='tight')
fig

# %%
# PI transfer function
kp_arr = [0.5, 1, 2, 5, 10]
ti_arr = [0.1, 0.5, 0.7, 1]
kp_fixed = 1
ti_fixed = 0.1

# Kp variation with fixed ti
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp in kp_arr:
    h_pos = sig.TransferFunction(
        [kp*k, kp*k/ti_fixed], [1, p, kp*k, kp*k/ti_fixed])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$K_p = {kp}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
#plt.xlim([0, 0.750])
plt.title(
    f"PI: Motor closed-loop model step response, various $K_P$, fixed $\\tau _I = {ti_fixed}$")
plt.savefig('step_response_pi_kp-var.pdf', bbox_inches='tight')
a = fig

# ti variation with fixed Kp
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for ti in ti_arr:
    h_pos = sig.TransferFunction(
        [kp_fixed*k, kp_fixed*k/ti], [1, p, kp_fixed*k, kp_fixed*k/ti])  # closed loop
    t, ampl = sig.step2(h_pos)
    axs.plot(t, ampl, label=f"$\\tau _I = {ti}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.xlim([0, 2])
plt.title(
    f"PI: Motor closed-loop model step response, fixed $K_P = {kp_fixed}$, various $\\tau _I$")
plt.savefig('step_response_pi_ti-var.pdf', bbox_inches='tight')

# ramp response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

h_pos = sig.TransferFunction([kp_fixed*k, kp_fixed*k/ti_fixed],
                             [1, p, kp_fixed*k, kp_fixed*k/ti_fixed, 0])  # closed loop
t, ampl = sig.step2(h_pos)
axs.plot(
    t, ampl, label=f"$K_p = {kp_fixed}, \\tau_I = {ti_fixed}$", linewidth=0.8)
axs.plot(t, t, label=f"Input signal", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PI: Motor closed-loop model ramp response, fixed $\\tau _I = {ti_fixed}, K_P = {kp_fixed}$")
plt.savefig('ramp_response_pi.pdf', bbox_inches='tight')

# parabola response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

ti_fixed = 10
kp_fixed = 1
h_pos = sig.TransferFunction([kp_fixed*k, kp_fixed*k/ti_fixed],
                             [1, p, kp_fixed*k, kp_fixed*k/ti_fixed, 0, 0])  # closed loop
tx = np.linspace(0, 100, 1000)
t, ampl = sig.step2(h_pos, T=tx)
axs.plot(
    t, ampl, label=f"$K_p = {kp_fixed}, \\tau_I = {ti_fixed}$", linewidth=0.8)
axs.plot(t, t*t/2, label=f"Input signal", linewidth=0.8, linestyle="dotted")
#axs.plot(t, t*t/2, label=f"Input signal", linewidth=0.8, linestyle="dotted")
#axs.plot(t, ampl-(t*t/2), label="Error")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PI: Motor closed-loop model parabola response, fixed $\\tau _I = {ti_fixed}, K_P = {kp_fixed}$")
plt.savefig('parabola_response_pi.pdf', bbox_inches='tight')

# parabola response error
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

axs.plot(t, t*t/2-ampl, label="error", linewidth=0.8, c="red")
axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PI: Motor closed-loop model parabola response error, fixed $\\tau _I = {ti_fixed}, K_P = {kp_fixed}$")
plt.savefig('parabola_response_pi-err.pdf', bbox_inches='tight')

# %%
# PID transfer function

parameter_arr = [(1, 1, 1), (5, 1, 1), (1, 1, 10), (10, 1, 10)]

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp, ti, td in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*td, kp*k, kp*k/ti], [1, p + kp*k*td, kp*k, kp*k/ti])  # closed loop
    tx = np.linspace(0, 4e-3, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _D = {td}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PID: Motor closed-loop model step response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('step_response_pid.pdf', bbox_inches='tight')
fig

# %%
# PI-D transfer function
parameter_arr = [(1, 1, 1), (1, 10, 1), (1, 1, 2), (1, 10, 0.1)]

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp, ti, td in parameter_arr:
    h_pos = sig.TransferFunction(
        [k*kp, k*kp/ti], [1, p + kp*k*td, kp*k, kp*k/ti])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _D = {td}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PI-D: Motor closed-loop model step response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('step_response_pi-d.pdf', bbox_inches='tight')
fig

# %%
# PID-D transfer function
parameter_arr = [(1, 1, 1, 1), (1, 10, 1, 1), (1, 10, 1, 0.1),
                 (1, 1, 2, 1), (1, 10, 0.1, 1), (1, 10, 0.1, 0.1)]

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp, ti, td1, td2 in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*td1, kp*k, kp*k/ti], [1, p + kp*k*(td1+td2), kp*k, kp*k/ti])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PID-D: Motor closed-loop model step response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('step_response_pid-d.pdf', bbox_inches='tight')
a = fig
# ramp response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for kp, ti, td1, td2 in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*td1, kp*k, kp*k/ti], [1, p + kp*k*(td1+td2), kp*k, kp*k/ti, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2}$", linewidth=0.8)
axs.plot(t, t, label="Input function", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"PID-D: Motor closed-loop model ramp response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('ramp_response_pid-d.pdf', bbox_inches='tight')

# parabola response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for kp, ti, td1, td2 in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*td1, kp*k, kp*k/ti], [1, p + kp*k*(td1+td2), kp*k, kp*k/ti, 0, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2}$", linewidth=0.8)
axs.plot(t, t*t/2, label="Input function", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(f"PID-D: Motor closed-loop model parabola response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('parabola_response_pid-d.pdf', bbox_inches='tight')
a

# %%
# D|PID transfer function
parameter_arr = [(1, 1, 1, 1), (1, 10, 1, 1), (1, 10, 1, 0.1),
                 (1, 1, 2, 1), (1, 10, 0.1, 0.1)]

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for kp, ti, td1, td2 in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti])  # closed loop
    tx = np.linspace(0, 15, 1000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"D/PID: Motor closed-loop model step response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('step_response_dpid.pdf', bbox_inches='tight')
a = fig

# ramp response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for kp, ti, td1, td2 in parameter_arr:
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2}$", linewidth=0.8)
axs.plot(t, t, label="Input function", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"D/PID: Motor closed-loop model ramp response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('ramp_response_dpid.pdf', bbox_inches='tight')
b = fig

# parabola response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

parameter_arr = [(1, 10, 1, 1), (1, 10, 1, 2)]

for kp, ti, td1, td2 in parameter_arr:
    if td2 == 2:
        td2 = p/(k*kp)
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti, 0, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"$K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2:.4g}$", linewidth=0.8)
axs.plot(t, t*t/2, label="Input function", linewidth=0.8, linestyle="dotted")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(f"D/PID: Motor closed-loop model parabola response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('parabola_response_dpid.pdf', bbox_inches='tight')
fig

# %%
# D|PID stability analysis
parameter_arr = [(True, 1, 1, 1, 2), (True, 1, 1, 1, 1),
                 (False, -1, -1, 1, 2), (False, -1, -1, 1, 1)]

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
plt.axhline(y=1, color='r', linestyle='--', linewidth=0.8)

for stable, kp, ti, td1, td2 in parameter_arr:
    if td2 == 2:
        td2 = p/(k*abs(kp))
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti])  # closed loop
    tx = np.linspace(0, 15, 1000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"Stable = {stable}, $K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2:.4g}$", linewidth=0.8)

axs.get_lines()[-1].set_color('mediumturquoise')

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.ylim([-10, 2.5])
plt.title(f"D/PID: Controller stability analysis, step response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('stability_step_response_dpid.pdf', bbox_inches='tight')
a = fig

# ramp response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for stable, kp, ti, td1, td2 in parameter_arr:
    if td2 == 2:
        td2 = p/(k*abs(kp))
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"Stable = {stable}, $K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2:.4g}$", linewidth=0.8)
axs.plot(t, t, label="Input function", linewidth=0.8, linestyle="dotted")
axs.get_lines()[-2].set_color('mediumturquoise')

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.ylim([-10, 45])
plt.title(f"D/PID: Controller stability analysis, ramp response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('stability_ramp_response_dpid.pdf', bbox_inches='tight')
b = fig

# parabola response
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for stable, kp, ti, td1, td2 in parameter_arr:
    if td2 == 2:
        td2 = p/(k*abs(kp))
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti, 0, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(
        t, ampl, label=f"Stable = {stable}, $K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2:.4g}$", linewidth=0.8)
axs.plot(t, t*t/2, label="Input function", linewidth=0.8, linestyle="dotted")
axs.get_lines()[-2].set_color("mediumturquoise")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.ylim([-100, 800+50])
plt.title(f"D/PID: Controller stability analysis, parabola response, various $\\tau _I, \\tau_D, K_P$")
plt.savefig('stability_parabola_response_dpid.pdf', bbox_inches='tight')

# parabola response error
fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
axs.set_prop_cycle(color=["indigo", "magenta"])

for stable, kp, ti, td1, td2 in parameter_arr:
    if not stable:
        continue
    if td2 == 2:
        td2 = p/(k*abs(kp))
    h_pos = sig.TransferFunction(
        [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti, 0, 0])  # closed loop
    tx = np.linspace(0, 40, 2000, endpoint=True)
    t, ampl = sig.step2(h_pos, T=tx)
    axs.plot(t, t*t/2-ampl,
             label=f"Error: Stable = {stable}, $K_P = {kp}$, $\\tau _I = {ti}$, $\\tau _{{D1}} = {td1}$, $\\tau _{{D2}} = {td2:.4g}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"amplitude")
axs.set_xlabel("time (s)")
plt.title(
    f"D/PID: Controller stability analysis, parabola response error, various $\\tau_D$")
plt.savefig('stability_parabola_response_dpid-err.pdf', bbox_inches='tight')
fig
