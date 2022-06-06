# %%
from pprint import pprint
from random import sample
from collections import OrderedDict
from typing import List, Tuple
import multiprocess as mps
import csv
import time
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import (
    MultipleLocator, AutoMinorLocator, AutoLocator, FormatStrFormatter)

import dill
dill.settings['recurse'] = True


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
    "pgf.preamble": "\\usepackage{siunitx}\n\\usepackage[dvipsnames]{xcolor}",
}
mpl.rcParams.update(pgf_with_latex)
# plt.rcParams['figure.dpi'] = 300 # 200 e.g. is really fine, but slower
mpl.rcParams["figure.figsize"] = list(set_size(472.03123, fraction=0.9))
plt.style.use('seaborn-notebook')

np.seterr(divide="raise")

# %%
# helper functions for contoller design


def compute_mp(x: np.array, y: np.array, bound: tuple, ref: float) -> Tuple[bool, float]:
    xmax = x[np.argmax(y)]
    ymax = np.max(y)
    mp_relation = ymax/ref
    return ((bound[0] <= ymax <= bound[1]), mp_relation)


def compute_ts(x: np.array, y: np.array, ref: float, tolerance: float, limit: float) -> Tuple[bool, float]:
    lbound = ref*(1-tolerance)
    ubound = ref*(1+tolerance)
    yrev = y[::-1]
    idx = np.argmax((yrev <= lbound) | (yrev >= ubound))
    ts = x[len(y) - idx - 1]
    return ((ts <= limit), ts)


def compute_tr(x: np.array, y: np.array, ref: float, limit: float) -> Tuple[bool, float]:
    idxd = np.where(np.diff(np.sign(y-ref)))[0][0]
    tr = (x[idxd] + x[idxd+1])/2
    return ((tr <= limit), tr)


def transform_coefs(zeta: float, beta: float, beta2: float, p: float, k: float) -> Tuple[float, float, float, float]:
    kp = (p**2*(2*beta+1/(zeta**2)))/(beta2**2*k)
    td1 = (beta2*(beta-beta2+2))/(p*(2*beta+1/(zeta**2)))
    td2 = p/(k*kp)
    ti = (beta2*(zeta**2)*(2*beta+1/(zeta**2)))/(p*beta)
    return (kp, ti, td1, td2)

def transform_coefs_motor_telelabo(zeta: float, beta: float, beta2: float, p: float, k: float, gearbox: int, sample_period: float) -> Tuple[float, float, float, float]:
    (kp_base, ti_base, td1_base, td2_base) = transform_coefs(zeta, beta, beta2, p, k/gearbox)
    kp = kp_base
    td1 = kp*td1_base/sample_period
    td2 = kp*td2_base/sample_period
    ti = kp/ti_base*sample_period
    return (round(kp, 3), round(ti, 3), round(td1, 3), round(td2, 3))

def create_dataframes_from_motor_data_csv(experiment_name: str) -> List[pd.DataFrame]:
    output_vars = ["POS", "REF", "ERR", "U", "USAT"]
    dfs = dict.fromkeys(map(lambda x: x.lower(), output_vars))
    for param in output_vars:
        tdf = pd.read_csv(f"~/repos/secoStudentsQueueAppExported/data_clean/step_sys/{experiment_name}-MOTOR3{param}", sep=" ", header=None)
        tdf.columns = ["time", param.lower()]
        dfs[param.lower()] = tdf
    return dfs

def generate_tex_tables_from_results(mp_valid, ts_valid, tr_valid, param_dict):
    beta_table = ""
    beta2_table = ""
    for iz, zeta in enumerate(zeta_values):
        beta_table += f"${zeta:.2f}$\t&\t$[{np.min(beta_values[mp_valid[iz]]):.1f}, {np.max(beta_values[mp_valid[iz]]):.1f}]$ \\\\\n"
        for ib, beta in enumerate(param_dict[iz][1]):
            beta2_table += f"${zeta:.2f}$\t&\t${beta:.2f}$\t&\t$[{np.min(beta2_values[ts_valid[iz][ib]]):.1f}, {np.max(beta2_values[ts_valid[iz][ib]]):.1f}]$"
            beta2_table += f"\t&\t$[{np.min(beta2_values[tr_valid[iz][ib]]):.1f}, {np.max(beta2_values[tr_valid[iz][ib]]):.1f}]$ \\\\\n"
    # output .tex files containing the tables
    with open("beta_table.tex", "w") as f:
        f.write(beta_table)
    with open("beta2_table.tex", "w") as f:
        f.write(beta2_table)
    # print the tables
    print("Mp(beta) table. cols: zeta, valid beta range")
    print(beta_table)
    print("ts,tr(beta2) table. cols: zeta, beta, valid beta2 range for ts, valid beta2 range for tr")
    print(beta2_table)

def generate_csv_from_results(mp_valid, ts_valid, tr_valid, param_dict, filename="valid_tuples.csv"):
    with open(filename, "w") as f:
        f.write("zeta,beta,beta2,ts_beta2_min,ts_beta2_median,ts_beta2_max,tr_beta2_min,tr_beta2_median,tr_beta2_max\n")
        for iz, zeta in enumerate(zeta_values):
            for ib, beta in enumerate(param_dict[iz][1]):
                ts_valid_beta2_vector = beta2_values[ts_valid[iz][ib]]
                tr_valid_beta2_vector = beta2_values[tr_valid[iz][ib]]
                f.write(f"{zeta:.2f},{beta:.2f},{np.min(ts_valid_beta2_vector):.1f},{np.median(ts_valid_beta2_vector):.1f},{np.max(ts_valid_beta2_vector):.1f},{np.min(tr_valid_beta2_vector):.1f},{np.median(tr_valid_beta2_vector):.1f},{np.max(tr_valid_beta2_vector):.1f}\n")

def generate_tex_tables_sut(sut_names: List[str], sut_coef: List[List[float]], sut_parametric: OrderedDict, sut_labo: OrderedDict, filename="systems_under_test.tex"):
    with open(filename, "w") as f:
        for iin, name in enumerate(sut_names):
            f.write(f"{name}\t&\t")
            f.write("\t&\t".join(map(lambda x: f"${x:.3f}$", sut_coef[iin])) + "\t&\t")
            f.write("\t&\t".join(map(lambda x: f"${x:.3f}$", sut_parametric[name])) + "\t&\t")
            f.write("\t&\t".join(map(lambda x: f"${x:.3f}$", sut_labo[name])) + "\\\\\n")

def generate_csv_sut(sut_names: List[str], sut_coef: List[List[float]], sut_parametric: OrderedDict, sut_labo: OrderedDict, filename="systems_under_test.csv"):
    with open(filename, "w") as f:
        f.write("sys_name,zeta,beta,beta2,kp,ti,td1,td2,kp_labo,ki_labo,kd1_labo,kd2_labo\n")
        for iin, name in enumerate(sut_names):
            f.write(f"{name},")
            f.write(",".join(map(lambda x: f"{x:.3f}", sut_coef[iin])) + ",")
            f.write(",".join(map(lambda x: f"{x:.3f}", sut_parametric[name])) + ",")
            f.write(",".join(map(lambda x: f"{x:.3f}", sut_labo[name])) + "\n")
# %%
# motor transfer function constants
k = 2652.28
p = 64.986


# %%
"""
beta vs mp validation simulations (takes ~ 2 min to run on 1 core CPU)
"""

zeta_values = np.asarray([0.450, 0.550, 0.6, 0.7, 0.8])
beta_values = np.arange(0.1, 60, 0.1)
beta2_default_value = 0.8
ref = 1
mp_bound = (ref*(1+5/100), ref*(1+14/100))
mp_arr = np.zeros((zeta_values.shape[0], beta_values.shape[0]))
mp_valid = np.zeros_like(mp_arr, dtype=bool)

parallel_mp_args = []


def parallel_mp_solve(tf: sig.TransferFunction, iz: int, ib: int) -> Tuple[int, int, Tuple[bool, float]]:
    t, a = sig.step2(tf)
    return (iz, ib, compute_mp(t, a, mp_bound, ref))


# with tqdm(total=np.prod(mp_arr.shape), ascii=' >=', ncols=100) as pbar:
# with tqdm(total=np.prod(mp_arr.shape)) as pbar:
#pbar.set_description("Simulating for Mp search")
for iz, zeta in enumerate(zeta_values):
    for ib, beta in enumerate(beta_values):
        # parameter conversion from frequency and dampening coefficients to controller constants
        (kp, ti, td1, td2) = transform_coefs(
            zeta, beta, beta2_default_value, p, k)
        h_dpid = sig.TransferFunction(
            [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti])
        parallel_mp_args.append((h_dpid, iz, ib))

print(
    f"Starting Mp cross-validation simulation using {mps.cpu_count()} CPU cores...")
print(f"Python GIL Multiprocessing set {np.prod(np.shape(mp_arr))} iters")
tic = time.perf_counter()
with mps.Pool(mps.cpu_count()) as pool:
    parallel_res = pool.starmap(parallel_mp_solve, parallel_mp_args)
toc = time.perf_counter()
print(f"Simulated {np.prod(np.shape(mp_arr))} TFs in {toc - tic:0.4f} seconds")

for result in parallel_res:
    (iz, ib, (valid_mp, mpi)) = result
    mp_valid[iz, ib], mp_arr[iz, ib] = (valid_mp, mpi)

np.savez_compressed('mp_data', mp_arr=mp_arr, mp_valid=mp_valid)


# %%
"""
mp validation plot
"""

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
axs.axhspan(ymin=mp_bound[0], ymax=mp_bound[1], color="lime", alpha=0.15)

for iz, mp_vector in enumerate(mp_arr):
    axs.plot(beta_values, mp_vector,
             label=f"$\\zeta = {zeta_values[iz]:.2f} \longrightarrow \\beta_{{\\min}} = {np.min(beta_values[mp_valid[iz]]):.1f},\\ \\beta_{{\\max}} = {np.max(beta_values[mp_valid[iz]]):.1f}$", linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"$M_p$")
axs.set_xlabel("$\\beta$")
axs.set_title(f"D/PID: $M_p (\\beta)$ cross-validation with $\\zeta$ subset")
plt.savefig('dpid-beta-vs-mp-validate.pdf', bbox_inches='tight')


# %%
"""
beta2 ts, tr validation simulations (takes ~ 5 min to run on 1 core CPU)
"""
# configuration for validation of beta2
ts_limit = 0.45
tr_limit = 0.25
tolerance = 0.02

# generate beta2 test vector
beta2_values = np.arange(0.1, 50, 0.1)

# simulation to obtain beta2 values
zeta_beta_pairs = OrderedDict([(z, []) for z in zeta_values])

# 3-tensor to store the results of ts computations and the valid coefficients
ts_arr = np.zeros((zeta_values.shape[0], 3, beta2_values.shape[0]))
ts_valid = np.zeros_like(ts_arr, dtype=bool)
tr_arr = np.zeros_like(ts_arr)
tr_valid = np.zeros_like(ts_arr, dtype=bool)

# pick beta valid extrema and some middle values as mp-compliant parameters to test
for iz, zeta in enumerate(zeta_values):
    zeta_beta_pairs[zeta].extend([np.min(beta_values[mp_valid[iz]]),
                                  np.median(beta_values[mp_valid[iz]]), np.max(beta_values[mp_valid[iz]])])

parallel_ts_tr_args = []


def parallel_ts_tr_solve(tf: sig.TransferFunction, iz: int, ib: int, ib2: int) -> Tuple[int, int, int, Tuple[bool, float], Tuple[bool, float]]:
    t, a = sig.step2(tf)
    return (iz, ib, ib2, compute_ts(t, a, ref, tolerance, ts_limit), compute_tr(t, a, ref, tr_limit))


for iz, (zeta, beta_selection) in enumerate(zeta_beta_pairs.items()):
    for ib, beta in enumerate(beta_selection):
        for ib2, beta2 in enumerate(beta2_values):
            (kp, ti, td1, td2) = transform_coefs(zeta, beta, beta2, p, k)
            h_dpid = sig.TransferFunction(
                [kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti])
            parallel_ts_tr_args.append((h_dpid, iz, ib, ib2))

print(
    f"Starting ts, tr cross-validation simulation using {mps.cpu_count()} CPU cores...")
print(f"Python GIL Multiprocessing set {np.prod(np.shape(ts_arr))} iters")
tic = time.perf_counter()
with mps.Pool(mps.cpu_count()) as pool:
    parallel_res = pool.starmap(parallel_ts_tr_solve, parallel_ts_tr_args)
toc = time.perf_counter()
print(f"Simulated {np.prod(np.shape(ts_arr))} TFs in {toc - tic:0.4f} seconds")

for result in parallel_res:
    (iz, ib, ib2, (valid_ts, tsi), (valid_tr, tri)) = result
    ts_valid[iz, ib, ib2], ts_arr[iz, ib, ib2] = (valid_ts, tsi)
    tr_valid[iz, ib, ib2], tr_arr[iz, ib, ib2] = (valid_tr, tri)

np.savez_compressed('ts_tr_data', ts_arr=ts_arr,
                    ts_valid=ts_valid, tr_arr=tr_arr, tr_valid=tr_valid)


# %%
"""
ts validation plot
"""

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
axs.axhspan(ymin=0, ymax=ts_limit, color="lime", alpha=0.15)
axs.set_prop_cycle(color=[*np.array(list(mcolors.TABLEAU_COLORS.items()))[:, 0], *np.array(list(mcolors.TABLEAU_COLORS.items()))[:5, 0]],
                   linestyle=[*(["solid"] * 10), *(["dashed"] * 5)])

param_dict = list(zeta_beta_pairs.items())

#plot_cm = plt.get_cmap('tab20')
#plot_colors = [plot_cm(i) for i in np.linspace(0, 1, len(ts_arr) * len(ts_arr[0]))]

for iz, ts_matrix in enumerate(ts_arr):
    for ib, ts_vector in enumerate(ts_matrix):
        axs.plot(beta2_values, ts_vector,
                 label=f"$\\zeta = {zeta_values[iz]:.2f}, \\beta = {param_dict[iz][1][ib]:.2f}$", linewidth=0.8)

axs.legend(fontsize="xx-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"$t_s$")
axs.set_xlabel("$\\beta_2$")
axs.set_title(
    f"D/PID: $t_s (\\beta_2)$ cross-validation with $\\zeta, \\beta$ subset")
plt.savefig('dpid-beta2-vs-ts-validate.pdf', bbox_inches='tight')


# %%
"""
tr validation plot, it doesn't give useful info as all simulations are within tr constraint
"""

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()
axs.axhspan(ymin=0, ymax=tr_limit, color="lime", alpha=0.15)
axs.set_prop_cycle(color=[*np.array(list(mcolors.TABLEAU_COLORS.items()))[:, 0], *np.array(list(mcolors.TABLEAU_COLORS.items()))[:5, 0]],
                   linestyle=[*(["solid"] * 10), *(["dashed"] * 5)])

param_dict = list(zeta_beta_pairs.items())

for iz, tr_matrix in enumerate(tr_arr):
    for ib, tr_vector in enumerate(tr_matrix):
        axs.plot(beta2_values, tr_vector,
                 label=f"$\\zeta = {zeta_values[iz]:.2f}, \\beta = {param_dict[iz][1][ib]:.2f}$", linewidth=0.8)

axs.legend(fontsize="xx-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel(r"$t_r$")
axs.set_xlabel("$\\beta_2$")
axs.set_title(
    f"D/PID: $t_r (\\beta_2)$ cross-validation with $\\zeta, \\beta$ subset")
plt.savefig('dpid-beta2-vs-tr-validate.pdf', bbox_inches='tight')
# %%
generate_tex_tables_from_results(mp_valid, ts_valid, tr_valid, param_dict)
generate_csv_from_results(mp_valid, ts_valid, tr_valid, param_dict)

# %%
system_names = ["A", "B", "C", "D", "E", "E2"]
sut = OrderedDict.fromkeys(system_names)
sut_coef = [[0.6, 34.1, 15], [0.6, 21.55, 20], [0.45, 12, 3], [0.8, 18.9, 13.8], [0.45, 15, 5], [0.6, 10, 2.5]]
sut_parametric = OrderedDict.fromkeys(system_names)

gearbox = 23
sampling_period = 5e-3

for i, sut_name in enumerate(sut):
    sut[sut_name] = transform_coefs_motor_telelabo(*sut_coef[i], p, k, gearbox, sampling_period)
    sut_parametric[sut_name] = transform_coefs(*sut_coef[i], p, k)

pprint(sut)
# %%
fig, axs = plt.subplots(3, 2, sharex="all", sharey="all")
sys_sim = OrderedDict.fromkeys(sut_parametric.keys())

ref = 1
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

for isys, sys in enumerate(sut_parametric):
    (kp, ti, td1, td2) = sut_parametric[sys]
    h_dpid = sig.TransferFunction([kp*k*(td1+td2), kp*k, kp*k/ti], [1, p + kp*k*td1, kp*k, kp*k/ti])
    t, a = sig.step(h_dpid, T=np.linspace(0, 2, 1000))
    sys_sim[sys] = {"t": t, "a": a}
    this_ax = axs[int(isys/2), isys % 2]
    this_ax.axhline(ref, linestyle="--", color="red", linewidth=0.8)
    this_ax.plot(t, a, label="H(s)", linewidth=0.8)
    res_mp = compute_mp(t, a, mp_bound, ref)
    ok_mp = r"\textcolor{Green}{OK}" if res_mp[0] else r"\textcolor{Red}{KO}"
    res_ts = compute_ts(t, a, ref, tolerance, ts_limit)
    ok_ts = r"\textcolor{Green}{OK}" if res_ts[0] else r"\textcolor{Red}{KO}"
    at = AnchoredText(f"\\textbf{{{sys}}} response:\n$M_p = {(res_mp[1]-1)*100:.2f}\\%$ {ok_mp} \n$t_s = \SI{{{res_ts[1]:.2f}}}{{\\second}}$ {ok_ts}", prop=dict(size=10), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    this_ax.add_artist(at)

for ax in axs.flat:
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(linestyle='--', linewidth=0.5, which="major")
    ax.grid(linestyle='-.', linewidth=0.1, which="minor")
fig.supylabel("position (rad)")
fig.supxlabel("time (s)")
fig.suptitle(f"Simulated D/PID response for controllers under test")
plt.savefig('dpid-sut-simulation.pdf', bbox_inches='tight')
# %%
fig, axs = plt.subplots(3, 2, sharex="all", sharey="all")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

for isys, sys in enumerate(sut_parametric):
    sut_dfs = create_dataframes_from_motor_data_csv(f"sys{sys}")
    this_ax = axs[int(isys/2), isys % 2]
    this_ax.axhline(ref, linestyle="--", color="red", linewidth=0.8)
    this_ax.plot(sut_dfs["pos"]["time"], sut_dfs["pos"]["pos"], label="pos", linewidth=0.8, c="indigo")
    res_mp = compute_mp(sut_dfs["pos"]["time"], sut_dfs["pos"]["pos"], mp_bound, ref)
    ok_mp = r"\textcolor{Green}{OK}" if res_mp[0] else r"\textcolor{Red}{KO}"
    res_ts = compute_ts(sut_dfs["pos"]["time"], sut_dfs["pos"]["pos"], ref, tolerance, ts_limit)
    ok_ts = r"\textcolor{Green}{OK}" if res_ts[0] else r"\textcolor{Red}{KO}"
    at = AnchoredText(f"\\textbf{{{sys}}} response:\n$M_p = {(res_mp[1]-1)*100:.2f}\\%$ {ok_mp} \n$t_s = \SI{{{res_ts[1]:.2f}}}{{\\second}}$ {ok_ts}", prop=dict(size=10), frameon=True, loc='lower right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    this_ax.add_artist(at)

for ax in axs.flat:
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(linestyle='--', linewidth=0.5, which="major")
    ax.grid(linestyle='-.', linewidth=0.1, which="minor")
fig.supylabel("position (rad)")
fig.supxlabel("time (s)")
fig.suptitle(f"Real D/PID response for controllers under test")
plt.savefig('dpid-sut-real.pdf', bbox_inches='tight')
# %%
fig, axs = plt.subplots(3, 2, sharex="all", sharey="all")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

for isys, sys in enumerate(sut_parametric):
    sut_dfs = create_dataframes_from_motor_data_csv(f"sys{sys}")
    this_ax = axs[int(isys/2), isys % 2]
    this_ax.axhline(ref, linestyle="--", color="red", linewidth=0.8)
    this_ax.plot(sys_sim[sys]["t"], sys_sim[sys]["a"]*ref, label="simulation", linewidth=0.8)
    this_ax.plot(sut_dfs["pos"]["time"], sut_dfs["pos"]["pos"], label="real", linewidth=0.8)
    this_ax.legend(title=f"\\textbf{{{sys}}}", fontsize="x-small", title_fontsize=10, loc="lower right")

for ax in axs.flat:
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(linestyle='--', linewidth=0.5, which="major")
    ax.grid(linestyle='-.', linewidth=0.1, which="minor")
fig.supylabel("position (rad)")
fig.supxlabel("time (s)")
fig.suptitle(f"Step response comparison between simulated and real D/PID response for controllers under test")
plt.savefig('dpid-sut-comparison.pdf', bbox_inches='tight')
# %%
fig, axs = plt.subplots(3, 2, sharex="all", sharey="all")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

for isys, sys in enumerate(sut_parametric):
    sut_dfs = create_dataframes_from_motor_data_csv(f"sys{sys}")
    this_ax = axs[int(isys/2), isys % 2]
    this_ax.axhline(0, linestyle="--", color="red", linewidth=0.8)
    this_ax.plot(sys_sim[sys]["t"], ref-sys_sim[sys]["a"]*ref, label="simulation", linewidth=0.8, c="indigo")
    this_ax.plot(sut_dfs["err"]["time"], sut_dfs["err"]["err"], label="real", linewidth=0.8, c="magenta")
    this_ax.legend(title=f"\\textbf{{{sys}}}", fontsize="x-small", title_fontsize=10, loc="upper right")

for ax in axs.flat:
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(linestyle='--', linewidth=0.5, which="major")
    ax.grid(linestyle='-.', linewidth=0.1, which="minor")
fig.supylabel("position (rad)")
fig.supxlabel("time (s)")
fig.suptitle(f"Step error comparison between simulated and real D/PID response for controllers under test")
plt.savefig('dpid-sut-comparison-error.pdf', bbox_inches='tight')
# %%
E2_dfs = create_dataframes_from_motor_data_csv("sysE2")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

for ov in ["u", "usat"]:
    E2_dfs[ov].plot(ax=axs, x="time", y=ov, linewidth=0.8)

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_ylabel("motor control input voltage (V)")
axs.set_xlabel("time (s)")
axs.set_title("TeleLabo motor control signals for E2 controller")
plt.savefig('e2-control-signals.pdf', bbox_inches='tight')
# %%
E2_dfs_ramp = create_dataframes_from_motor_data_csv("sysE2_ramp")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

E2_dfs_ramp["ref"].plot(ax=axs, x="time", y="ref", linewidth=0.8, linestyle="--", c="red", label="ref (rad)")
for ov in ["pos", "err"]:
    E2_dfs_ramp[ov].plot(ax=axs, x="time", y=ov, linewidth=0.8, label=f"{ov} (rad)")
for ov in ["u"]:
    E2_dfs_ramp[ov].plot(ax=axs, x="time", y=ov, linewidth=0.8, label=f"{ov} (V)")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_xlabel("time (s)")
axs.set_title("Real motor ramp response with E2 controller")
plt.savefig('e2-ramp.pdf', bbox_inches='tight')
# %%
E2_dfs_parab = create_dataframes_from_motor_data_csv("sysE2_parab")

ref = 3.14
mp_bound = (ref*(1+5/100), ref*(1+14/100))
tolerance = 0.02
ts_limit = 0.45
tr_limit = 0.25

if "fig" in vars():
    del fig

fig = plt.figure(figsize=set_size(472.03123, fraction=0.9))
axs = fig.gca()

E2_dfs_parab["ref"].plot(ax=axs, x="time", y="ref", linewidth=0.8, linestyle="--", c="red", label="ref (rad)")
for ov in ["pos", "err"]:
    E2_dfs_parab[ov].plot(ax=axs, x="time", y=ov, linewidth=0.8, label=f"{ov} (rad)")
for ov in ["u"]:
    E2_dfs_parab[ov].plot(ax=axs, x="time", y=ov, linewidth=0.8, label=f"{ov} (V)")

axs.legend(fontsize="x-small")
axs.yaxis.set_major_locator(AutoLocator())
axs.yaxis.set_minor_locator(AutoMinorLocator())
axs.xaxis.set_major_locator(AutoLocator())
axs.xaxis.set_minor_locator(AutoMinorLocator())
axs.grid(linestyle='--', linewidth=0.5, which="major")
axs.grid(linestyle='-.', linewidth=0.1, which="minor")
axs.set_xlabel("time (s)")
axs.set_title("Real motor parabola response with E2 controller")
plt.savefig('e2-parab.pdf', bbox_inches='tight')
# %%
generate_csv_sut(system_names, sut_coef, sut_parametric, sut)
generate_tex_tables_sut(system_names, sut_coef, sut_parametric, sut)