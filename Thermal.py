import pandas as pd
import numpy as np
import seaborn as sn
from matplotlib import pyplot as plt
from scipy import signal
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)



plt.xlabel('T[K]')

def data_graph(T_data,Cp_R_data):
    plt.figsize=set_size(100)
    plt.ylabel('$\Delta C_p$[R]')
    plt.ylim(-0.3,18)
    plt.vlines([673],-0.3,17.5,linestyles='--',color='black',linewidth=1)
    plt.plot(T_data,Cp_R_data,color ='black',linewidth=0.5)
    plt.show()
def scaled_graph(T_data,Cp_R_data):
    Cp_R_T_data = Cp_R_data/T_data
    peak_position = signal.find_peaks(Cp_R_T_data,0.025)[0]
    print(peak_position)
    plt.figsize=set_size(100)

    plt.plot(T_data,Cp_R_T_data,color ='black',linewidth=0.5)
    plt.ylabel('$\Delta C_p/T$[R/K]')
    plt.xlabel('T[K]')

    plt.ylim(-0.0003,0.030)
    plt.vlines([673],-0.0003,0.025,linestyles='--',color='black',linewidth=1)
    plt.savefig('cp_R_T.pgf')
    plt.show()
def integrator(T_data, Cp_R_T_data, Tf):
    result = 0
    for i in range(len(T_data)):
        if T_data[i+1] == Tf:
            break
        else:
            dx = T_data[i+1] - T_data[i]
            result += Cp_R_T_data[i]*dx
    return result
def deltaS(T_data,Cp_R_T_data):
    return  [integrator(T_data, Cp_R_T_data, T) for T in T_data[1:]]

def entropy_graph(T_data,deltasS):
    plt.figsize=set_size(100)
    plt.ylabel('$\Delta S[R]')
    plt.xlabel('T[K]')
    plt.plot(T_data[1:],deltasS,color ='black',linewidth=0.5)
    plt.ylim(-0.01,0.46)
    plt.savefig('DeltaS.pgf')
def log(t,a,c,d):
    return a*np.log(t) + c*t+d

def cuadratic(t,a,b,c):
    return a*t**2 +b*t + c

def extrapolate(deltsS,T_data):
    x =np.array(deltsS)[np.array(deltsS)>0.42]
    y = (len(x))
    print(y)
    t = T_data[len(T_data)-1-y+1:]
    print(len(t))
    plt.plot(t,x)
    plt.show()
    popt, pcov = scipy.optimize.curve_fit(cuadratic, t, x)
    z = np.linspace(680.25,720,1000)
    popt, pcov = scipy.optimize.curve_fit(log, t, x)
    l = log(z,popt[0],popt[1],popt[2])
    plt.plot(z,l)
    plt.show()
    return popt
def extrapolation_graph(popt,deltsS,T_data):
    z = np.linspace(680.25,900,1000)
    l = log(z,popt[0],popt[1],popt[2])
    plt.plot(T_data[1:],deltsS)
    plt.plot(z,l)
    print(log(900,popt[0],popt[1],popt[2]))

def main():
    df = pd.read_excel('Cu3AuData.xlsx')
    T_data = df['T/K']
    Cp_R_data = df['ΔC_p/R']