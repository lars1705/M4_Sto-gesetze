import numpy as np
from uncertainties import ufloat, unumpy, std_dev
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import curve_fit
from wheel.cli import unpack_f

unc = 2 * 0.1  #cm
dreiecksv = unc / np.sqrt(3)

steigungen = []

import pandas as pd

def mittelwert_t1(header:str, y_0:float, path:str="daten1.csv", c:int=1):
    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].str.replace(",", ".").astype(float)
    print(df)
    m_werte1 = []

    for i in range(5):
        auslenkungen_res = []
        for j in range(5):
            if c==1:
                auslenkungen_res.append(ufloat(df[header][j+5*i]-y_0, dreiecksv))
            elif c==2:
                auslenkungen_res.append(ufloat(y_0-df[header][j + 5 * i], dreiecksv))
        print(auslenkungen_res)
        m_wert = sum(auslenkungen_res) / len(auslenkungen_res)
        m_werte1.append(m_wert)
    print("-----", m_werte1)
    return m_werte1


dreicksv_2 = 0.1 / np.sqrt(3) + 0.25  # in cm, noch besprechen

def mittelwert_t2(header:str="Auslenkung 2 (resultierend) T2", path:str="daten2.csv"):
    m_werte2 = []

    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].astype(str).str.replace(",", ".").astype(float)

    m_werte2 = []
    for i in range(5):
        auslenkungen_res = []
        for j in range(5):
            auslenkungen_res.append(ufloat(df[header][j + 5 * i]-26.4, dreicksv_2))
        print(auslenkungen_res)
        m_wert = sum(auslenkungen_res) / len(auslenkungen_res)
        m_werte2.append(m_wert)
    print("\n>>>>>>>>>>>>",m_werte2)
    return m_werte2

def f(x, m, n):
    return m*x+n


m_werte1 = mittelwert_t1("Auslenkung 2 (resultierend) T1", y_0=26.4)


def plot_ausgleichsgerade(x_name:str, mittelwerte_y:list, x_0:float, c:int, path:str="daten1.csv"):

    unc = .1/np.sqrt(3)

    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].str.replace(",", ".").astype(float)
    if c==1:
        x1 = [ufloat(x_0-v, unc) for v in df[x_name][::5]]
    elif c==2:
        x1 = [ufloat(v-x_0, unc) for v in df[x_name][::5]]
    print(x1)
    #y1 = [ufloat(v, unc) for v in df[y_name]]
    y1 = mittelwerte_y
    print(mittelwerte_y)
    x1_nominal = nominal_values(x1)
    y1_nominal = nominal_values(y1)
    x1_err = std_devs(x1)
    y1_err = std_devs(y1)

    params, params_covariance = curve_fit(f, x1_nominal, y1_nominal, p0=[1, 0], sigma=y1_err, absolute_sigma=True)
    m_fit, n_fit = ufloat(params[0], np.sqrt(params_covariance[0, 0])), ufloat(params[1],
                                                                               np.sqrt(params_covariance[1, 1]))

    steigungen.append(m_fit)

    #plt.figure(figsize=(15, 7))
    fig, ax = plt.subplots(figsize=(15, 7))
    print(x1_err, y1_err)
    plt.errorbar(x1_nominal, y1_nominal, xerr=x1_err, yerr=y1_err, marker='.', markersize=8, linestyle="none", label="Messdaten")
    #plt.errorbar(zeiten_nom, positionen_nom, yerr=y1_err,fmt=".", label='Messdaten')
    plt.plot(x1_nominal, f(x1_nominal, unumpy.nominal_values(m_fit), unumpy.nominal_values(n_fit)), 'r-',
            label=f'Ausgleichsgerade: {m_fit} * x  + {n_fit}')
    plt.tick_params(axis='both', labelsize="16")
    plt.xlabel('Auslenkung $x_1$ [cm]', fontsize="16")
    plt.ylabel('Auslenkung $x_2$ (resultierend) [cm]', fontsize="16")
    plt.legend(fontsize="17")
    plt.grid(True)
    plt.show()


def get_h(h_0, s, L, H):
    return h_0 - s / ((1+ (L / H)**2)**(1/2))



def plot_rollende_Kugel(path="daten2.csv"):
    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].astype(str).str.replace(",", ".").astype(float)

    fallhöhen = df["Auslenkung 1 T2"][::5]
    unc_fh = 0.25/np.sqrt(3)
    fallhöhen = [ufloat(fh, unc_fh) for fh in fallhöhen]

    auslenkungen = mittelwert_t2()

    fallhöhen_nom = ...



if __name__ == "__main__":
    #m_werte2 = mittelwert_t1("Auslenkung 4 T1")
    mittelwert_t2()

    #plot_ausgleichsgerade("Auslenkung 1 T1", mittelwerte_y=m_werte1, x_0=17.62, c=1)
    #plot_ausgleichsgerade("Auslenkung 3 T1", mittelwerte_y=mittelwert_t1("Auslenkung 4 T1", 17.62, c=2), x_0=26.4, c=2)
#
    #print(steigungen, sum(steigungen))

    H = ufloat(33.25, .25)
    h_0 = H-ufloat(3.1, .25)
    L = ufloat(58.8, .25) - ufloat(1, .25)
    abstaende_s = [50,40,30,20,10,0]
    for abstand in abstaende_s:
        print(get_h(h_0,ufloat(abstand, .25), L, H))




