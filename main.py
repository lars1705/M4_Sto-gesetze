import numpy as np
#from Cython.Shadow import returns
from uncertainties import ufloat, unumpy, std_dev
import matplotlib.pyplot as plt
from uncertainties.unumpy import nominal_values, std_devs
from scipy.optimize import curve_fit
import pandas as pd

#from wheel.cli import unpack_f

#unc = 2 * 0.1  #cm
dreiecksv_auslenkung_messstab = 0.1 / np.sqrt(6)  # in cm
messunsicherheit_waage = .0001 / np.sqrt(6)  #kg
messunsicherheit_waage_hersteller = 0  #kg
messunsicherheit_waage_ges = std_dev(ufloat(1, messunsicherheit_waage)+ufloat(1,messunsicherheit_waage_hersteller))
dreicksv_messband_1 = 0.025  # in cm
dreiecksv_messband_2 = dreiecksv_auslenkung_messstab
dreiecksv_uns_messband_ges = std_dev(ufloat(1, dreicksv_messband_1)+ufloat(1,dreiecksv_messband_2))
#print(dreiecksv_uns_messband_ges)

steigungen = []


H = ufloat(33.25, dreiecksv_uns_messband_ges) * .01  #in m
h_0 = H-ufloat(3.1, dreiecksv_uns_messband_ges) * .01  #in m
L = (ufloat(58.8, dreiecksv_uns_messband_ges) - ufloat(1, dreiecksv_uns_messband_ges)) *.01  #in m

abstaende_s = [0, 10, 20, 30, 40, 50]  #in cm
abstaende_s = [ufloat(abstand, dreiecksv_uns_messband_ges) * .01 for abstand in abstaende_s]  #in m, mit MU


l = ufloat(189.1, dreiecksv_uns_messband_ges) * .01  # in m

M_1 = ufloat(176.7 * .001, messunsicherheit_waage_ges)
M_2 = ufloat(490.7 * .001, messunsicherheit_waage_ges) #kg
M_3 = ufloat(63.9 * .001, messunsicherheit_waage_ges)


def mittelwert_t1(header:str, y_0:float, header_2:str, path:str="daten1.csv", c:int=1):
    y_0 = ufloat(y_0, dreiecksv_auslenkung_messstab)
    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].str.replace(",", ".").astype(float)
    #print(df)
    m_werte1 = []

    for i in range(5):
        auslenkungen_res = []
        for j in range(5):
            if c==1:
                auslenkungen_res.append(ufloat(df[header][j+5*i], dreiecksv_auslenkung_messstab) - y_0)
            elif c==2:
                auslenkungen_res.append(y_0-ufloat(df[header][j + 5 * i], dreiecksv_auslenkung_messstab))
            #print(std_devs(auslenkungen_res))

        #print(auslenkungen_res)
        m_wert = sum(auslenkungen_res) / len(auslenkungen_res)
        print(std_devs(sum(auslenkungen_res)))
        m_werte1.append(m_wert)
    print("-----", m_werte1)



    data = []
    for i, x in enumerate(df[header_2][::5].values):
        if c==1:
            data.append([ufloat(17.62, dreiecksv_auslenkung_messstab)-ufloat(x, dreiecksv_auslenkung_messstab), m_werte1[i]])
        elif c==2:
            data.append([ufloat(x, dreiecksv_auslenkung_messstab)-ufloat(26.4, dreiecksv_auslenkung_messstab), m_werte1[i]])
    df_2 = pd.DataFrame(data, columns=["Manuelle Auslenkung [cm]", "Mittelwert resultierende Auslenkung [cm]"])


    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df_2.values, colLabels=[r"Manuelle Auslenkung $a_2$ [cm]", r"Mittelwert resultierende Auslenkung $\bar{a}_1'$ [cm]"], loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(25)
    fig.tight_layout()
    plt.show()

    return m_werte1



def mittelwert_t2(header:str="Auslenkung 2 (resultierend) T2", path:str="daten2.csv"):
    m_werte2 = []

    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].astype(str).str.replace(",", ".").astype(float)

    for i in range(6):
        auslenkungen_res = []
        for j in range(5):
            auslenkungen_res.append(ufloat(df[header][j + 5 * i], dreiecksv_auslenkung_messstab)-ufloat(26.4, dreiecksv_auslenkung_messstab))
        #print(auslenkungen_res)
        m_wert = sum(auslenkungen_res) / len(auslenkungen_res)
        m_werte2.append(m_wert)
    print("\n>>>>>>>>>>>>",m_werte2)



    data = []
    for i, x in enumerate(df["Auslenkung 1 T2"][::5].values):
        data.append(
            [ufloat(x, dreiecksv_uns_messband_ges), m_werte2[i]])

    df_2 = pd.DataFrame(data, columns=["Manuelle Auslenkung [cm]", "Mittelwert resultierende Auslenkung [cm]"])
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df_2.values, colLabels=[r"Abstand vom oberen Ende der Fallrinne $s$ [cm]",
                                                      r"Mittelwert resultierende Auslenkung $\bar{a}'$ [cm]"],
                     loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(25)
    fig.tight_layout()
    plt.show()

    return m_werte2

def f(x, m, n):
    return m*x+n



def plot_ausgleichsgerade(x_name:str, mittelwerte_y:list, x_0:float, c:int, path:str="daten1.csv"):
    x_0 = ufloat(x_0, dreiecksv_auslenkung_messstab)
    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].str.replace(",", ".").astype(float)
    if c==1:
        x1 = [x_0-ufloat(v, dreiecksv_auslenkung_messstab) for v in df[x_name][::5]]
    elif c==2:
        x1 = [ufloat(v, dreiecksv_auslenkung_messstab)-x_0 for v in df[x_name][::5]]
    #print(x1)
    #y1 = [ufloat(v, dreiecksv_auslenkung_maß) for v in df[y_name]]
    y1 = mittelwerte_y
    #print(mittelwerte_y)
    x1_nominal = nominal_values(x1)
    y1_nominal = nominal_values(y1)
    x1_err = std_devs(x1)
    y1_err = std_devs(y1)
    print(y1)

    params, params_covariance = curve_fit(f, x1_nominal, y1_nominal, p0=[1, 0], sigma=y1_err, absolute_sigma=True)
    m_fit, n_fit = ufloat(params[0], np.sqrt(params_covariance[0, 0])), ufloat(params[1],
                                                                               np.sqrt(params_covariance[1, 1]))

    steigungen.append(m_fit)

    plt.figure(figsize=(15, 7))  #reset zu (15, 7) für legende nicht über md
    #print(x1_err, y1_err)
    plt.errorbar(x1_nominal, y1_nominal, xerr=x1_err, yerr=y1_err, marker='.', markersize=4, linestyle="none", label="Messdaten mit Unsicherheiten")
    #plt.errorbar(zeiten_nom, positionen_nom, yerr=y1_err,fmt=".", label='Messdaten')
    plt.plot(x1_nominal, f(x1_nominal, unumpy.nominal_values(m_fit), unumpy.nominal_values(n_fit)), 'r-',
            label=f'Ausgleichsgerade: {m_fit} * x  + {n_fit}')
    plt.tick_params(axis='both', labelsize="35")
    plt.xlabel('Auslenkung $a_2$ [cm]', fontsize="35")
    plt.ylabel(r"Auslenkung $a_1'$ (resultierend) [cm]", fontsize="35")
    plt.legend(fontsize="35")
    plt.grid(True)
    plt.show()


def get_h(h_0, s, L, H):
    return h_0 - s / ((1+ (L / H)**2)**(1/2))

def get_epsilon(m, l, m_1=M_3, m_2=M_2):
    return m**2 * ((m_1+m_2)**2) / (8*l*m_1**2)



def plot_rollende_Kugel(path="daten2.csv", show_figure=True):
    df = pd.read_csv(path, delimiter=",")
    for column in df:
        df[column] = df[column].astype(str).str.replace(",", ".").astype(float)

    fallhöhen = abstaende_s[::-1]
    h_s = [(get_h(h_0=h_0, s=s, L=L, H=H))**(1/2) *100 for s in fallhöhen]  #in sqrt cm
    auslenkungen = np.array(mittelwert_t2())  #in cm

    hs_nom = nominal_values(h_s)
    auslenkungen_nom = nominal_values(auslenkungen)
    hs_err = std_devs(h_s)
    auslenkungen_err = std_devs(auslenkungen)
    #print(hs_nom, auslenkungen_nom, hs_err, auslenkungen_err)

    params, params_covariance = curve_fit(f, hs_nom, auslenkungen_nom, p0=[1, 0], sigma=auslenkungen_err, absolute_sigma=True)
    m_fit, n_fit = ufloat(params[0], np.sqrt(params_covariance[0, 0])), ufloat(params[1],
                                                                               np.sqrt(params_covariance[1, 1]))

    plt.figure(figsize=(10, 6))  #reset zu (15, 7) für legende nicht über md
    plt.errorbar(hs_nom, auslenkungen_nom, xerr=hs_err, yerr=auslenkungen_err, marker='.', markersize=5, linestyle="none", label="Messdaten")
    #plt.errorbar(zeiten_nom, positionen_nom, yerr=y1_err,fmt=".", label='Messdaten')
    plt.plot(hs_nom, f(hs_nom, unumpy.nominal_values(m_fit), unumpy.nominal_values(n_fit)), 'r-',
            label=f'Ausgleichsgerade: {m_fit} * x - {abs(n_fit)}')
    plt.tick_params(axis='both', labelsize="35")
    plt.xlabel('Fallhöhe als $\sqrt{h(s)}$ [$\sqrt{cm}$]', fontsize="35")
    plt.ylabel('Auslenkung $a$ (resultierend) [cm]', fontsize="35")
    plt.legend(fontsize="35")
    plt.grid(True)

    if show_figure:
        plt.show()

    return m_fit


if __name__ == "__main__":
    mittel_werte1 = mittelwert_t1("Auslenkung 2 (resultierend) T1", 26.4, header_2="Auslenkung 1 T1")
    mittel_werte_2 = mittelwert_t1("Auslenkung 4 T1", 17.62, c=2, header_2="Auslenkung 3 T1")  #c=2 !!


    print(mittel_werte1)
    plot_ausgleichsgerade("Auslenkung 1 T1", mittelwerte_y=mittel_werte1, x_0=17.62, c=1)
    plot_ausgleichsgerade("Auslenkung 3 T1", mittelwerte_y=mittel_werte_2, x_0=26.4, c=2)

    print("steigungen, summe", steigungen, sum(steigungen))


    for abstand in abstaende_s:
        print("h:", get_h(h_0,abstand, L, H)*100)  #in cm


    plot_rollende_Kugel()

    print("epsilon: ", get_epsilon(m=plot_rollende_Kugel(show_figure=False), l=l))
