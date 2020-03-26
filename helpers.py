import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


START_DATE = '1/22/20'


def exp_T(x, A, T, Off=0):
    return A * np.exp( np.log(2) * x / T ) + Off


def gauss(x, A, sigma, center):
    return A * np.exp(- (x - center)**2 / (2 * sigma**2 ))


C_PRIO = ['Germany', 'US', 'Italy', 'China']
def country_prio(c):
    if c in C_PRIO:
        return 0, C_PRIO.index(c)
    else:
        return 1, c

    
class CovidPlot(object):
    
    def __init__(self):
        self.dataframe = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
        self.countries = sorted(list(set(self.dataframe['Country/Region'])),
                                key=country_prio)
        fig = plt.figure(num='COVID-19 Cases')
        self.ax_tot = fig.add_axes([0.1, 0.4, 0.88, 0.56], label='Horst')
        self.ax_tot.set_title('Wurst')
        self.ax_new = fig.add_axes([0.1, 0.05, 0.88, 0.26])
        self.ax_new.set_title('Wurst')
        rows = self.dataframe.loc[:,START_DATE:]
        cols = rows.keys()
        self.dates = pd.to_datetime(cols)
        self.days = np.arange(len(self.dates))
        self.fig = fig

    
    def plot_country(self, country, future_weeks=26,
                     gaussfit=False,
                     plot_initial=False
                    ):
        self.ax_tot.clear()
        self.ax_new.clear()

        dataframe = self.dataframe[self.dataframe['Country/Region'] == country].loc[:,START_DATE:]
        df = pd.DataFrame(dataframe.sum(), self.dates)
        A, T, Off = curve_fit(exp_T, self.days, df[0])[0]
        p0 = [10**4, 8, 60]
        bounds = ([1000, 1, 0], [1E7, 1E4, 1E4])
        if gaussfit:
            try:
                A_g, sigma, center = curve_fit(gauss, self.days, df[0], p0, bounds=bounds)[0]
                print(A_g, sigma, center)
            except RuntimeError:
                gaussfit = False
        dates_proj = pd.date_range(self.dates[0], self.dates[-1] + pd.Timedelta(weeks=future_weeks))
        days_proj = np.arange(len(dates_proj))
        
        proj = pd.DataFrame(exp_T(days_proj, A, T, Off))
        self.ax_tot.plot(df.index, df, 'o',
                         dates_proj, proj
                        )
        if plot_initial:
            self.ax_tot.plot(dates_proj, gauss(days_proj, *p0), '--', alpha=0.2)
        if gaussfit:
            sim = gauss(days_proj, A_g, sigma, center)
            self.ax_tot.plot(dates_proj, sim, '--')
            self.ax_tot.set_ylim([0, sim.max()])
        #else:
        #    self.ax_tot.set_ylim([0, df[0].max()*1.1])
        self.ax_new.plot(self.dates, df[0].diff(), 'o')
        self.ax_new.plot(dates_proj, proj.diff())
        self.ax_tot.text(0.25, 0.9,
                         'Total # of cases\n# cases doubles every {0:.1f} days'.format(T),
                         transform=self.ax_tot.transAxes, bbox={'facecolor': 'lightgrey'},
                         horizontalalignment='center', verticalalignment='center')
        self.ax_new.text(0.25, 0.9,
                         '# of new cases per day',
                         transform=self.ax_new.transAxes, bbox={'facecolor': 'lightgrey'},
                         horizontalalignment='center', verticalalignment='center')
