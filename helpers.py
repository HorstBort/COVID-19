import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


START_DATE = '1/22/20'

# From: https://de.wikipedia.org/wiki/COVID-19-Pandemie_in_Deutschland
EVENTS = {'Germany': {pd.to_datetime(k): v
                      for k, v in {#'3/19/20': 'DLR',
                                   '3/8/20': {'title': 'No more big events', 'color': 'lightpink'},
                                   '3/17/20': {'title': 'Borders/shops closed', 'color': 'orange'},
                                   '3/22/20': {'title': 'Lockdown', 'color': 'crimson'},
                                   '3/26/20': {'title': 'Test criteria broadened', 'color': 'darkcyan', 'type': 'single'},
                                   '4/20/20': {'title': 'Small shops opening', 'color': 'palevioletred'}
                                   }.items()
                     }
         }

# From: https://de.wikipedia.org/wiki/COVID-19-Pandemie_in_Deutschland#Testkapazit%C3%A4ten_und_Anteil_positiver_Ergebnisse
# https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Situationsberichte/Gesamt.html
NR_TESTS_PER_CALWEEK = {'Germany': {k: v/7.0 for k, v in {11: 127457, 12: 348619, 13: 361515, 14: 408348, 15: 379233, 16: 330027,
                                                          17: 357876, 18: 317979, 19: 382154}.items()}}

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


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
        self.dataframe_rec = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
        self.dataframe_deaths = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
        self.countries = sorted(list(set(self.dataframe['Country/Region'])),
                                key=country_prio)
        fig = plt.figure(num='COVID-19 Cases', figsize=(10, 6))
        self.ax_tot = fig.add_axes([0.1, 0.4, 0.82, 0.56], label='Horst')
        self.ax_tot.set_title('Wurst')
        self.ax_new = fig.add_axes([0.1, 0.05, 0.82, 0.26])
        self.ax_new.set_title('Wurst')
        self.ax_tests = self.ax_new.twinx()
        rows = self.dataframe.loc[:,START_DATE:]
        cols = rows.keys()
        self.dates = pd.to_datetime(cols)
        self.weekends = pd.DataFrame([1 if d.weekday() > 4 else 0 for d in self.dates], self.dates)
        self.tests_per_day = {'Germany': pd.DataFrame([NR_TESTS_PER_CALWEEK.get('Germany', {}).get(d.weekofyear) for d in self.dates], self.dates)}
        self.days = np.arange(len(self.dates))
        self.fig = fig
        fig.show()

    
    def plot_country(self, country,# future_weeks=26,
                     log_tot=False,
                     log_new=False
                     #gaussfit=False,
                     #plot_initial=False
                    ):
        gaussfit = False
        plot_initial = False
        self.ax_tot.clear()
        self.ax_new.clear()
        self.ax_tests.clear()

        dataframe = self.dataframe[self.dataframe['Country/Region'] == country].loc[:,START_DATE:]
        dataframe_rec = self.dataframe_rec[self.dataframe_rec['Country/Region'] == country].loc[:,START_DATE:]
        dataframe_deaths = self.dataframe_deaths[self.dataframe_deaths['Country/Region'] == country].loc[:,START_DATE:]
        df = pd.DataFrame(dataframe.sum(), self.dates)
        df_rec = pd.DataFrame(dataframe_rec.sum(), self.dates)
        df_deaths = pd.DataFrame(dataframe_deaths.sum(), self.dates)
        #A, T, Off = curve_fit(exp_T, self.days, df[0])[0]
        p0 = [10**4, 8, 60]
        bounds = ([1000, 1, 0], [1E7, 1E4, 1E4])
        if gaussfit:
            try:
                A_g, sigma, center = curve_fit(gauss, self.days, df[0], p0, bounds=bounds)[0]
                print(A_g, sigma, center)
            except RuntimeError:
                gaussfit = False
        #dates_proj = pd.date_range(self.dates[0], self.dates[-1] + pd.Timedelta(weeks=future_weeks))
        #days_proj = np.arange(len(dates_proj))
        
        #proj = pd.DataFrame(exp_T(days_proj, A, T, Off))
        self.ax_tot.plot(df, 'o', label='Confirmed cases', markersize=3)
        self.ax_tot.plot(df_rec, 'og', label='Recovered (JHU guesstimate)', zorder=-5, markersize=3)
        self.ax_tot.plot(df_deaths, 'ok', label='Deaths', zorder=-10, markersize=3)
        self.ax_tot.plot(df - df_rec - df_deaths, 'o', c='magenta', label='Currently infected (JHU guesstimate)', zorder=-2, markersize=3)
        self.ax_tot.plot(df.diff().rolling(window=14).sum(), 'o', c='skyblue', label='Currently infected (my guesstimate)', zorder=-2, markersize=3)
        
                         #dates_proj, proj
        self.ax_tot.plot(self.tests_per_day.get(country, []), 's', label='# Tests per day', color='darkcyan', alpha=0.5, markersize=3)
        
        events = EVENTS.get(country)
        if events is not None:
            evs_span = sorted(list(events.items()))
            for ii, (ev_date, ev) in enumerate(evs_span):
                ev_title = ev['title']
                col = ev['color']
                ev_end = evs_span[ii + 1][0] if ii < len(evs_span) - 1 else self.dates[-1]
                ev_end = self.dates[-1]
                if ii < len(evs_span) - 1:
                    d_fol = [d for d, e in evs_span[ii + 1:] if not e.get('type') == 'single']
                    if d_fol:
                        ev_end = d_fol[0]
                self.ax_tot.axvline(ev_date, label=ev_title, color=col, alpha=0.5)
                self.ax_tot.axvline(ev_date + pd.Timedelta(weeks=2), label='+ two weeks', color=col, linestyle='--', alpha=0.3)
                
                if not ev.get('type') == 'single':
                    self.ax_tot.axvspan(ev_date, ev_end, color=col, alpha=0.1, zorder=-20)
                
                self.ax_new.axvline(ev_date, color=col, alpha=0.5)
                self.ax_new.axvline(ev_date + pd.Timedelta(weeks=2), color=col, linestyle='--', alpha=0.3)
                #self.ax_tot.text(ev_date, 0.9 * df.max(), ev_title)
                
        if plot_initial:
            self.ax_tot.plot(dates_proj, gauss(days_proj, *p0), '--', alpha=0.2)
        if gaussfit:
            sim = gauss(days_proj, A_g, sigma, center)
            self.ax_tot.plot(dates_proj, sim, '--')
            self.ax_tot.set_ylim([0, sim.max()])
        #else:
        #    self.ax_tot.set_ylim([0, df[0].max()*1.1])
        new_cases = df.diff()
        self.ax_new.plot(self.dates, new_cases , 'o', label='New conf. cases', markersize=3)
        self.ax_new.fill_between(self.dates, self.weekends[0] * new_cases[0].max(), step='pre', alpha=0.2, label='Weekends')
        self.ax_tests.plot(self.tests_per_day.get(country, []), 's', label='# Tests per day', color='darkcyan', alpha=0.5, markersize=3)
        #self.ax_new.plot(dates_proj, proj.diff())
        #self.ax_tot.text(0.25, 0.9,
        #                 'Total # of cases\n# cases doubles every {0:.1f} days'.format(T),
        #                 transform=self.ax_tot.transAxes, bbox={'facecolor': 'lightgrey'},
        #                 horizontalalignment='center', verticalalignment='center')
        #self.ax_new.text(0.25, 0.9,
        #                 '# of new cases per day',
        #                 transform=self.ax_new.transAxes, bbox={'facecolor': 'lightgrey'},
        #                 horizontalalignment='center', verticalalignment='center')
        
        self.ax_tot.legend(loc='center left', fontsize='x-small')
        self.ax_new.legend()
        self.ax_tests.legend(loc='center left')
        self.ax_tests.tick_params(axis='y', labelcolor='darkcyan')
        if log_tot:
            self.ax_tot.set_yscale('log')
        if log_new:
            self.ax_new.set_yscale('log')
            self.ax_tests.set_yscale('log')
            self.ax_tests.set_ylim([10, self.ax_tests.get_ylim()[1]])
        else:
            self.ax_tests.set_ylim([0, self.ax_tests.get_ylim()[1]])
        
            