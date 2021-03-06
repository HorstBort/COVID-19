## COVID-19 Data Explorer

[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://notebooks.gesis.org/binder/v2/gh/HorstBort/COVID-19.git/master?filepath=COVID-19.ipynb)

Fetches current data from https://github.com/CSSEGISandData/COVID-19 (Johns Hopkins University, updated daily).

Fits exponential function and plots current data and projection.

Explore by selecting country and weeks for projection.

**HOWTO**

Select next cell and click `run` above or press `Shift+Enter`.

**Experimental**:

 * Select `gaussfit` to fit Gaussian to data (very rough estimate, since we are just at the beginning of the curve :-/)
 * Select `plot_initial` to plot initial guess for gaussian fit.