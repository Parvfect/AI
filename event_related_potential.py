from scipy.io import loadmat
from pylab import *
from IPython.lib.display import YouTubeVideo  # Enable YouTube videos
rcParams['figure.figsize']=(12,3)  # Change the default figure size


data = loadmat('matfiles/02_EEG-1.mat')         # Load the data,
EEGa = data['EEGa']                             # ... and get the EEG from one condition,
t = data['t'][0]                                # ... and a time axis,
ntrials = len(EEGa)                             # ... and compute the number of trials.

mn = EEGa.mean(0)                               # Compute the mean signal across trials (the ERP).
sd = EEGa.std(0)                                # Compute the std of the signal across trials.
sdmn = sd / sqrt(ntrials)                       # Compute the std of the mean.

plot(t, mn, 'k', lw=3)                          # Plot the ERP of condition A,
plot(t, mn + 2 * sdmn, 'k:', lw=1)              # ... and include the upper CI,
plot(t, mn - 2 * sdmn, 'k:', lw=1)              # ... and the lower CI.
xlabel('Time [s]')                              # Label the axes,
ylabel('Voltage [$\mu$ V]')
title('ERP of condition A')                     # ... provide a useful title,
show()       