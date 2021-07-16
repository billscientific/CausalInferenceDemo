# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # #   Little demo script for playing with the plots
# # # #   presented in the Causal Inference meeting
# # # #   on July 16, 2021. Written by Johannes Bill.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # #  Usage: 1. Choose model parameters 
# # # #         2. Goto  "-->  SELECT WHAT TO PLOT  <-- " , below,
# # # #            to choose what to plot.
# # # #         
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
import pylab as pl
import mpl_defaults
pl.rc('figure', dpi=2*pl.rcParams['figure.dpi'])  # Zoom in on screen w/o affecting saved figures
np.random.seed(1928373) # makes samples reproducible

# # # # # # # # # # # # # # # # # # # # 
# # #        MODEL PARAMETERS     # # #
# # # # # # # # # # # # # # # # # # # # 
sig2v = 0.25**2          # Variance of visual percept 
sig2a = 1.00**2          # Variance of auditory percept
sig2p = 2.0**2           # Prior variance about sound source(s); mean is fixed at 0
P1 = 0.50                # Prior belief in C=1 (= one sound source); called p_common in Koerding et al., 2007

sig2err = 0.20**2        # Reporting error (think of it as motor error)

# # #   END OF MODEL PARAMETERS

# # #  Plot params # # #
defaultKwargs = dict(lw=1.)
SAVEFIG = True           # Set to True to save a series of plots to file
fname = lambda id, n: f"fig_{id}_{n:d}.png"

# # #   ALL THE INTERESTING STUFF GOES HERE  # # #

# Observations
Xv = np.linspace(-5, 5, 501)        # x-values for x_v
Xa = np.zeros(501)                  # auditory data are fixed

# # # Derived quantities from Koerding et al., 2007
LC1 = 1 / ( 2*np.pi * np.sqrt(sig2v*sig2a + sig2v*sig2p + sig2a*sig2p) ) * \
      np.exp( -1/2 * ( (Xv-Xa)**2 * sig2p + (Xv-0.)**2 * sig2a + (Xa-0.)**2 * sig2v ) / \
                     ( sig2v*sig2a + sig2v*sig2p + sig2a*sig2p )  )               # eqn  4 in Koerding et al., 2007

LC2 = 1 / ( 2*np.pi * np.sqrt( (sig2v + sig2p) * (sig2a + sig2p) ) ) * \
      np.exp( -1/2 * ( (Xv-0.)**2 / (sig2v + sig2p) + (Xa-0.)**2 / (sig2a + sig2p) ) )  # eqn  6 in Koerding et al., 2007

PC1 = LC1 * P1 / ( LC1 * P1 + LC2 * (1 - P1) )                                    # eqn  2 in Koerding et al., 2007
SHatAC1 = ( Xv/sig2v + Xa/sig2a + 0./sig2p ) / ( 1/sig2v + 1/sig2a + 1/sig2p )    # eqn 12 in Koerding et al., 2007
SHatAC2 = ( Xa/sig2a + 0./sig2p ) / ( 1/sig2a + 1/sig2p )                         # eqn 11 in Koerding et al., 2007

# # #  Selection models

SHatAMap = PC1 * SHatAC1 + (1 - PC1) * SHatAC2                                    # Take the mean; eqn 10 in Koerding et al., 2007
errMap = np.sqrt(sig2err) * np.ones(len(Xv))                                      # Std for Take the mean
SHatMAPs = np.random.normal(SHatAMap, np.sqrt(sig2err))                           # Examples for Take the mean

SHatAMs = (PC1 > 0.5) * SHatAC1 + (PC1 <= 0.5) * SHatAC2                          # Most likely mode

SHatAPm = SHatAMap                                                                # Sampled mode
errPm =  np.sqrt(sig2err + PC1 * SHatAC1**2 + (1 - PC1) * SHatAC2**2 - SHatAMap**2) # Std for sampled mode
SHatPMs = np.array([ np.array([np.random.normal(SHatAC2, np.sqrt(sig2err)), np.random.normal(SHatAC1, np.sqrt(sig2err))])[int(s),i] for i,s in enumerate(np.random.rand(len(PC1)) < PC1) ])  # Examples for Sampled mode

# # #  Collection of plots  # # #

# sa vs sv
plotParamSaXv = {
    'id' : "sasv", # for file names
    'title' : "If auditory observation $x_a = 0$:" ,
    'x' : dict(val = Xv, lim = (-3.0, 3.0), label = "Observed visual cue $x_v$"),
    'y' : { 'lim' : (-3.0, 3.0) ,
            'label' : "Avg. auditory response $\hat s_a$",
            'identity' : dict(val = Xv, label=None, kwargs={'color' : '0.8', 'ls' : ':', 'lw' : 0.75}),
            'C1' : dict(val = SHatAC1, label="One cause (C=1)", kwargs={'color' : '0.5', 'ls' : '--', 'lw' : 0.75}),
            'C2' : dict(val = SHatAC2, label="Two causes (C=2)", kwargs={'color' : '0.5', 'ls' : ':', 'lw' : 0.75}),
            'MS' : dict(val = SHatAMs, label="Most likely mode", kwargs={'color' : 'b'}),
            'MAP' : dict(val = SHatAMap, label="Take the mean", kwargs={'color' : 'r'}),
            'MAPe' : dict(val = SHatAMap, err=errMap, label=None, kwargs={'color' : 'r'}),
            'MAPs' : dict(val = SHatMAPs, label=None, kwargs={'color' : 'r', 'marker' : '.', 'ms' : 1., 'lw' : 0.}),
            'PM' : dict(val = SHatAPm, label="Sampled mode", kwargs={'color' : 'g'}),
            'PMs' : dict(val = SHatPMs, label=None, kwargs={'color' : 'g', 'marker' : '.', 'ms' : 1., 'lw' : 0.}),
            'PMe' : dict(val = SHatAPm, err=errPm, label=None, kwargs={'color' : 'g'}),
            'PMb1' : dict(val = SHatAC1, err=errMap, label="Probability matching", kwargs={'color' : 'g'}),
            'PMb2' : dict(val = SHatAC2, err=errMap, label=None, kwargs={'color' : 'g'}),
    }
}

# PC1 vs xv
plotParampc1xv = {
    'id' : "pc1xv", # for file names
    'title' : "If auditory observation $x_a = 0$:" ,
    'x' : dict(val = Xv, lim = (-3.0, 3.0), label = "Observed visual cue $x_v$"),
    'y' : { 'lim' : (0, 1.0) ,
            'label' : "Belief one cause, $P(C{=}1 \,|\, x_v,\, x_a{=}0)$",
            'PC1' : dict(val = PC1, label=None, kwargs={'color' : 'b'}),
    }
}


# # # # # # # # # # # # # # # # # # # # # #
# # #  -->  SELECT WHAT TO PLOT  <--  # # # 
# # # # # # # # # # # # # # # # # # # # # #

# # #  Usage: (Un-)comment the examples, below

# # #  (1) Plot the causal belief p(C=1 | x_v, x_a=0)
plotParam = plotParampc1xv
plotKeys = ('PC1',)

# # #  (2) Plot the average reported location for different models (--> look at saved figures)
# plotParam = plotParamSaXv
# plotKeys = ('C2', 'C1', 'MS', 'MAP', 'PM')

# # #  (3) Plot examples of reported location for the "Sampled mode" model
# plotParam = plotParamSaXv
# plotKeys = ('C2', 'C1', 'PM', 'PMs')


# # # #  END OF PLOT SELECTION  # # # #

# # #  PLOTTING  # # #
fig = pl.figure(figsize=(3,2.25))
pp = plotParam
pl.xlabel(pp['x']['label'])
pl.ylabel(pp['y']['label'])
pl.title(pp['title'], pad=2)
pl.subplots_adjust(0.15, 0.15, 0.97, 0.93)

artists = []
for key in plotKeys:
    kwargs = defaultKwargs.copy()
    kwargs.update(pp['y'][key]['kwargs'])
    x, y = pp['x']['val'], pp['y'][key]['val']
    l = pl.plot(x, y, label=pp['y'][key]['label'], **kwargs)[0]
    if 'err' in pp['y'][key]:
        err = pp['y'][key]['err']
        kwargs['alpha'] = 0.25
        kwargs['lw'] = 0.
        a = pl.fill_between(x, y-err, y+err, **kwargs)
        l = [l, a]
    artists.append(l)

pl.xlim(pp['x']['lim'])
pl.ylim(pp['y']['lim'])
leg = pl.legend(loc='upper left', fontsize=6)
if leg.get_lines() == []:
    leg.set_visible(False)


# make the artists visible one by one in saved files
if SAVEFIG:
    for i,artist in enumerate(reversed(artists)):
        n = len(artists) - i
        fig.savefig(fname(pp['id'], n))
        if np.iterable(artist):
            for a in artist:
                a.set_visible(False)
        else:
            artist.set_visible(False)
    fig.savefig(fname(pp['id'], 0))
    # And make visible on the screen again
    for artist in artists:
        if np.iterable(artist):
            for a in artist:
                a.set_visible(True)
        else:
            artist.set_visible(True)
pl.show()


