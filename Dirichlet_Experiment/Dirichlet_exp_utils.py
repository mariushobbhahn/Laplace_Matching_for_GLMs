## this is the utils file for German Elections and BTW Bezirksebene
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

####### Plotting tools ####### 

# basic plots: plot mean of predictions
def plot_elections_mean(sm_result, filename, x, parties, party_colors, save=False, legend=True, n_parties=9):
    
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    legend_size=18
    lw_size=3

    plt.figure(figsize=(10, 5))
    plt.xlim((np.min(x), np.max(x)))
    for i in range(n_parties):
        plt.plot(x, sm_result[:, i], label=parties[i], color=party_colors[parties[i]], lw=3)

    if legend:
        leg = plt.legend(prop={'size': legend_size}) 
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(15.0)
            
    plt.tight_layout()
    if save:
        plt.savefig(filename)
        
    plt.show();
    
# plot means of cumulative results
def plot_elections_cum(sm_results_cum, x, parties, party_colors,  save=False, filename='test.pdf', legend=False, n_parties=9):

    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=20)
    legend_size=18
    lw_size=3
    
    fig = plt.figure(figsize=(10, 5))
    plt.ylim((0.0, 1.00))
    plt.xlim((np.min(x), np.max(x)))
    for i in range(n_parties):
        c = party_colors[parties[i]]
        
        plt.plot(x, sm_results_cum[:, i], label=parties[i], color=c)
        y2 = sm_results_cum[:, i].astype('float')
        if i == 0:
            plt.fill_between(x, y1=np.zeros(len(sm_results_cum)), y2=y2, color=c)
        else:
            y1 = sm_results_cum[:, i-1].astype('float')
            plt.fill_between(x, y1=y1, y2=y2, color=party_colors[parties[i]])
    
    if legend:
        leg = plt.legend(prop={'size': legend_size}) 
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(15.0)
        
    fig.tight_layout()
    if save:
        plt.savefig(filename)
        plt.show();


# plot samples for cumulative election results
def plot_elections_cum_samples(samples_cum, x, parties, party_colors, save=False, filename='test.pdf', legend=False, n_parties=9):
    
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=20)
    legend_size=18
    lw_size=3
    
    sm_results_cum = samples_cum.mean(0)
    
    fig = plt.figure(figsize=(10, 5))
    plt.ylim((0.0, 1.00))
    plt.xlim((np.min(x), np.max(x)))
    
    #plot the means and fill between
    alpha_fill = 0.4
    alpha_samples = 0.6
    
    #"""
    for i in range(n_parties):
        c = party_colors[parties[i]]
        
        plt.plot(x, sm_results_cum[:, i], label=parties[i], color=c)
        y2 = sm_results_cum[:, i].astype('float')
        if i == 0:
            plt.fill_between(x, y1=np.zeros(len(sm_results_cum)), y2=y2, color=c, alpha=alpha_fill)
        else:
            y1 = sm_results_cum[:, i-1].astype('float')
            plt.fill_between(x, y1=y1, y2=y2, color=c, alpha=alpha_fill)
    #"""
    
    #plot all samples
    for s in samples_cum:
        
        for i in range(n_parties):
            c = party_colors[parties[i]]
            
            plt.plot(x, s[:, i], color=c, alpha=alpha_samples)
    
    
    if legend:
        leg = plt.legend(prop={'size': legend_size}) 
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(15.0)
        
    fig.tight_layout()
    if save:
        plt.savefig(filename)
        plt.show();
        
# plot cumulative results but color strength is dependent on uncertainty for that point in time
def plot_elections_cum_std(sm_results_cum, results_v, x, parties, party_colors, save=False, filename='test.pdf', legend=True, n_parties=9):

    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=20)
    legend_size=18
    lw_size=3
    
    plt.figure(figsize=(10, 5))
    plt.ylim((0.0, 1.00))
    lb = 0
    r_prev = np.zeros(x.max() - x.min()+1)
    
    v_min, v_max = results_v.min(), results_v.max()
    for i in range(n_parties):
        
        c = party_colors[parties[i]]
        r = sm_results_cum[:,i].astype('float')
        n = len(r)
        v = results_v[:, i].astype('float')

        y_min, y_max = r.min(), r.max()
        x_min, x_max = x.min(), x.max()
        
        ax = plt.gca()
        line, = ax.plot(x, r, color=c, label = parties[i])
            
        zorder = line.get_zorder()
        
        z = np.empty((x_max - x_min+1, 1, 4), dtype=float)
        c_norm = 1/v * v_min
        rgb = mcolors.colorConverter.to_rgb(c)
        z[:,:,:3] = rgb
        z[:,:,-1] = c_norm.reshape(-1,1)
        z = z.reshape(1, -1, 4)
        
        im = ax.imshow(z, aspect='auto', extent=[x_min, x_max, lb, y_max], origin='lower', zorder=zorder)
        
        xy = np.column_stack([x, r])
        xy_prev = np.column_stack([x, r_prev])[::-1]

        xy = np.vstack([[x_min, lb], xy, [x_max, lb], xy_prev])

        clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
        ax.add_patch(clip_path)
        im.set_clip_path(clip_path)
        
        lb = y_min
        r_prev = r
    
    if legend:
        leg = plt.legend(prop={'size': legend_size}) 
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(15.0)       

    plt.tight_layout()
    if save:
        plt.savefig(filename)
        plt.show();

# combine everything: cumulative samples with std as color strength
def plot_elections_cum_samples_std(samples_cum, results_v, x, parties, party_colors, save=False, filename='test.pdf', legend=False, n_parties=9):
    
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=20)
    legend_size=18
    lw_size=3
    
    sm_results_cum = samples_cum.mean(0)
    
    fig = plt.figure(figsize=(10, 5))
    plt.ylim((0.0, 1.00))
    plt.xlim((np.min(x), np.max(x)))
    
    #plot the means and fill between
    alpha_fill = 0.5
    alpha_samples = 0.6
    
    lb = 0
    r_prev = np.zeros(x.max() - x.min()+1)
    
    v_min, v_max = results_v.min(), results_v.max()
    for i in range(n_parties):
        
        c = party_colors[parties[i]]
        r = sm_results_cum[:,i].astype('float')
        n = len(r)
        v = results_v[:, i].astype('float')

        y_min, y_max = r.min(), r.max()
        x_min, x_max = x.min(), x.max()
        
        ax = plt.gca()
        line, = ax.plot(x, r, color=c, label = parties[i])
            
        zorder = line.get_zorder()
        
        z = np.empty((x_max - x_min+1, 1, 4), dtype=float)
        c_norm = 1/v * v_min
        rgb = mcolors.colorConverter.to_rgb(c)
        z[:,:,:3] = rgb
        z[:,:,-1] = c_norm.reshape(-1,1)
        z = z.reshape(1, -1, 4)
        
        im = ax.imshow(z, aspect='auto', extent=[x_min, x_max, lb, y_max], origin='lower', zorder=zorder, alpha=alpha_fill)
        
        xy = np.column_stack([x, r])
        xy_prev = np.column_stack([x, r_prev])[::-1]

        xy = np.vstack([[x_min, lb], xy, [x_max, lb], xy_prev])

        clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
        ax.add_patch(clip_path)
        im.set_clip_path(clip_path)
        
        lb = y_min
        r_prev = r
    
    #plot all samples
    for s in samples_cum:
        
        for i in range(n_parties):
            c = party_colors[parties[i]]
            
            plt.plot(x, s[:, i], label=parties[i], color=c, alpha=alpha_samples)
    
    
    if legend:
        leg = plt.legend(prop={'size': legend_size}) 
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(15.0)
        
    fig.tight_layout()
    if save:
        plt.savefig(filename)
        plt.show();
       




####### Elliptical slice sampling ########

#
# Elliptical slice sampling; taken from a github repo; 
# credits belong to original author
#
import math
def elliptical_slice(initial_theta,prior,lnpdf,pdf_params=(),
                     cur_lnpdf=None,angle_range=None):
    """
    NAME:
       elliptical_slice
    PURPOSE:
       Markov chain update for a distribution with a Gaussian "prior" factored out
    INPUT:
       initial_theta - initial vector
       prior - cholesky decomposition of the covariance matrix 
               (like what numpy.linalg.cholesky returns), 
               or a sample from the prior
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params= parameters to pass to the pdf
       cur_lnpdf= value of lnpdf at initial_theta (optional)
       angle_range= Default 0: explore whole ellipse with break point at
                    first rejection. Set in (0,2*pi] to explore a bracket of
                    the specified width centred uniformly at random.
    OUTPUT:
       new_theta, new_lnpdf
    HISTORY:
       Originally written in matlab by Iain Murray (http://homepages.inf.ed.ac.uk/imurray2/pub/10ess/elliptical_slice.m)
       2012-02-24 - Written - Bovy (IAS)
    """
    D= len(initial_theta)
    if cur_lnpdf is None:
        cur_lnpdf= lnpdf(initial_theta,*pdf_params)

    # Set up the ellipse and the slice threshold
    if len(prior.shape) == 1: #prior = prior sample
        nu= prior
        print("first condition is True")
    else: #prior = cholesky decomp
        if not prior.shape[0] == D or not prior.shape[1] == D:
            raise IOError("Prior must be given by a D-element sample or DxD chol(Sigma)")
        nu= np.dot(prior,np.random.normal(size=D))
    hh = math.log(np.random.uniform()) + cur_lnpdf

    # Set up a bracket of angles and pick a first proposal.
    # "phi = (theta'-theta)" is a change in angle.
    if angle_range is None or angle_range == 0.:
        # Bracket whole ellipse with both edges at first proposed point
        phi= np.random.uniform()*2.*math.pi
        phi_min= phi-2.*math.pi
        phi_max= phi
    else:
        # Randomly center bracket on current point
        phi_min= -angle_range*np.random.uniform()
        phi_max= phi_min + angle_range
        phi= np.random.uniform()*(phi_max-phi_min)+phi_min

        
    # Slice sampling loop
    while True:
        # Compute xx for proposed angle difference and check if it's on the slice
        xx_prop = (initial_theta*math.cos(phi)).reshape(-1) + (nu*math.sin(phi)).reshape(-1)
        cur_lnpdf = lnpdf(xx_prop,*pdf_params)
        if cur_lnpdf > hh:
            # New point is on slice, ** EXIT LOOP **
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise RuntimeError('BUG DETECTED: Shrunk to current position and still not acceptable.')
        # Propose new angle difference
        phi = np.random.uniform()*(phi_max - phi_min) + phi_min
    return (xx_prop,cur_lnpdf)