import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg 
mpl.rcParams['pdf.fonttype'] = 42

import numpy as np
from StringIO import StringIO
import pandas as pd
import os
from IPython.display import display
from collections import defaultdict
import csv

from IPython.display import display
from collections import defaultdict
from collections import Counter
from itertools import combinations
from scipy import stats
import os
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import numpy as np



def density_plot(x, y, nbins=42, log=False):
    mask = (~np.isnan(x)) & (~np.isnan(y))
#     axTemperature = plt.axes(rect_temperature) # temperature plot
#     axHistx = plt.axes(rect_histx) # x histogram
#     axHisty = plt.axes(rect_histy) # y histogram
    x = x[mask]
    y = y[mask]
    H, xedges, yedges = np.histogram2d(x,y,bins=nbins)
    ix = np.searchsorted(xedges, x)
    ix[ix == nbins] = nbins - 1
    iy = np.searchsorted(yedges, y)
    iy[iy == nbins] = nbins - 1
    v = H[ix, iy]
    i = v.argsort()
    cc = v[i]
    if log:
        cc = np.log(cc + 1)
    plt.scatter(x[i], y[i], c=cc, s=3, edgecolor='')


def plot_better(width=10, height=5, grid='xy', legend=False, visible_axes=True):
    plt.figure(figsize=(width, height))
    ax = plt.subplot(111)
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    if visible_axes:
        ax.spines["bottom"].set_visible(True) 
        ax.spines["bottom"].set_color('gray') 
        ax.spines["left"].set_visible(True)   
        ax.spines["left"].set_color('gray')
    else:
        ax.spines["bottom"].set_visible(False)  
        ax.spines["left"].set_visible(False) 
    
    if grid == 'xy':
        ax.xaxis.grid(True) 
        ax.yaxis.grid(True) 
    if grid == 'x':
        ax.xaxis.grid(True) 
    if grid == 'y':
        ax.yaxis.grid(True) 
    if legend:
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1), frameon=False)

    return ax
    


def improve_plot(ax, grid='xy', legend=False, visible_axes=True):
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    if visible_axes:
        ax.spines["bottom"].set_visible(True) 
        ax.spines["bottom"].set_color('gray') 
        ax.spines["left"].set_visible(True)   
        ax.spines["left"].set_color('gray')
    else:
        ax.spines["bottom"].set_visible(False)  
        ax.spines["left"].set_visible(False) 
    
    if grid == 'xy':
        ax.xaxis.grid(True) 
        ax.yaxis.grid(True) 
    if grid == 'x':
        ax.xaxis.grid(True) 
    if grid == 'y':
        ax.yaxis.grid(True) 
    if legend:
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1), frameon=False)

    return ax


#returns index of closest value in array
def closest_wavelength_index(nm_array, wavelength):
    return np.argmin([abs(value-wavelength) for value in nm_array])

#finds maximum value between from_nm and to_nm
def max_between(sample_as_Spectrum_instance, from_nm=350, to_nm=700):
    sample = sample_as_Spectrum_instance
    f=closest_wavelength_index(sample.nm,from_nm)
    t=closest_wavelength_index(sample.nm,to_nm)    
    return(max(sample.values[f:t]))

# rounds an integer to hundrets
def round_int(x):
    return 100 * ((int(x)+50) / 100)

# Lambert-Beer law
def get_concentration(OD, EC, l=1):
    return 1.*OD/(EC*l)

# Lambert-Beer law
def get_EC(OD, C, l=1):
    return 1.*OD/C

# sum of all values in the spectrum
def get_emission_sum(Spectrum_instance_for_emission):
    return float(sum(Spectrum_instance_for_emission.values))

# get value at wavelength closest to the query wavelength
def get_value_at_nm(Spectrum_instance, wavelength):
    return float(Spectrum_instance.values[closest_wavelength_index(Spectrum_instance.nm, wavelength)])


# finds minimum value between from_nm and to_nm
def min_between(sample_as_Spectrum_instance, from_nm, to_nm):
    sample = sample_as_Spectrum_instance
    f=closest_wavelength_index(sample.nm,from_nm)
    t=closest_wavelength_index(sample.nm,to_nm)    
    return(min(sample.values[f:t]))


#creates directory if it does not exist
def ensure_dir(f):
    import os
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

#intended to work with single-sample .csv files produced by Spectrum.save_as_cv() method.
def read_my_own_csv(filepath, delimiter=','):
    from StringIO import StringIO
    f = open(filepath,'r')
    lines = f.readlines()
    blank = lines.index('\n')
    data = genfromtxt(StringIO('\n'.join([line[:-1] for line in lines[1:blank]])),delimiter=delimiter)
    sample_as_object = Spectrum(lines[0][:-1], a, lines[blank+1:])
    #capturing sample state
    for line in lines[blank+1:]:
        if 'Sample state: ' in line:
            sample_as_object.state = line[13:]
            break
    return sample_as_object

# substracts second spectrum from the first one, returns new Spectrum instance
def diff_spectrum(First_Spectrum_instance, Second_Spectrum_instance):
    #checking if spectra are determined for the same wavelength range
    if max([a-b for a,b in zip(First_Spectrum_instance.nm,Second_Spectrum_instance.nm)]) > 1 or len(First_Spectrum_instance.nm) != len(Second_Spectrum_instance.nm):
        print('Shit happened at differential spectrum calculation: ' +
              'wavelength range is different for provided spectra.')
    diff = [a-b for a,b in zip(First_Spectrum_instance.values, Second_Spectrum_instance.values)]
    data_array = array(zip(ko_abs[0].nm,diff))
    name = 'Diff. spectrum for %s and %s' %(First_Spectrum_instance.name, Second_Spectrum_instance.name)
    info = First_Spectrum_instance.info
    info.append('THIS LINE SEPARATES SAMPLES INFORMATION '
                +'USED FOR DIFFERENTIAL SPECTRUM CALCULATION\n') 
    info.extend(Second_Spectrum_instance.info)
    return Spectrum(name, data_array, info)
        
    
class Spectrum:
    
    def __init__(self, sample_name, data_np_array, info_as_str_list=[], from_Pandas_DataFrame=False):
        if from_Pandas_DataFrame:
            data_np_array = np.array(data_np_array)
        self.name = sample_name
        self.nm = data_np_array[:,0][~np.isnan(data_np_array[:,0])]
        self.values = data_np_array[:,1][~np.isnan(data_np_array[:,0])]
        if self.nm[0] > self.nm[-1]:
            self.nm = self.nm[::-1]
            self.values = self.values[::-1]
        self.info = info_as_str_list        
        self.is_baseline='baseline' in self.name.lower()
        self.concentration = None
        self.spectrum_type = None
        for line in self.info:
            if 'scan mode ' in line.lower():
                if 'emission' in line.lower():
                    self.spectrum_type = 'Emission'
                    break
                if 'excitation' in line.lower():
                    self.spectrum_type = 'Excitation'
                    break
            else:
                self.spectrum_type = 'Absorbance'                
        self.state = ''


    def __str__(self):
        return 'Sample name: %s\nSpectrum type: %s | Measured from %s to %s nm\n' %(self.name, self.spectrum_type, self.nm[0], self.nm[-1])

    def copy(self, new_name=None, new_info=None, new_state=None):
        from copy import deepcopy
        c = deepcopy(self)
        if new_name:
            c.name = new_name
        if new_info:
            c.info = new_info
        if new_state:
            c.state = new_state
        return c
    
    def clear_copy(self, new_name):
        from copy import deepcopy
        c = deepcopy(self)
        c.name = new_name
        c.info = []
        c.state = ''
        return c
    
    # works only if sample.name does not contain 'baseline'
    def baseline_correction(self, baseline_as_Spectrum_instance):
        bs = baseline_as_Spectrum_instance
        #checking whether baseline is determined for the same wavelength range
        if min(bs.nm) != min(self.nm) or max(bs.nm) != max(self.nm):
            print('Baseline is measured for a different wavelength region (from '+
                  str(min(bs.nm))+'nm to '+str(max(bs.nm))+'nm) than sample (from '+
                  str(min(self.nm))+'nm to '+str(max(self.nm))+
                  'nm).\nBaseline correction was skipped.')
            return None
        if 'baseline' not in self.name.lower():
            self.values = [self.values[i] - bs.values[i] for i in range(0,len(self.nm))]
            self.state += ' baseline_substracted'
    
    def make_zero_at(self, wavelength):
        value_to_hold = self.values[closest_wavelength_index(self.nm,wavelength)]
        self.values = [value - value_to_hold for value in self.values]
        self.state += ' zeroed_at_'+str(wavelength)

    def make_zero_at_min(self, from_nm=500, to_nm=650):
        ind_from = closest_wavelength_index(self.nm,from_nm)
        ind_to = closest_wavelength_index(self.nm,to_nm)
        min_wavelength = np.argmin(self.values[ind_from:ind_to+1])
        value_to_hold = self.values[min_wavelength+ind_from]
        self.values = [value - value_to_hold for value in self.values]
        self.state += ' zeroed_at_'+str(self.nm[min_wavelength+ind_from])

        
    def normalize(self, from_nm=400, to_nm=650, set_final_max_value=1):
        if not from_nm:
            from_nm=min(self.nm)
        if not to_nm:
            to_nm=max(self.nm)
        max_value = max_between(self, from_nm, to_nm)
        self.values = [np.divide(value,max_value)*set_final_max_value for value in self.values]
        self.state += ' normalized'


    def normalize_to_sum(self, from_nm=400, to_nm=650, set_final_max_value=1):
        if not from_nm:
            from_nm=min(self.nm)
        if not to_nm:
            to_nm=max(self.nm)
        max_value = np.sum(self.values[closest_wavelength_index(self.nm,from_nm): 
            closest_wavelength_index(self.nm,to_nm)])
        self.values = [np.divide(value,max_value)*set_final_max_value for value in self.values]
        self.state += ' normalized_to_sum_from_%s_to_%s_nm' %(from_nm, to_nm)

    def normalize_to_be_equal_at_nm(self, spectrum_ref, nm):
        ref_value_at_nm = get_value_at_nm(spectrum_ref, nm)
        max_value = get_value_at_nm(self, self.max_at(nm))
        current_value_at_nm = get_value_at_nm(self, nm)
        final_max_value = ref_value_at_nm * (1. * max_value / current_value_at_nm)
        self.normalize(set_final_max_value=final_max_value)
        self.state += ' normalized'


    def normalize_at_nm(self, nm, set_final_max_value=1):
        value_at_nm = get_value_at_nm(self, nm)
        self.values = [np.divide(value,value_at_nm)*set_final_max_value for value in self.values]
        self.state += ' normalized'




    def save_as_csv(self, filepath, include_info=True, delimiter=',', textulating=True):
        ensure_dir(filepath)
        f = open(filepath,'w')
        f.write('wavelength' + delimiter + self.name + '\n')
        for i in range(0,len(self.nm)):
            f.write(str(self.nm[i]) + delimiter + str(self.values[i]) + '\n')
        f.write('\n')
        if include_info:
            f.write(self.name + '\n')
            f.write('Sample state: ' + self.state + '\n')
            for line in self.info:
                f.write(line)
        if textulating:
            print('\nSample "%s" \nin the state "%s " \nwas saved to %s.\n'%(self.name,self.state,filepath))
        
    def get_acquisition_time(self):
        return 'khui'
    
    def correct_shift_at_349nm(self):
        shift = self.values[closest_wavelength_index(self.nm,349)] - self.values[closest_wavelength_index(self.nm,348)]
        min_ = min(np.argmin(self.nm), closest_wavelength_index(self.nm,348))
        max_ = min(np.argmin(self.nm), closest_wavelength_index(self.nm,348))
        for i in range(min_,max_):
            self.values[i] -= shift            
        self.state += ' 349nm_shift_corrected'     
        
    # method by Sasha - accepts one-dimension arrays as input
    def smooth(self,window_len=5,window='hanning'):
        x = np.array(self.values)
        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."    
        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."    
        if window_len<3:
            return x    
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"    
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')    
        y=np.convolve(w/w.sum(),s,mode='same')
        self.values = y[window_len:-window_len+1]
        self.state += ' smoothed '
        
        
    def first_local_min(self, from_nm=350, to_nm=400, smoothing_window_len=10):
        from scipy.signal import argrelextrema
        temp = self.copy()
        ind_from = closest_wavelength_index(temp.nm,from_nm)
        ind_to = closest_wavelength_index(temp.nm,to_nm)
        temp.smooth(window_len=smoothing_window_len)
        return ind_from + np.argmin(self.values[ind_from:ind_to])
        
    
    def clarify(self, left_nm_range=(350,400), right_nm_range=(600,700), smoothing_window_len=10, zero_at=650):
        left_min_index = self.first_local_min(from_nm=left_nm_range[0], to_nm=left_nm_range[1], smoothing_window_len=smoothing_window_len)
        right_min_index = self.first_local_min(from_nm=right_nm_range[0], to_nm=right_nm_range[1], smoothing_window_len=smoothing_window_len)
        increment = min(self.values)
        logvalues = [np.log(value-increment+0.001) for value in self.values]
        # y = kx + b
        # k = (y1-y2)/(x1-x2)
        # b = y1 - kx1
        k = (logvalues[left_min_index]-logvalues[right_min_index])/(self.nm[left_min_index]-self.nm[right_min_index])
        b = logvalues[left_min_index] - k * self.nm[left_min_index]
        #print('k=%s, b=%s' %(k,b))
        linevalues = [np.exp(k*nm+b)-increment for nm in self.nm]
        self.values = [value - linevalue for value, linevalue in zip(self.values,linevalues)] 
        self.make_zero_at(zero_at)
        self.state += ' corrected for scattering '
        
    def max_at(self, from_nm=350, to_nm=700, smoothing_window_len=10):
        temp = self.copy()
        ind_from = closest_wavelength_index(temp.nm,from_nm)
        ind_to = closest_wavelength_index(temp.nm,to_nm)
        temp.smooth(window_len=smoothing_window_len)
        return self.nm[ind_from + np.argmax(self.values[ind_from:ind_to])]

def get_list_of_Spectrum_instances(filepath, textulating=True, plotting=False, norm=False, line_end_char=None):
    f = open(filepath, 'r')
    full_list_of_lines = f.readlines()
    first_line_with_data = 2

    #sniffing delimiter and dealing with comma-separated floatings
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(full_list_of_lines[0]).delimiter

    if not line_end_char:
        line_end_char = full_list_of_lines[0].split(delimiter)[-1]

    try:
        index_of_first_blank_line = full_list_of_lines.index(line_end_char)
    except ValueError:
        print('Parser did not find a blank line in the file that separates data from sample information. Check line_end_char')
        index_of_first_blank_line = len(full_list_of_lines)


    data_only_lines = full_list_of_lines[first_line_with_data:index_of_first_blank_line]
    if delimiter != ',':
        data_only_lines = [line.replace(',','.') for line in full_list_of_lines[first_line_with_data:index_of_first_blank_line]]

    #collecting data
    data = np.genfromtxt(StringIO('\n'.join([line[:-1*len(line_end_char)].rstrip(delimiter) for line in data_only_lines])), 
                     delimiter=delimiter,missing_values='',filling_values=None)
    
    #collecting information about each sample from the bottom of the file
    sample_infos = []
    blank_line = index_of_first_blank_line+1
    for i in range(index_of_first_blank_line+1,len(full_list_of_lines)):
        if full_list_of_lines[i] == line_end_char:
            sample_infos.append(full_list_of_lines[blank_line:i])
            sample_infos.append([])
            blank_line = i+1


    #dealing with sample names
    raw_names = (full_list_of_lines[0].rstrip(delimiter+line_end_char)+',').split(delimiter)
    # print full_list_of_lines[0]
    # print full_list_of_lines[0].rstrip(delimiter+line_end_char)+','
    # print raw_names
    # print len(raw_names)
    # print 
    # print data[0]
    # print len(data[0])



    #creating Spectrum instances
    spectra = []
    for i in range(0,len(data[0]),2):   
        spectra.append(Spectrum(raw_names[i], data[:,i:i+2], sample_infos[i]))

    if textulating:
        print('\n')
        for i in range(0,len(spectra)):
            print('['+str(i)+'] '+ spectra[i].name)

    if norm:
        for sample in spectra:
            sample.normalize(from_nm=350, to_nm=700)

    if plotting:
        plot_better()
        leg = []
        for sample in spectra:
            plt.plot(sample.nm,sample.values)
            leg.append(sample.name)
        figname = filepath + '\n'
        plt.title(figname)
        plt.xlabel('Wavelength, nm')
        plt.legend(leg, loc=2, bbox_to_anchor=(1.05, 1), frameon=False) 
    
    return spectra

#supposed to work with Abs spectra only
def do_it_as_I_like(filepath, baseline_correction=True, zero_at=650, correct_349nm=True, norm=False, textulating=False, 
                    plotting=True, save_plots_to=None, baseline_instance=None, baseline_not_returned=True):
    
    samples = get_list_of_Spectrum_instances(filepath, textulating=textulating)
    baselines_list = [sample for sample in samples if 'baseline' in sample.name.lower()]
    number_of_baselines = len(baselines_list)
    
    if baseline_correction:        
        if number_of_baselines == 0 and not baseline_instance:
            print('\nDid not find baseline, sorry. Please provide correct baseline in baseline_instance optional parameter.')
        if number_of_baselines == 1:
            if baseline_instance:
                for sample in samples:
                    sample.baseline_correction(baseline_instance)
                if baseline_not_returned:
                    samples.remove(baselines_list[0])
                if textulating:
                    print('\nProvided baseline ("'+baseline_instance.name
                          +'") was used for baseline correction.')

            else:
                for sample in samples:
                    sample.baseline_correction(baselines_list[0])
                if baseline_not_returned:
                    samples.remove(baselines_list[0])
                if textulating:
                    print('\nSample "'+baselines_list[0].name
                          +'" was used for baseline correction.')

                
        if number_of_baselines > 1:
            print('\nMore than one baseline found. Please provide correct baseline in baseline_instance optional parameter.')
    
    
    for sample in samples:
        if zero_at:
            sample.make_zero_at(zero_at)
        if correct_349nm:
            sample.correct_shift_at_349nm()
        if norm:
            sample.normalize(from_nm=350,to_nm=650)
        
    if plotting:
        plot_better()
        maxlist=[]
        minlist=[]
        leg = []        
        for sample in samples:
            plt.plot(sample.nm,sample.values)
            maxlist.append(max_between(sample,350, max(sample.nm)))
            minlist.append(min_between(sample,350, max(sample.nm)))
            leg.append(sample.name)
        plt.xlim(300,650)
        plt.ylim(min(minlist)-0.01,1.1*max(maxlist))
        figname = filepath + '\n'
        plt.title(figname)
        plt.xlabel('Wavelength, nm')
        plt.ylabel('Optical density')
        plt.legend(leg, loc=2, bbox_to_anchor=(1.05, 1), frameon=False) 
        if save_plots_to:
            plt.savefig(save_plots_to + '%s.png' %(filepath),dpi=1200)
            #savefig(path_for_saving + '%s.svg' %(figname),bbox_extra_artists=[lgd.legendPatch])
            plt.savefig(save_plots_to + '%s.eps' %(filepath))
    
    return samples


def find_max_in_spectra(sample_list, name):
    for sample in sample_list:
        if name in sample.name:
            return int(sample.max_at())
    else:
        return None

