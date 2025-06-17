import numpy as np
import uproot as uproot
from cycler import cycler

# DAC conversion factor in [mV/bit]
dacConv = 2000 / 2**14

# sampling time in [ns]
samplingTime = 2

# absolute threshold for the time-over-threshold analysis in [mV]
tot_absTh = 120
# relative threshold for the time-over-threshold analysis
tot_relTh = 0.15

pres = {'axes.titlecolor':'white',
        'axes.edgecolor':'white', 
        'xtick.color':'white', 
        'ytick.color':'white', 
        'figure.facecolor':'none', 
        'axes.labelcolor':'white',
        'axes.facecolor':'none', 
        'legend.facecolor':'white',
        'legend.framealpha':0.2,
        'legend.labelcolor':'white',
        'figure.dpi':200,
        'axes.prop_cycle':cycler(color=['deepskyblue', 'orange', 'yellowgreen', 'tomato', 'orchid', 'w']),
        }

post = {'figure.dpi':300,
        'axes.prop_cycle':cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#000000']),
       }

regu = {'figure.dpi':100,
        'axes.prop_cycle':cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#000000']),
       }

def ratio (A, B):
    '''
    ratio of two numbers/arrays and its error
    '''
    ratio = A[0] / B[0]
    ratioErr = np.sqrt( (A[1]/B[0])**2 + (A[0]*B[1]/B[0]**2)**2 )
    
    return np.array([ratio, ratioErr])


def meanWithError (data:np.ndarray[float]) -> np.ndarray[float]:
    '''
    calculate the mean of a data set and the uncertainty of the mean
    '''
    return np.array([data.mean(), data.std()/np.sqrt(len(data))])


def linFct (p:np.ndarray[float], x:np.ndarray[float]) -> np.ndarray[float]:
    '''
    linear polynomial function
    
    p[1]*x + p[0]
    '''
    return p[1]*x + p[0]

    
def quadFct (p:np.ndarray[float], x:np.ndarray[float]) -> np.ndarray[float]:
    '''
    quadratic polynomial function
    
    p[2]*x**2 + p[1]*x + p[0]
    '''
    return p[2]*x**2 + p[1]*x + p[0]


def chargeFct (p:np.ndarray[float], x:np.ndarray[float]) -> np.ndarray[float]:
    '''
    quadratic polynomial with additional scaling constant

    p[3] * ( p[2]*x**2 + p[1]*x + p[0] )
    '''
    return p[3] * ( p[2]*x**2 + p[1]*x + p[0] )


def getDOOCS (run, type='charge'):
    '''
    get the data from the beamline diagnostics

    run:
        run number to be analyzed

    type:
        charge: charge from the toroid beam charge transformer in [pC]
        posX:   horizontal position from the beam position monitor in [mm]
        posY:   vertical position from the beam position monitor in [mm]
    '''

    filename = f'processed/run_{run:05d}.npz'

    # create the empty data array
    data = np.load(filename)

    # select the type of data 
    if type=='charge':
        return data['chargeToroid']
    elif type=='posX':
        return data['posX']
    elif type=='posY':
        return data['posY']
    else:
        raise ValueError(f'type "{type}" not implemented')

def getCALO (run:int, channel:int=0, type:str='int') -> np.ndarray[float]:
    '''
    get the digitized data from the PMTs
    
    run: 
        run number to be analyzed
    
    channel: 
        channel number to be analyzed
        0-7:   PMT channels
        8,9:   tiles before dump
        10,11: tiles after dump

    type:
        type of the data
        amp:     maximum amplitude of the waveform in [mV]
        ampPos:  position of the amplitude maximum in [ns]
        int:     integral of the waveform in [mV µs]
        tot:     time over threshold in [ns]
        tot_rel: relative time over threshold in [ns]
        adc:     raw waveform in [adc counts]
        wave:    baseline subtracted waveform in [mV]
    '''

    filename = f'data/measurements/run_{run:05d}.root'
    
    with uproot.open(filename) as fd:
            
            # get the baseline values of the channel in [counts]
            bsl = fd['data;1'][f'baseline{channel}'].array().to_numpy()
            
            # get the waveforms of the channel in [counts]
            adc = fd['data;1'][f'data{channel}'].array().to_numpy()
                
            # subtract the baseline and convert to signal in [mV]
            sig_mV = abs(dacConv * (bsl[:,np.newaxis] - adc))
        
            # create the time array in [ns]
            time_ns = samplingTime * np.arange(sig_mV.shape[-1])
        
            # get the amplitude in [mV]
            amplitude = abs(dacConv*fd['data;1'][f'amplitude{channel}'].array().to_numpy())

            # get the position of maximum amplitude in [ns]
            amplitudePosition = samplingTime * fd['data;1'][f'amplitude_position{channel}'].array().to_numpy()
        
            # get the integral in [mV µs]
            integral = abs(dacConv*fd['data;1'][f'integral{channel}'].array().to_numpy()) * samplingTime/1e3

            # calculate the time over threshold in [ns]
            tot = samplingTime * np.sum(sig_mV>tot_absTh, axis=1)
            tot_rel = samplingTime * np.sum(sig_mV>tot_relTh*amplitude[:,np.newaxis], axis=1)

    # create the mask to remove all empty events
    mask = (sig_mV.std(axis=1)>50)

    if type=='amp':
        return amplitude[mask]
    elif type=='ampPos':
        return amplitudePosition[mask]
    elif type=='int':
        return integral[mask]
    elif type=='tot':
        return tot[mask]
    elif type=='tot_rel':
        return tot_rel[mask]
    elif type=='adc':
        return adc[mask]
    elif type=='wave':
        return sig_mV[mask]
    else:
        raise ValueError(f'type "{type}" not implemented')