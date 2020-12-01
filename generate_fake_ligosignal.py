import pycbc
import pycbc.noise
import pycbc.psd
import pycbc.filter
import pycbc.waveform
import pycbc.detector
from pycbc import types, fft, waveform
import numpy as np    

def generate_noise(params):
    
    delta_f = 1./params['tlen']
    flen = int(params['fsamp_psd'] / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, params['flow'])
    
    delta_t = 1.0 / params['fsamp_noise']
    tsamples = int(params['tlen'] / delta_t)

    n = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
    ntilde = n.to_frequencyseries()
    
    return psd, n, ntilde

def generate_tdsignal(params):
    
    hp, hc = pycbc.waveform.get_td_waveform(approximant = params['approximant'],mass1=params['mass1'], mass2=params['mass2'], spin1z=params['spin1z'], spin2z=params['spin2z'], 
                                                distance=params['distance'], coa_phase=params['coa_phase'], f_lower=params['flow'], delta_t=1/params['fsamp_noise'], amplitude_order=params['amplitude_order'])
        
    return hp, hc

def generate_fdsignal(params):
    
    hp, hc = pycbc.waveform.get_fd_waveform(approximant = params['approximant'],mass1=params['mass1'], mass2=params['mass2'], spin1z=params['spin1z'], spin2z=params['spin2z'], 
                                                distance=params['distance'], coa_phase=params['coa_phase'], f_lower=params['flow'], delta_f=1./params['tlen'], amplitude_order=params['amplitude_order'])
        
    return hp, hc


def invfft(delta_t, hp):
    
    tlen = int(1./delta_t / hp.delta_f)
    hp.resize(tlen/2 + 1)
    hp_td = types.TimeSeries(types.zeros(tlen), delta_t=hp.delta_t)
    fft.ifft(hp, hp_td)

    return hp_td

def generate_filter(ifo, ntilde, params):
    
    detector = pycbc.detector.Detector(ifo)
    hp, hc = pycbc.waveform.get_fd_waveform(approximant = params['approximant'],mass1=params['mass1'], mass2=params['mass2'], spin1z=params['spin1z'], spin2z=params['spin2z'], 
                                                distance=params['distance'], coa_phase=params['coa_phase'], f_lower=params['flow'], delta_f=ntilde.delta_f, amplitude_order=params['amplitude_order'])
    fp,fc = detector.antenna_pattern(params['ra'], params['dec'], params['psi'], params['tgps'])  #Antenna pattern
    A = np.exp(1j*params['phi0'])/params['distance'] * ( fp*(1+np.cos(params['iota'])**2)/2 - 1j*fc*np.cos(params['iota']) )
    
    hp.resize(len(ntilde))
    return hp, A

def calculate_logL(h0, psd, data, A, flow):
    
    #(h0|h0)
    h0_h0 = pycbc.filter.sigmasq(h0, psd=psd, low_frequency_cutoff=flow)
    h_h = A*A.conj() * h0_h0
    #(d|d)
    d_d = pycbc.filter.sigmasq(data, psd=psd, low_frequency_cutoff=flow) 

    #complex SNR time series, z(\hat{h0},d)
    z =  pycbc.filter.matched_filter(h0, data, psd=psd,
                                      low_frequency_cutoff=flow) 
    d_h = np.real(np.sqrt(h0_h0) * A * z.data)
    ln_L = types.TimeSeries(d_h - 0.5 * (d_d + h_h), delta_t=z.delta_t)
    snr_mf = max(abs(z))
    
    return h_h, d_d, d_h, ln_L, z, snr_mf 

def generate_fddata(ifo, hp, hc, ntilde, params):
    
    detector = pycbc.detector.Detector(ifo)
    fp,fc = detector.antenna_pattern(params['ra'], params['dec'], params['psi'], params['tgps'])  #Antenna pattern
    freq = hp.get_sample_frequencies().data      # Frequency vector
    shift  = types.FrequencySeries(np.exp(-2j*np.pi*freq*params['tc']),delta_f=1./params['tlen'])   #Time shift
    signal = ( fp*(1 + np.cos(params['iota'])**2)/2 * hp + fc*np.cos(params['iota']) * hc )* shift   # Adding Time shift in the signal
    signal_new = ( fp*(1 + np.cos(params['iota'])**2)/2 * hp + fc*np.cos(params['iota']) * hc )
    
    signal.resize(len(ntilde))
    signal_new.resize(len(ntilde))
    data = ntilde + signal
    data_new = ntilde + signal_new
    
    return fp, fc, shift, signal, signal_new, data, data_new

def generate_fddata_inj(ifo, f, hp, hc, ntilde, params):
    
    detector = pycbc.detector.Detector(ifo)
    fp,fc = detector.antenna_pattern(params['ra'], params['dec'], params['psi'], params['tgps'])  #Antenna pattern     # Frequency vector
    shift  = types.FrequencySeries(np.exp(-2j*np.pi*f*params['tc']),delta_f=1./params['tlen'])   #Time shift
    signal = ( fp*hp + fc*hc)* shift   # Adding Time shift in the signal
   
    signal.resize(len(ntilde))
    data = ntilde + signal
    
    return fp, fc, shift, signal, data