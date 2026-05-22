import numpy as np

def generate_synthetic_seismogram(rhob, dt_sonic, frequency=30, dt=0.002):
    """
    Computes Acoustic Impedance, Reflection Coefficients, and convolves
    with a Ricker wavelet to map log data to seismic time-domain.
    """
    # 1. Acoustic Impedance (AI) = Density * Velocity (Velocity = 1,000,000 / Sonic Travel Time)
    velocity = 1000000 / dt_sonic
    acoustic_impedance = rhob * velocity
    
    # 2. Reflection Coefficients (RC)
    rc = np.zeros(len(acoustic_impedance))
    rc[1:] = (acoustic_impedance[1:] - acoustic_impedance[:-1]) / (acoustic_impedance[1:] + acoustic_impedance[:-1])
    
    # 3. Create Ricker Wavelet
    t = np.arange(-0.06, 0.06, dt)
    wavelet = (1 - 2 * (np.pi * frequency * t)**2) * np.exp(-(np.pi * frequency * t)**2)
    
    # 4. Convolve to create Synthetic Seismogram
    synthetic = np.convolve(rc, wavelet, mode='same')
    return acoustic_impedance, synthetic
