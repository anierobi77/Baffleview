import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def correlate_wells(well_a_gr, well_b_gr):
    """
    Uses Dynamic Time Warping to find the optimal path match 
    between high-resolution Gamma Ray logs of two different wells.
    """
    # Calculate the warping path and distance
    distance, path = fastdtw(well_a_gr, well_b_gr, dist=euclidean)
    
    # Path contains tuples of indices (idx_well_a, idx_well_b) that line up geologically
    return distance, path
