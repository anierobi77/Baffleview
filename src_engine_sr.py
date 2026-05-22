import numpy as np
import pandas as pd

def classify_bell_canyon_facies(gr, resistivity, porosity):
    """
    Applies Anierobi Ekweogwu's East Ford Field parameters to distinguish 
    the Ramsey sandstone reservoir units from the SH1 thin-bedded siltstone baffle.
    """
    # 1. SH1 Laminated Siltstone Barrier / Caprock Seal Detection
    # Highly radioactive, low resistivity, tight micro-porosity matrix
    if gr > 115 and resistivity < 12.0:
        return {
            "Facies": "SH1 Laminated Siltstone Baffle",
            "Permeability": "Tight (< 0.1 mD)",
            "Role": "Vertical Flow Barrier / CCS Seal"
        }
    
    # 2. Ramsey Sandstone Core Reservoir Facies (Ramsey 1 & 2)
    # Well-sorted, ultra-fine grained arkoses
    elif gr <= 95 and porosity >= 0.15:
        # Calcite cement distribution is the primary control on quality (Dutton & Ekweogwu)
        if resistivity > 25.0:  
            return {
                "Facies": "Ramsey Sandstone (Clean Channel)",
                "Permeability": "High Pay (40 - 50 mD)",
                "Role": "Primary Production Target"
            }
        else:
            return {
                "Facies": "Ramsey Sandstone (Calcite-Cemented / Levee)",
                "Permeability": "Marginal (3 - 5 mD)",
                "Role": "Baffled Flow Zone"
            }
            
    # 3. Ambient Siltstones / Overbank Splays
    else:
        return {
            "Facies": "Overbank Splay / Lobe Fringe",
            "Permeability": "Variable (0.5 - 2.0 mD)",
            "Role": "Reservoir Baffle"
        }
