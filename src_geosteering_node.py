import numpy as np

def calculate_steering_action(current_md, lwd_gr, lwd_res, b_config):
    """
    Evaluates real-time LWD data against active benchmarks to calculate 
    proactive trajectory adjustments for directional drillers.
    """
    # Detect if the bit is tracking into the tight SH1 flow baffle
    if lwd_gr > b_config['sh1_gr_cutoff'] and lwd_res < b_config['sh1_res_max']:
        return {
            "Status": "?? EMERGENCY BOUNDARY EXIT ALERT",
            "Action": "Drop inclination immediately. Proximity to upper SH1 Siltstone constraint exceeded.",
            "Color": "red"
        }
    
    # Verify bit location within high-pay clean Ramsey Sandstone
    elif lwd_gr <= 95.0:
        return {
            "Status": "?? ON TARGET: In-Zone",
            "Action": "Maintain current projection. Optmizing exposure inside high-permeability Ramsey channel.",
            "Color": "green"
        }
        
    else:
        return {
            "Status": "?? CAUTION: Approaching Boundary",
            "Action": "Monitor tool-face. Logs indicate transition into marginal overbank facies.",
            "Color": "warning"
        }
