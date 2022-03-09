import numpy as np
import scipy as sp

MAYA_PHONEME_NAMES = ['Ah', 'Aa', 'Eh', 'Ee', 'Ih', 'Oh', 'Uh', 'U', 'Eu', 
                      'Schwa', 'R', 'S', 'ShChZh', 'Th',
                      'JY', 'LNTD', 'GK', 'MBP', 'FV', 'W']

PHONEME_THRESHOLD = np.array([0.12, 0.23, 0.18, 0.02, 10, 0.19, 0.18, 0.05, 10, 0.16,
                              0.18, 0.29, 0.29, 0.27, 10, 10, 10, 0.004, 0.29, 0.16])

# PHONEME_THRESHOLD = np.array([0.35, 0.23, 0.18, 0.17, 10, 0.19, 0.18, 0.19, 10, 0.16,
#                               0.18, 0.29, 0.29, 0.27, 10, 10, 10, 0.004, 0.29, 0.16]) # perfect

def smooth(x, window_len, window='hanning'):
    if window_len < 3:
            return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    
    y = np.convolve(w / w.sum(), s, mode='valid')
    
    return y


def postprocess_model_outputs(reg_output, cls_output):
    """ Postproces raw outputs of the VisemeNet.
    Args:
        reg_output: Shape as (Num Frames, 22)
        cls_output: Shape as (Num Frames, 22)
    Return:
        viseme_outputs: JALI based Lip blendshapes coefficients
    """
    assert reg_output.shape == cls_output.shape
    
    num_frames, num_maya_params = reg_output.shape
    num_translate = 2
    for i in range(num_translate, num_maya_params):
        # Cls. output
        cls_output[2:-3, i] = sp.signal.medfilt(cls_output[2:-3, i], kernel_size=[9])
        cls_output[:, i] = smooth(cls_output[:, i], window_len=9)[4:-4]
        # Reg. output
        reg_output[:, i] = sp.signal.medfilt(reg_output[:, i], kernel_size=[9])
        reg_output[:, i] = smooth(reg_output[:, i], window_len=9)[4:-4]

    viseme_outputs = np.zeros_like(cls_output)
    viseme_outputs[:, 0] = smooth(cls_output[:, 0], window_len=15)[7:-7]
    viseme_outputs[:, 1] = smooth(cls_output[:, 1], window_len=15)[7:-7]

    for i in range(num_translate, num_maya_params):
        tmp = cls_output[:, i] * reg_output[:, i]
        l_idx = tmp > PHONEME_THRESHOLD[i-2]
        viseme_outputs[l_idx, i] = reg_output[l_idx, i]

        viseme_outputs[:, i] = smooth(viseme_outputs[:, i], window_len=15)[7:-7]

        r = 0
        while r < viseme_outputs.shape[0]:
            if viseme_outputs[r, i] > 0.1:
                active_begin = r
                for r2 in range(r, viseme_outputs.shape[0]):
                    if viseme_outputs[r2, i] < 0.1 or r2 == viseme_outputs.shape[0] - 1:
                        active_end = r2
                        r = r2
                        break
                
                if (active_begin == active_end):
                    break
                max_reg = np.max(reg_output[active_begin:active_end, i])
                max_pred = np.max(viseme_outputs[active_begin:active_end, i])
                rate = max_reg / max_pred
                viseme_outputs[active_begin:active_end, i] = viseme_outputs[active_begin:active_end, i] * rate
            r += 1
        viseme_outputs[:, i] = smooth(viseme_outputs[:, i], 15)[7:-7]

        r = 0
        while r < viseme_outputs.shape[0]:
            if viseme_outputs[r, i] > 0.1:
                active_begin = r
                for r2 in range(r, viseme_outputs.shape[0]):
                    if viseme_outputs[r2, i] < 0.1 or r2 == viseme_outputs.shape[0] - 1:
                        active_end = r2
                        r = r2
                        break
                
                max_reg = np.max(reg_output[active_begin:active_end, i])
                if(i==19 or i==20 or i==21):
                    if(max_reg>0.7):
                        max_reg = 1
                max_pred = np.max(viseme_outputs[active_begin:active_end, i])
                rate = max_reg / max_pred
                viseme_outputs[active_begin:active_end, i] = viseme_outputs[active_begin:active_end, i] * rate
            r += 1
    
    return viseme_outputs
