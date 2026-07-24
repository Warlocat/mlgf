import numpy as np
from fcdmft.ac.pade import PadeAC
from fcdmft.ac.two_pole import TwoPoleAC
from mlgf.lib.ml_helper import get_pade18

def get_sigma_R(sigmaI, mlf, gf_omega, eta, ac_method="pade", ac_idx=None, ao_rep=False, sigmaI_diff_ref_ao=None):
    nmo = sigmaI.shape[0]
    nocc = mlf['nocc']

    if sigmaI_diff_ref_ao is not None:
        ao_rep = True

    if ao_rep:
        c = mlf['mo_coeff_ref']
        ovlp = mlf['ovlp_ref']
        sc_s = ovlp @ c
        sigmaI_ao = np.zeros((ovlp.shape[0], ovlp.shape[1], sigmaI.shape[-1]), dtype=sigmaI.dtype)
        for iw in range(sigmaI.shape[-1]):
            sigmaI_ao[:, :, iw] = sc_s @ sigmaI[:, :, iw] @ sc_s.T
            if sigmaI_diff_ref_ao is not None:
                sigmaI_ao[:, :, iw] += sigmaI_diff_ref_ao[:, :, iw]
        sigmaI = sigmaI_ao

    omega_fit = getattr(mlf, 'omega_fit', None)
    if omega_fit is None:
        if "omega_fit" in mlf.keys():
            omega_fit = mlf['omega_fit']
        else:
            omega_fit, _ = get_pade18()
            omega_fit = mlf['ef'] + 1j*omega_fit

    if ac_idx is None:
        ac_idx = np.arange(sigmaI.shape[-1])

    # analytic continuation
    if ac_method == 'twopole':
        acobj = TwoPoleAC(list(range(nmo)), nocc)
    elif ac_method == 'pade':
        acobj = PadeAC()
        acobj.idx = ac_idx
    elif ac_method == 'pes':
        raise NotImplementedError
    else:
        raise ValueError('Unknown GW-AC type %s' % (str(ac_method)))

    acobj.ac_fit(sigmaI, omega_fit, axis=-1)
    return acobj.ac_eval(gf_omega + 1j*eta)
    