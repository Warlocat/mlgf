from ._data import Data
import pyscf.pbc.dft, pyscf.lib
import numpy as np

class Data_gwpt(Data):
    """Data class for storing GWPT data."""

    def get_mf(self, scf_constructor=pyscf.pbc.dft.RKS):
        """Creates a pyscf mean-field object from the saved data

        Args:
            scf_constructor (function, optional): Constructor for mean-field object. Defaults to pyscf.dft.RKS.

        Returns:
            mf: mean-field object
        """
        if('mol' in self.data):
            mf = scf_constructor(self.mol, xc='hf')
            if 'e_tot' in self.data:
                mf.__dict__.update({'e_tot': self.e_tot, 'mo_energy_ref': self.mo_energy,
                                    'mo_occ': self.mo_occ, 'mo_coeff_ref': self.mo_coeff})
            else:
                mf.__dict__.update({'mo_energy_ref': self.mo_energy, 'mo_occ': self.mo_occ,
                                    'mo_coeff_ref': self.mo_coeff})
                
            return mf
        else:
            raise ValueError('You need to set the mol attribute before calling get_mf')
    
    @staticmethod
    def load_chk(f, purge_keys = [], force_eigv_direction = True, val_core_dats = None, symmetrize_sigmaI = True, basis = 'saiao'):
        """primary load function for Data objects (from chk files)

        Args:
            f (str): .chk file name 
            purge_keys (list, optional): attributes to remove if wanting to save memory. Defaults to [].
            force_eigv_direction (bool, optional): force the SAIAO rotation to force the eigenvector directions. Defaults to True.
            val_core_dats (list, optional): data for redefining core orbitals for seperate projections. Defaults to None.
            symmetrize_sigmaI (bool, optional): symmetryize the sigma(iw) matrices (e.g. if not symmetry for linear solved sigma). Defaults to True.

        Returns:
            Data: the resulting Data for the molecule, with SAIAO features in numpy format
        """
        assert force_eigv_direction
        assert val_core_dats is None
        basis = 'saiao'
        mlf = pyscf.lib.chkfile.load(f, 'mlf')

        if symmetrize_sigmaI and 'sigmaI_ref' in mlf.keys():
            mlf['sigmaI_ref'] = (mlf['sigmaI_ref'] + np.transpose(mlf['sigmaI_ref'], axes=(1, 0, 2)))/2

        md = Data_gwpt({key: value for key, value in mlf.items() if key not in purge_keys})
        
        md.fname = f
        md.basis = basis
        return md
    
    # Calculate dynamical features
    # def calc_dyn(self, dyn_imag_freq_points, ftr_suffix = '', add_ef = True):

    #     if add_ef:
    #         dyn_imag_freq_points = self.ef.copy() + dyn_imag_freq_points.copy()
    #     gf_dyn, hyb_dyn = get_custom_freq_gfhf_features(
    #         self.mo_energy, self[f'fock_{self.basis}'], self[f'C_{self.basis}_mo'], dyn_imag_freq_points, mlf_chkfile = getattr(self, 'fname', ''))
    #     setattr(self, f'gf_dyn{ftr_suffix}', gf_dyn)
    #     setattr(self, f'hyb_dyn{ftr_suffix}', hyb_dyn)

    #     hyb_dyn_off = get_hyb_off(self[f'fock_{self.basis}'], gf_dyn, dyn_imag_freq_points)
    #     setattr(self, f'hyb_dyn_off{ftr_suffix}', hyb_dyn_off)
    
    