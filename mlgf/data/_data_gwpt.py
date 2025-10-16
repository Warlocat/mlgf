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
                mf.__dict__.update({'e_tot': self.e_tot, 'mo_energy': self.mo_energy,
                                    'mo_occ': self.mo_occ, 'mo_coeff': self.mo_coeff})
            else:
                mf.__dict__.update({'mo_energy': self.mo_energy, 'mo_occ': self.mo_occ,
                                    'mo_coeff': self.mo_coeff})
                
            return mf
        else:
            raise ValueError('You need to set the mol attribute before calling get_mf')
    
    @staticmethod
    def load_chk(f, purge_keys = [], symmetrize_sigmaI = True):
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
        basis = 'saiao'
        mlf = pyscf.lib.chkfile.load(f, 'mlf')

        if symmetrize_sigmaI and 'sigmaI_ref' in mlf.keys():
            mlf['sigmaI_ref'] = (mlf['sigmaI_ref'] + np.transpose(mlf['sigmaI_ref'], axes=(1, 0, 2)))/2

        md = Data({key: value for key, value in mlf.items() if key not in purge_keys})
        # if 'mol' in md.data:
        #     if isinstance(md.data['mol'], str):
        #         md.data['mol'] = Mole.loads(md.data['mol'])
        md.fname = f
        md.basis = basis
        return md