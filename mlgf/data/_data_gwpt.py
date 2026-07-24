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
    def load_chk(f, f_ref, purge_keys = [], force_eigv_direction = True, val_core_dats = None, symmetrize_sigmaI = True, basis = 'saiao'):
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
        mlf = pyscf.lib.chkfile.load(f, 'mlf')

        if symmetrize_sigmaI and 'sigmaI_ref' in mlf.keys():
            mlf['sigmaI_ref'] = (mlf['sigmaI_ref'] + np.transpose(mlf['sigmaI_ref'], axes=(1, 0, 2)))/2
        if symmetrize_sigmaI and 'sigmaI_ref_ao' in mlf.keys():
            mlf['sigmaI_ref_ao'] = (mlf['sigmaI_ref_ao'] + np.transpose(mlf['sigmaI_ref_ao'], axes=(1, 0, 2)))/2
            if 'sigmaI_ref' not in mlf.keys():
                nmo = mlf['mo_coeff_ref'].shape[1]
                mlf['sigmaI_ref'] = np.zeros((nmo, nmo, mlf['sigmaI_ref_ao'].shape[2]), dtype=mlf['sigmaI_ref_ao'].dtype)
                for iw in range(mlf['sigmaI_ref_ao'].shape[2]):
                    mlf['sigmaI_ref'][:,:,iw] = mlf['mo_coeff_ref'].T @ mlf['sigmaI_ref_ao'][:,:,iw] @ mlf['mo_coeff_ref']

        mlf_ref = pyscf.lib.chkfile.load(f_ref, 'mlf')
        ucell_ref = pyscf.pbc.lib.chkfile.load_cell(f_ref)
        kpts = pyscf.lib.chkfile.load(f_ref, "scf/kpts")
        from pyscf.pbc.tools.k2gamma import get_phase
        scell_ref = get_phase(ucell_ref, kpts)[0]

        from mlgf.lib.ml_helper import get_chk_saiao
        from pyscf.pbc import dft
        mf_ref_fake = dft.RKS(scell_ref)
        mf_ref_fake.xc = mlf['xc']
        mf_ref_fake.mo_energy = mlf['mo_energy_ref']
        mf_ref_fake.mo_coeff = mlf['mo_coeff_ref']
        mf_ref_fake.mo_occ = mlf['mo_occ']
        C_ao_iao, C_iao_saiao, _ = get_chk_saiao(mf_ref_fake, mlf['fock_ref'], minao = "gth-cc-pvdz-lc-minao", force_eigv_direction = force_eigv_direction)
        C_ao_saiao = C_ao_iao @ C_iao_saiao
        
        sigmaI_ao = mlf['sigmaI_ref_ao']
        sigmaI_ao_diff_ref = mlf_ref['sigmaI_ref_ao']
        sigmaI_ao_diff = []
        sigmaI_mo_diff = []
        for iw in range(sigmaI_ao.shape[2]):
            sigmaI_ao_diff.append(sigmaI_ao[:,:,iw] - sigmaI_ao_diff_ref[:,:,iw])
            sigmaI_mo_diff.append(mlf['mo_coeff_ref'].T @ sigmaI_ao_diff[-1] @ mlf['mo_coeff_ref'])
        sigmaI_ao_diff = np.asarray(sigmaI_ao_diff).transpose(1,2,0)
        sigmaI_mo_diff = np.asarray(sigmaI_mo_diff).transpose(1,2,0)
        mlf['sigmaI_ref_ao_diff'] = sigmaI_ao_diff
        mlf['sigmaI_ref_diff'] = sigmaI_mo_diff

        # overwrite interface features
        mlf['mo_coeff'] = mlf['mo_coeff_ref']
        mlf['mo_energy'] = mlf['mo_energy_ref']

        from mlgf.lib.ml_helper import get_saiao_features_with_mol_ref
        get_saiao_features_with_mol_ref(scell_ref, mlf, C_ao_saiao)

        # ref data
        mf_ref_fake = dft.RKS(scell_ref).density_fit()
        mf_ref_fake.mo_energy = mlf_ref['mo_energy']
        mf_ref_fake.mo_coeff = mlf_ref['mo_coeff']
        mf_ref_fake.mo_occ = mlf_ref['mo_occ']
        from mlgf.lib.ml_helper import get_saiao_features as get_saiao_features_original
        get_saiao_features_original(scell_ref, mlf_ref, C_ao_saiao)

        mlf["dm_saiao_diff"] = mlf["dm_saiao"] - mlf_ref["dm_saiao"]
        mlf["fock_saiao_diff"] = mlf["fock_saiao"] - mlf_ref["fock_saiao"]
        mlf["hcore+vj_saiao_diff"] = mlf["hcore+vj_saiao"] - mlf_ref["hcore_saiao"] - mlf_ref["vj_saiao"]
        mlf["hcore_saiao_diff"] = mlf["hcore_saiao"] - mlf_ref["hcore_saiao"]
        mlf["vk_saiao_diff"] = mlf["vk_saiao"] - mlf_ref["vk_saiao"]
        mlf["vj_saiao_diff"] = mlf["vj_saiao"] - mlf_ref["vj_saiao"]
        mlf["vxc_saiao_diff"] = mlf["vxc_saiao"] - mlf_ref["vxc_saiao"]

        md = Data_gwpt({key: value for key, value in mlf.items() if key not in purge_keys})
        
        md.fname = f
        md.basis = basis
        return md
    
    # Calculate dynamical features
    def calc_dyn(self, dyn_imag_freq_points, ftr_suffix = '', add_ef = True):
        from mlgf.lib.ml_helper import get_custom_freq_gfhf_features, get_hyb_off
        if add_ef:
            # working with full space mo_energy and nocc here
            nocc = self["nocc"]
            ef_ref = (self["mo_energy_ref"][nocc - 1] + self["mo_energy_ref"][nocc]) * 0.5
            dyn_imag_freq_points = ef_ref.copy() + dyn_imag_freq_points.copy()
        gf_dyn, hyb_dyn = get_custom_freq_gfhf_features(
            self['mo_energy_ref'], self[f'fock_{self.basis}'], self[f'C_{self.basis}_mo'], dyn_imag_freq_points, mlf_chkfile = getattr(self, 'fname', ''))
        setattr(self, f'gf_dyn{ftr_suffix}', gf_dyn)
        setattr(self, f'hyb_dyn{ftr_suffix}', hyb_dyn)        

        hyb_dyn_off = get_hyb_off(self[f'fock_{self.basis}'], gf_dyn, dyn_imag_freq_points)
        setattr(self, f'hyb_dyn_off{ftr_suffix}', hyb_dyn_off)
    
    