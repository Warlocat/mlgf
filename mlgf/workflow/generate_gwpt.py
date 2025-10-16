import os
import numpy as np
import argparse
import scipy
import json
import warnings
from functools import reduce
from mlgf.lib.ml_helper import gGW_mo_saiao, get_sigma_fit, get_orbtypes, get_orb_type, get_core_orbital_indices, get_orbtypes_df, get_saiao_charges, get_saiao_locality

def do_dft_gw_calculation(mol, mol_ref, chkfile, **kwargs):
    """
    do DFT calculation
    
    Args:
        mol (cell): pyscf cell
        mol_ref (cell): pyscf cell for reference geometry
        chkfile (str): pyscf/mlgf chkfile
        **kwawrgs (dict) : calculation parameters.

    Returns:
        dict: dictionary with electronic structure data
    """   
    from pyscf.pbc import dft

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'pbe,pbe')
    init_guess = kwargs.get('init_guess', "minao")
    diis_start_cycle = kwargs.get('diis_start_cycle', 1)
    dm_init = kwargs.get('dm_init', None)
    conv_tol = kwargs.get('conv_tol', 1e-9)
    exxdiv = kwargs.get('exxdiv', None) # for hybrid functional
    max_cycle = kwargs.get('max_cycle', 50)

    if os.path.isfile(chkfile):
        mlf = lib.chkfile.load(chkfile, 'mlf')
        if mlf is None:
            mlf = {}
    else:
        mlf = {}
    
    # Hartree-Fock calculation
    mf = dft.RKS(mol).density_fit()
    mf.chkfile = chkfile
    mf.xc = xc
    mf.conv_tol = conv_tol
    mf.diis_start_cycle = diis_start_cycle
    mf.init_guess = init_guess
    mf.exxdiv = exxdiv
    mf.max_cycle = max_cycle

    for key, val in kwargs.items():
        if key == 'dm_init':
            continue
        if val is not None:
            setattr(mf, key, val)
    
    mf.kernel(dm0 = dm_init)

    # DFT/HF calculation outputs, cheap stuff below
    mlf['e_mf'] = mf.e_tot                         # DFT energy    
    nocc =  mol.nelectron // 2           
    mlf['nocc'] = nocc                             # occupation number/number of electrons
    mlf['mo_occ'] = np.asarray(mf.mo_occ)          # occupation number of each orbital
    mlf['mo_energy'] = np.asarray(mf.mo_energy)    # orbital energy
    mlf['mo_coeff'] = np.asarray(mf.mo_coeff)      # orbital coefficient
    mlf['ovlp'] = np.asarray(mf.get_ovlp())        # overlap matrix
    mlf['hcore'] = np.asarray(mf.get_hcore())      # hcore matrix
    mlf['dm_hf'] = np.asarray(mf.make_rdm1())      # mean field density matrix

    # Fock matrix computed with calling mf.get_fock() which calls mf.get_veff()
    from gwpt.tools.gamma2k import get_fock_from_mo
    mlf['fock'] = np.asarray(get_fock_from_mo(mlf['mo_energy'], mlf['mo_coeff'], mlf['ovlp']))   

    # More expesnive stuff (K matrix computed 2x, J matrix computed 1x)
    vj, vk = mf.get_jk()
    mlf['vj'] = np.asarray(vj)                     # Coulomb matrix
    mlf['vk'] = np.asarray(vk)                     # exchange matrix
    mlf['vxc'] = np.asarray(mf.get_veff() - vj)    # exchange-correlation matrix

    # the definition of the hamiltonian is a bit tricky here, need to multiply by -0.5 to get the correct definition of vk
    mlf['vk_hf'] = -0.5*np.asarray(vk)
    mlf['ef'] = (mf.mo_energy[nocc-1] + mf.mo_energy[nocc]) / 2.0
    mlf['xc'] = xc

    # redundant and reference cell information
    from gwpt.mol.int_redundant import copy_mf_gamma, compact_mat, get_ghost_mole, get_redundant_mole,expand_mat
    cell_redundant = get_redundant_mole([mol, mol_ref])
    cell_ghost = get_ghost_mole(mol, cell_redundant)
    mf_red = copy_mf_gamma(mf, cell_ghost)
    hcore_red = mf_red.get_hcore()
    mlf['hcore_redundant'] = np.asarray(hcore_red)
    mlf['hcore_ref'] = np.asarray(compact_mat(hcore_red, cell_ref, cell_redundant))
    dm_red = expand_mat(mlf['dm_hf'], mol, cell_redundant)
    vj_red, vk_red = mf_red.get_jk(cell_ghost, dm=dm_red)
    mlf['vj_redundant'] = np.asarray(vj_red)
    mlf['vk_redundant'] = np.asarray(vk_red)
    mlf['vj_ref'] = np.asarray(compact_mat(vj_red, cell_ref, cell_redundant))
    mlf['vk_ref'] = np.asarray(compact_mat(vk_red, cell_ref, cell_redundant))
    mlf['vk_hf_ref'] = -0.5*mlf['vk_ref']
    veff_red = mf_red.get_veff(cell_ghost, dm=dm_red)
    mlf['vxc_redundant'] = np.asarray(veff_red - vj_red)
    mlf['vxc_ref'] = np.asarray(compact_mat(veff_red - vj_red, cell_ref, cell_redundant))
    fock_ref = compact_mat(veff_red + hcore_red, cell_ref, cell_redundant)
    mlf['fock_ref'] = np.asarray(fock_ref)
    mlf['ovlp_ref'] = np.asarray(mol_ref.pbc_intor("int1e_ovlp"))
    e, c = scipy.linalg.eigh(fock_ref, mlf['ovlp_ref'])
    mlf['mo_energy_ref'] = np.asarray(e)
    mlf['mo_coeff_ref'] = np.asarray(c)
    mocc_ref = c[:,mf.mo_occ>0]
    dm_ref = (mocc_ref*mf.mo_occ[mf.mo_occ>0]).dot(mocc_ref.conj().T)
    mlf['dm_hf_ref'] = np.asarray(dm_ref)

    from gwpt.pbc.gw_ac_red import GWAC_RED
    gw_red = GWAC_RED(mf, cell_redundant)
    nw = kwargs.get('nw', 100)
    nw2 = kwargs.get('nw2', None)
    orbs = kwargs.get('orbs', None)
    frozen = kwargs.get('frozen', None)
    ac_iw_cutoff = kwargs.get('ac_iw_cutoff', 5.0)
    freqs = kwargs.get('freqs', None)
    wts = kwargs.get('wts', None)
    ac_idx = kwargs.get('ac_idx', None)
    gw_red.rdm = True
    gw_red.ac = 'pade'
    gw_red.nw = nw
    gw_red.nw2 = nw2
    gw_red.ac_iw_cutoff = ac_iw_cutoff
    gw_red.frozen = frozen
    gw_red.orbs = orbs
    gw_red.verbose = 5
    gw_red.freqs = freqs # evaluations grid freqs
    gw_red.wts = wts # evaluation grid wts
    gw_red.ac_idx = ac_idx

    gw_red.partial_kernel(with_df=mf_red.with_df)
    ef = gw_red.get_ef(mo_energy=gw_red.mo_energy)
    
    # start from first nonzero frequency point as ML target, first point is 0.
    omega_fit = gw_red.freqs*1.0j + ef
    sigmaI_redundant = gw_red.sigmaI[:,:,1:]
    sigmaI_ref = []
    for iw in range(sigmaI_redundant.shape[2]):
        sigmaI_ref.append(compact_mat(sigmaI_redundant[:,:,iw], cell_ref, cell_redundant))
    sigmaI_ref = np.asarray(sigmaI_ref).transpose(1,2,0)
    
    # GW part
    for name, obj in zip(['ef','freqs', 'wts', 'sigmaI_redundant', 'sigmaI_ref', 'omega_fit'],
                     [ef, gw_red.freqs, gw_red.wts, sigmaI_redundant, sigmaI_ref, omega_fit]):
        mlf[name] = np.asarray(obj)

    # MO to SAIAO basis
    from mlgf.lib.ml_helper import get_chk_saiao
    mf_ref_fake = dft.RKS(mol_ref)
    mf_ref_fake.xc = xc
    mf_ref_fake.mo_energy = mlf['mo_energy_ref']
    mf_ref_fake.mo_coeff = mlf['mo_coeff_ref']
    mf_ref_fake.mo_occ = mlf['mo_occ']

    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf_ref_fake, mlf['fock_ref'], minao = "gth-cc-pvdz-lc-minao")
    C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)
    mlf = get_saiao_features(mol_ref, mlf, C_ao_saiao)

    mlf['fock_iao'] = fock_iao
    mlf['C_ao_iao'] = C_ao_iao
    mlf['C_iao_saiao'] = C_iao_saiao
    mlf['hcore+vj_saiao'] = mlf['hcore_saiao'] + mlf['vj_saiao']
    mlf['inds_core'] = get_core_orbital_indices(mol_ref)

    df_mol = get_orbtypes_df(mol_ref)
    mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    mlf['atomic_charge_saiao'] = get_saiao_charges(df_mol, mlf['dm_saiao'])
    mlf['boys_saiao'] = get_saiao_locality(mol_ref, mlf['C_ao_saiao'])

    return mlf

def get_saiao_features(mol, mlf, C_ao_saiao, categorical = True):
    """get ML features in SAIAO basis

    Args:
        mol : pyscf mol object
        custom_chkfile (string): mlgf chkfile object from generate.py
        C_ao_saao (np.float64, norb x norb): rotation matrix from AO to SAIAO
        categorical (boolean): whether to generate integer valued features for quantum numbers and orbital type

    Returns:
        dict: modified mlf dictionary with SAIAO basis features
    """    


    basis_name = 'saiao'
    mlf['C_ao_saiao'] = C_ao_saiao

    mo_energy = mlf['mo_energy_ref']
    mo_coeff = mlf['mo_coeff_ref']
    S_ao = mlf['ovlp_ref']
    nocc = mlf['nocc']
    dm = mlf['dm_hf_ref']
    nelectron = nocc*2
            
    # feature 1: density matrix
    dm_saiao = C_ao_saiao.T @ S_ao @ dm @ S_ao @ C_ao_saiao
    abs_diff_particle_number = abs(np.trace(dm_saiao)-nelectron)
    if abs_diff_particle_number > 1e-8:
        warnings.warn(f'dm_saiao particle number diff {abs_diff_particle_number:0.6e}')

    # feature 2 : Fock matrix
    fock = mlf['fock_ref']
    fock_saiao = C_ao_saiao.T @ fock @ C_ao_saiao

    # feature 3 : hcore matrix
    hcore = mlf['hcore_ref']
    hcore_saiao = C_ao_saiao.T @ hcore @ C_ao_saiao

    # feature 4 & 5 : J and K matrices
    vj, vk = mlf['vj_ref'], mlf['vk_ref']
    vj_saiao = C_ao_saiao.T @ vj @ C_ao_saiao
    vk_saiao = C_ao_saiao.T @ vk @ C_ao_saiao

    # feature 6 : mean-field GF (imag freq)
    # GF in MO basis on (ef + iw_n)
    ef = (mo_energy[nocc-1] + mo_energy[nocc]) / 2

    # GF in SAIAO basis
    C_saiao_mo = C_ao_saiao.T @ S_ao @ mo_coeff
    C_mo_saiao = C_saiao_mo.T
    
    selected_freqs = mlf['omega_fit'] #ALREADY IMAG
    full_sigma = mlf['sigmaI_ref'] # full sigma (>> len(omegaI))
    full_freqs = mlf['freqs']

    sigma_fit = get_sigma_fit(full_sigma, full_freqs, selected_freqs)
    sigma_saiao = gGW_mo_saiao(sigma_fit, C_mo_saiao)
    mlf[f'sigma_{basis_name}'] = sigma_saiao
    
    # mlf = {}
    mlf[f'dm_{basis_name}'] = dm_saiao
    mlf[f'fock_{basis_name}'] = fock_saiao
    mlf[f'hcore_{basis_name}'] = hcore_saiao
    mlf[f'vj_{basis_name}'] = vj_saiao
    mlf[f'vk_{basis_name}'] = vk_saiao
    mlf[f'C_{basis_name}_mo'] = C_saiao_mo

    if 'vxc_ref' in mlf.keys():
        vxc_saiao = C_ao_saiao.T @ mlf['vxc_ref'] @ C_ao_saiao
        mlf[f'vxc_{basis_name}'] = vxc_saiao

    if categorical:
        mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = get_orbtypes(mol)
        mlf['cat_orbtype_saiao'] = get_orb_type(mol, dm_saiao)
    return mlf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generate.py')
    parser.add_argument('--poscar', required = True, help='poscar file to start DFT calculation from.')
    parser.add_argument('--poscar_ref', required = True, help='poscar file for reference geometry.')
    parser.add_argument('--chk_file', required = True, help='.chk file for reading and writing electronic structure data.')
    
    # system parameters are command line args
    parser.add_argument('--charge', required = False, default=0, help='system charge, assumed neutral')
    parser.add_argument('--basis', required = False, default='gth-cc-pvdz-lc', help='basis set for DFT calculation.')
    parser.add_argument('--pseudo', required = False, default='gth-pbe', help='pseudopotential for DFT calculation.')   
    
    # calculation parameters stored in json
    parser.add_argument('--json_spec', required = False, default=None, help='json file holding keyword arguments for GW calculation')
    args = parser.parse_args()

    if args.json_spec is not None:
        assert('.json' in args.json_spec)
        with open(args.json_spec) as f:
            spec = json.load(f)
    else:
        spec = {}
    
    verbose = spec.get('verbose', 0)
    chk_file = args.chk_file
    poscar = args.poscar
    poscar_ref = args.poscar_ref

    from libdmet.utils.iotools import read_poscar
    cell = read_poscar(poscar)
    cell_ref = read_poscar(poscar_ref)

    cell.basis = args.basis
    cell.pseudo = args.pseudo
    cell.charge = args.charge
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build()
    cell_ref.basis = args.basis
    cell_ref.pseudo = args.pseudo
    cell_ref.charge = args.charge
    cell_ref.precision = 1e-12
    cell_ref.build()

    from pyscf import lib
    mlf = do_dft_gw_calculation(cell, cell_ref, chk_file, **spec)
    lib.chkfile.save(chk_file, 'mlf', mlf)



            


    



