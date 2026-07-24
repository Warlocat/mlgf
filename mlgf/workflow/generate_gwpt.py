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
    outcore = kwargs.get('outcore', False)
    segsize = kwargs.get('segsize', 100)
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
    # mf.grids.level = 9
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
    from gwpt.mol.int_redundant import copy_mf_gamma, compact_mat, get_ghost_mole, get_redundant_mole, expand_mat, get_backcoeff_red
    cell_redundant = get_redundant_mole([mol, mol_ref])
    cell_ghost = get_ghost_mole(mol, cell_redundant)
    mf_red = copy_mf_gamma(mf, cell_ghost)
    hcore_red = mf_red.get_hcore()
    mlf['hcore_redundant'] = np.asarray(hcore_red)
    mlf['hcore_ref'] = np.asarray(compact_mat(hcore_red, mol_ref, cell_redundant))
    dm_red = expand_mat(mlf['dm_hf'], mol, cell_redundant)
    veff_red = mf_red.get_veff(cell_ghost, dm=dm_red)
    vj_red, vk_red = mf_red.get_jk(cell_ghost, dm=dm_red)
    
    mlf['vj_redundant'] = np.asarray(vj_red)
    mlf['vk_redundant'] = np.asarray(vk_red)
    mlf['vj_ref'] = np.asarray(compact_mat(vj_red, mol_ref, cell_redundant))
    mlf['vk_ref'] = np.asarray(compact_mat(vk_red, mol_ref, cell_redundant))
    mlf['vk_hf_ref'] = -0.5*mlf['vk_ref']
    
    mlf['vxc_redundant'] = np.asarray(veff_red - vj_red)
    mlf['vxc_ref'] = np.asarray(compact_mat(veff_red - vj_red, mol_ref, cell_redundant))
    fock_ref = compact_mat(veff_red + hcore_red, mol_ref, cell_redundant)
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
    gw_red.outcore = outcore
    gw_red.segsize = segsize

    gw_red.partial_kernel(with_df=mf_red.with_df)
    ef = gw_red.get_ef(mo_energy=gw_red.mo_energy)
    
    # start from first nonzero frequency point as ML target, first point is 0.
    omega_fit = gw_red.freqs*1.0j + ef
    sigmaI_redundant_mo = gw_red.sigmaI[:,:,1:]

    back_coeff = get_backcoeff_red(gw_red.mo_coeff[:, gw_red.orbs], gw_red.mol, gw_red.mol_red)
    sigmaI_redundant_ao = np.zeros((gw_red.mol_red.nao_nr(), gw_red.mol_red.nao_nr(), sigmaI_redundant_mo.shape[-1]), 
                                dtype=sigmaI_redundant_mo.dtype)
    for iw in range(sigmaI_redundant_mo.shape[-1]):
        sigmaI_redundant_ao[:, :, iw] = back_coeff @ sigmaI_redundant_mo[:, :, iw] @ back_coeff.T

    # if with_diff_ref is not None:
    #     print('Calculating difference with reference calculation from ', with_diff_ref)
    #     mlf_diff_ref = lib.chkfile.load(with_diff_ref, 'mlf')
    #     sigmaI_diff_ref_ao = mlf_diff_ref['sigmaI_ref_ao']
    # else:
    #     sigmaI_diff_ref_ao = None

    sigmaI_ref_ao = []
    sigmaI_ref_mo = []
    for iw in range(sigmaI_redundant_ao.shape[2]):
        sigmaI_ref_ao.append(compact_mat(sigmaI_redundant_ao[:,:,iw], mol_ref, cell_redundant))
        # if sigmaI_diff_ref_ao is not None:
        #     sigmaI_ref_ao[-1] -= sigmaI_diff_ref_ao[:,:,iw]
        sigmaI_ref_mo.append(mlf['mo_coeff_ref'].T @ sigmaI_ref_ao[-1] @ mlf['mo_coeff_ref'])
    sigmaI_ref_mo = np.asarray(sigmaI_ref_mo).transpose(1,2,0)
    sigmaI_ref_ao = np.asarray(sigmaI_ref_ao).transpose(1,2,0)
    
    # GW part
    for name, obj in zip(['ef','freqs', 'wts', 'sigmaI_ref', 'sigmaI_ref_ao', 'omega_fit'],
                     [ef, gw_red.freqs, gw_red.wts, sigmaI_ref_mo, sigmaI_ref_ao, omega_fit]):
        mlf[name] = np.asarray(obj)

    # MO to SAIAO basis
    # from mlgf.lib.ml_helper import get_chk_saiao
    # mf_ref_fake = dft.RKS(mol_ref)
    # mf_ref_fake.xc = xc
    # mf_ref_fake.mo_energy = mlf['mo_energy_ref']
    # mf_ref_fake.mo_coeff = mlf['mo_coeff_ref']
    # mf_ref_fake.mo_occ = mlf['mo_occ']

    # C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf_ref_fake, mlf['fock_ref'], minao = "gth-cc-pvdz-lc-minao")
    # C_ao_saiao = np.dot(C_ao_iao, C_iao_saiao)
    # mlf = get_saiao_features(mol_ref, mlf, C_ao_saiao)

    # mlf['fock_iao'] = fock_iao
    # mlf['C_ao_iao'] = C_ao_iao
    # mlf['C_iao_saiao'] = C_iao_saiao
    # mlf['hcore+vj_saiao'] = mlf['hcore_saiao'] + mlf['vj_saiao']
    # mlf['inds_core'] = get_core_orbital_indices(mol_ref)

    # df_mol = get_orbtypes_df(mol_ref)
    # mlf['cat_orbtype_principal'], mlf['cat_orbtype_angular'] = np.diag(df_mol['principal']), np.diag(df_mol['angular'])
    # mlf['atomic_charge_saiao'] = get_saiao_charges(df_mol, mlf['dm_saiao'])
    # mlf['boys_saiao'] = get_saiao_locality(mol_ref, mlf['C_ao_saiao'])

    return mlf

def do_dft_calculation(mol, mol_ref, chkfile, force = True, **kwargs):
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
    # mf.grids.level = 9
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

    if force:
        mf_mg2 = dft.RKS(mol)
        mf_mg2.xc = xc
        mf_mg2.conv_tol = conv_tol
        mf_mg2.diis_start_cycle = diis_start_cycle
        mf_mg2.exxdiv = exxdiv
        mf_mg2.max_cycle = max_cycle
        mf_mg2._numint = dft.multigrid.MultiGridNumInt2(mol)
        mf_mg2.kernel(dm0 = mf.make_rdm1())
        mygrad = mf_mg2.nuc_grad_method()
        grad = mygrad.kernel()
        mlf['gradient'] = np.asarray(grad)

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
    from gwpt.mol.int_redundant import copy_mf_gamma, compact_mat, get_ghost_mole, get_redundant_mole, expand_mat, get_backcoeff_red
    cell_redundant = get_redundant_mole([mol, mol_ref])
    cell_ghost = get_ghost_mole(mol, cell_redundant)
    mf_red = copy_mf_gamma(mf, cell_ghost)
    hcore_red = mf_red.get_hcore()
    mlf['hcore_redundant'] = np.asarray(hcore_red)
    mlf['hcore_ref'] = np.asarray(compact_mat(hcore_red, mol_ref, cell_redundant))
    dm_red = expand_mat(mlf['dm_hf'], mol, cell_redundant)
    veff_red = mf_red.get_veff(cell_ghost, dm=dm_red)
    vj_red, vk_red = mf_red.get_jk(cell_ghost, dm=dm_red)
    
    mlf['vj_redundant'] = np.asarray(vj_red)
    mlf['vk_redundant'] = np.asarray(vk_red)
    mlf['vj_ref'] = np.asarray(compact_mat(vj_red, mol_ref, cell_redundant))
    mlf['vk_ref'] = np.asarray(compact_mat(vk_red, mol_ref, cell_redundant))
    mlf['vk_hf_ref'] = -0.5*mlf['vk_ref']
    
    mlf['vxc_redundant'] = np.asarray(veff_red - vj_red)
    mlf['vxc_ref'] = np.asarray(compact_mat(veff_red - vj_red, mol_ref, cell_redundant))
    fock_ref = compact_mat(veff_red + hcore_red, mol_ref, cell_redundant)
    mlf['fock_ref'] = np.asarray(fock_ref)
    mlf['ovlp_ref'] = np.asarray(mol_ref.pbc_intor("int1e_ovlp"))
    e, c = scipy.linalg.eigh(fock_ref, mlf['ovlp_ref'])
    mlf['mo_energy_ref'] = np.asarray(e)
    mlf['mo_coeff_ref'] = np.asarray(c)
    mocc_ref = c[:,mf.mo_occ>0]
    dm_ref = (mocc_ref*mf.mo_occ[mf.mo_occ>0]).dot(mocc_ref.conj().T)
    mlf['dm_hf_ref'] = np.asarray(dm_ref)

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

def do_kgw_ref_calculation(mol, kmesh, chkfile, **kwargs):
    from pyscf.pbc import dft
    from gwpt.tools.gamma2k import k2gamma_mat

    # Assign variables from kwargs with default values
    xc = kwargs.get('xc', 'pbe,pbe')
    init_guess = kwargs.get('init_guess', "minao")
    diis_start_cycle = kwargs.get('diis_start_cycle', 1)
    dm_init = kwargs.get('dm_init', None)
    conv_tol = kwargs.get('conv_tol', 1e-9)
    exxdiv = kwargs.get('exxdiv', None) # for hybrid functional
    max_cycle = kwargs.get('max_cycle', 50)
    kpts = mol.make_kpts(kmesh)

    if os.path.isfile(chkfile):
        mlf = lib.chkfile.load(chkfile, 'mlf')
        if mlf is None:
            mlf = {}
    else:
        mlf = {}
    
    # Hartree-Fock calculation
    kmf = dft.KRKS(mol, kpts).density_fit()
    kmf.with_df.build(j_only=False)
    kmf.chkfile = chkfile
    # kmf.grids.level = 9
    kmf.xc = xc
    kmf.conv_tol = conv_tol
    kmf.diis_start_cycle = diis_start_cycle
    kmf.init_guess = init_guess
    kmf.exxdiv = exxdiv
    kmf.max_cycle = max_cycle    
    kmf.kernel(dm0 = dm_init)

    fock_k = kmf.get_fock()
    ovlp_k = kmf.get_ovlp()
    hcore_k = kmf.get_hcore()
    j_k, k_k = kmf.get_jk()
    veff_k = kmf.get_veff()

    fock_r = k2gamma_mat(fock_k, mol, kpts).real
    ovlp_r = k2gamma_mat(ovlp_k, mol, kpts).real
    hcore_r = k2gamma_mat(hcore_k, mol, kpts).real
    vj_r = k2gamma_mat(j_k, mol, kpts).real
    vk_r = k2gamma_mat(k_k, mol, kpts).real
    veff_r = k2gamma_mat(veff_k, mol, kpts).real
    mlf['fock'] = np.asarray(fock_r)
    mlf['ovlp'] = np.asarray(ovlp_r)
    mlf['hcore'] = np.asarray(hcore_r)
    mlf['vj'] = np.asarray(vj_r)
    mlf['vk'] = np.asarray(vk_r)
    mlf['vk_hf'] = -0.5*np.asarray(vk_r)
    mlf['vxc'] = np.asarray(veff_r - vj_r)

    e, c = scipy.linalg.eigh(fock_r, ovlp_r)
    mlf['mo_energy'] = np.asarray(e)
    mlf['mo_coeff'] = np.asarray(c)
    nocc = mol.nelectron * len(kpts) // 2
    mlf['nocc'] = nocc
    mo_occ = np.zeros_like(e)
    mo_occ[:nocc] = 2
    mlf['mo_occ'] = np.asarray(mo_occ)
    mocc = c[:, :nocc]
    dm = mocc @ mocc.conj().T * 2.0
    mlf['dm_hf'] = np.asarray(dm)

    # MO to SAIAO basis
    from mlgf.lib.ml_helper import get_chk_saiao
    from pyscf.pbc.tools import super_cell
    supercell = super_cell(mol, kmesh)
    mf_ref_fake = dft.RKS(supercell).density_fit()
    mf_ref_fake.xc = xc
    mf_ref_fake.exxdiv = exxdiv
    mf_ref_fake.mo_energy = mlf['mo_energy']
    mf_ref_fake.mo_coeff = mlf['mo_coeff']
    mf_ref_fake.mo_occ = mlf['mo_occ']

    C_ao_iao, C_iao_saiao, fock_iao = get_chk_saiao(mf_ref_fake, mlf['fock'], minao = "gth-cc-pvdz-lc-minao")

    from mlgf.lib.ml_helper import get_saiao_features as get_saiao_features_original
    get_saiao_features_original(supercell, mlf, C_ao_iao @ C_iao_saiao)

    mlf['fock_iao'] = fock_iao
    mlf['C_ao_iao'] = C_ao_iao
    mlf['C_iao_saiao'] = C_iao_saiao

    from fcdmft.gw.pbc.krgw_ac import KRGWAC
    kgw = KRGWAC(kmf)
    nw = kwargs.get('nw', 100)
    nw2 = kwargs.get('nw2', None)
    orbs = kwargs.get('orbs', None)
    frozen = kwargs.get('frozen', None)
    ac_iw_cutoff = kwargs.get('ac_iw_cutoff', 5.0)
    freqs = kwargs.get('freqs', None)
    wts = kwargs.get('wts', None)
    ac_idx = kwargs.get('ac_idx', None)
    kgw.rdm = True
    kgw.fullsigma = True
    kgw.ac = 'pade'
    kgw.nw = nw
    kgw.nw2 = nw2
    kgw.ac_iw_cutoff = ac_iw_cutoff
    kgw.frozen = frozen
    kgw.orbs = orbs
    kgw.verbose = 5
    kgw.fc = False
    kgw.freqs = freqs # evaluations grid freqs
    kgw.wts = wts # evaluation grid wts
    kgw.ac_idx = ac_idx

    kgw.kernel()
    ksigmaI_mo = kgw.sigmaI[:,:,:,1:]

    mo_coeff = np.array(kgw.mo_coeff)[:, :, kgw.orbs]
    ksigmaI_ao = np.zeros((ksigmaI_mo.shape[0], mol.nao_nr(), mol.nao_nr(), ksigmaI_mo.shape[-1]), 
                                dtype=ksigmaI_mo.dtype)
    for ik in range(ksigmaI_mo.shape[0]):
        sc = ovlp_k[ik] @ mo_coeff[ik]
        for iw in range(ksigmaI_mo.shape[-1]):
            ksigmaI_ao[ik,:,:,iw] = sc @ ksigmaI_mo[ik,:,:,iw] @ sc.T.conj()

    sigmaI_ao = np.zeros((ksigmaI_ao.shape[0]*mol.nao_nr(), ksigmaI_ao.shape[0]*mol.nao_nr(), ksigmaI_ao.shape[-1]), dtype=ksigmaI_ao.dtype)
    for iw in range(ksigmaI_ao.shape[-1]):
        sigmaI_ao[:,:,iw] = k2gamma_mat(ksigmaI_ao[:,:,:,iw], mol, kpts)
    mlf['freqs'] = np.asarray(kgw.freqs)
    mlf['wts'] = np.asarray(kgw.wts)
    mlf['sigmaI_ref_ao'] = np.asarray(sigmaI_ao)

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
    
    if "omega_fit" in mlf.keys():
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

    parser.add_argument('--calc', required = True, default='dft', help='string specifying which calculation(s) to preform. Multiple calculations are specified by seperating with + symbol')

    parser.add_argument('--kmesh', required = False, default='1,1,1', help='k-point mesh for kdft reference calculation.')

    # whether to have reference calculation to calculate difference
    # parser.add_argument('--with_diff_ref', required = False, default=None, help='whether to have reference calculation to calculate difference')
    
    # calculation parameters stored in json
    parser.add_argument('--json_spec', required = False, default=None, help='json file holding keyword arguments for GW calculation')
    args = parser.parse_args()

    if args.json_spec is not None:
        assert('.json' in args.json_spec)
        with open(args.json_spec) as f:
            spec = json.load(f)
    else:
        spec = {}

    available_calc_types = [
        "dft+gwac", "dft+force", "dft", "kgw",
    ]
    calculation_type = args.calc.lower()
    assert calculation_type in available_calc_types, f'--calc must be 1 of {available_calc_types}.'
    
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
    if calculation_type == 'dft':
        mlf = do_dft_calculation(cell, cell_ref, chk_file, force=False, **spec)
    elif calculation_type == 'dft+force':
        mlf = do_dft_calculation(cell, cell_ref, chk_file, force=True, **spec)
    elif calculation_type == 'dft+gwac':
        mlf = do_dft_gw_calculation(cell, cell_ref, chk_file, **spec)
    elif calculation_type == 'kgw':
        kmesh = [int(x) for x in args.kmesh.split(',')]
        mlf = do_kgw_ref_calculation(cell, kmesh, chk_file, **spec)
    lib.chkfile.save(chk_file, 'mlf', mlf)



            


    



