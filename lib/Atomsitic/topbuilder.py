#!/usr/bin/env python
import sys
import numpy as np

def DeduceTopolgy(_bonds):
    '''
    deduce all angles, dihdedrals from input bonds info.
    '''
    nbonds = len(_bonds)
    _angles = []
    # angles
    for i in range(nbonds):
        for j in range(i+1, nbonds):
            a = _bonds[i]
            b = _bonds[j]
            center = list(set(a).intersection(set(b)))
            if len(center) == 0:
                continue
            else:
                diff = list(set(b).difference(set(a))) + list(set(a).difference(set(b)))
                _max = diff[0] if diff[0] > diff[1] else diff[1]
                _min = diff[0] if diff[0] < diff[1] else diff[1]
                angle = [_min, center[0], _max]
                _angles.append(angle)
    #  print(_angles)

    # dihs
    nangles = len(_angles)
    _dihs = []
    for i in range(nangles):
        for j in range(i+1, nangles):
            a = _angles[i] 
            b = _angles[j]
            if a[1] == b[1]: 
                continue
            else:
                center = list(set(a).intersection(set(b)))
                if len(center) != 2:
                    continue
                else:
                    diff = list(set(b).difference(set(a))) + list(set(a).difference(set(b)))
                    _max = diff[0] if diff[0] > diff[1] else diff[1]
                    _min = diff[0] if diff[0] < diff[1] else diff[1]
                    dih = [_min, center[0], center[1], _max]
                    _dihs.append(dih)
    #  print(_dihs)
    return _angles, _dihs


def DefineForceField():
    '''
    define force field
    '''
    class force_field:
        def __init__(self, _name):
            self.name = _name
        class cls_atomtypes:
            pass
        class cls_bondtypes:
            pass
        class cls_angletypes:
            pass
        class cls_dihedraltypes:
            pass

    _F = force_field('jxm')
    _F.atomtypes = _F.cls_atomtypes()
    _F.atomtypes.name = {}
    _F.atomtypes.name['jxm_001'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.180, 'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_002'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.120, 'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_003'] = {'bond_type':'CT', 'mass':'12.0110', 'charge':-0.060, 'sigma':'3.50000E-01', 'epsilon':'2.76144E-01'}
    _F.atomtypes.name['jxm_004'] = {'bond_type':'HC', 'mass':'1.0080',  'charge':0.060,  'sigma':'2.50000E-01', 'epsilon':'1.25520E-01'}

    _F.bondtypes = _F.cls_bondtypes()
    _F.bondtypes.name = {}
    _F.bondtypes.name['CT-CT'] = {'func':1, 'parameter':'0.1529 224262.400'}
    _F.bondtypes.name['CT-HC'] = {'func':1, 'parameter':'0.1090 284512.000'}

    _F.angletypes = _F.cls_angletypes()
    _F.angletypes.name = {}
    _F.angletypes.name['CT-CT-HC'] = {'func':1, 'parameter':'110.700    313.800'}
    _F.angletypes.name['HC-CT-HC'] = {'func':1, 'parameter':'107.800    276.144'}
    _F.angletypes.name['CT-CT-CT'] = {'func':1, 'parameter':'112.700    488.273'}

    _F.dihedraltypes = _F.cls_dihedraltypes()
    _F.dihedraltypes.name = {} 
    _F.dihedraltypes.name['HC-CT-CT-HC'] = {'func':3, 'parameter':'0.628   1.883   0.000  -2.510  -0.000   0.000'}
    _F.dihedraltypes.name['CT-CT-CT-CT'] = {'func':3, 'parameter':'2.301  -1.464   0.837  -1.674  -0.000   0.000'}
    _F.dihedraltypes.name['HC-CT-CT-CT'] = {'func':3, 'parameter':'0.628   1.883   0.000  -2.510  -0.000   0.000'}

    return _F


def DefineResidueTypes():
    '''
    define residue type, similar to the rtp file format in GROMACS.
    '''
    class residue_type(object):
        def __init__(self, _name, _atoms, _atom_types, _bonds, _angles, _dihs, _head, _tail):
            self.name = _name 
            self.atoms = _atoms
            self.atom_types = _atom_types
            self.bonds = np.array(_bonds).astype(np.int)
            self.angles = _angles
            self.dihs = _dihs
            self.head = _head
            self.tail = _tail

            self.natoms = len(self.atoms)

        def get_linkbond(self, _index):
            return self.bonds[np.argwhere(self.bonds == _index)[:, 0]]

    _ResidueTypes = {} 

    # define residue: ETH-head
    name = 'ETH'
    atoms = ['HT', 'C1', 'C2', 'H1C1', 'H2C1', 'H1C2', 'H2C2']
    atom_types = ['jxm_004', 'jxm_001', 'jxm_002', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004']
    bonds = [(0, 1), (1,2), (1,3), (1,4), (2, 5), (2,6)]
    angles, dihs = DeduceTopolgy(bonds)
    head = None 
    tail = atoms.index('C2')
    _ResidueTypes[name] = residue_type(name, atoms, atom_types, bonds, angles, dihs, head, tail)

    # define residue: ETH-middle
    name = 'ETM'
    atoms = ['C1', 'C2', 'H1C1', 'H2C1', 'H1C2', 'H2C2']
    atom_types = ['jxm_002', 'jxm_002', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004']
    bonds = [(0, 1), (0,2), (0,3), (1,4), (1, 5)]
    angles, dihs = DeduceTopolgy(bonds)
    head = atoms.index('C1')
    tail = atoms.index('C2')
    _ResidueTypes[name] = residue_type(name, atoms, atom_types, bonds, angles, dihs, head, tail)

    # define residue: ETH-middle
    name = 'ETT'
    atoms = ['C1', 'C2', 'H1C1', 'H2C1', 'H1C2', 'H2C2', 'HH']
    atom_types = ['jxm_002', 'jxm_001', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004']
    bonds = [(0, 1), (0,2), (0,3), (1,4), (1, 5), (1, 6)]
    angles, dihs = DeduceTopolgy(bonds)
    head = atoms.index('C1')
    tail = None 
    _ResidueTypes[name] = residue_type(name, atoms, atom_types, bonds, angles, dihs, head, tail)

    # define residue: NB
    name = 'NB'
    atoms = ['C1', 'C6', 'C2', 'HC1', 'C5', 'HC6', 'C3', 'C7', 'HC2', 'C4', 'HC5', 'H1C3', 'H2C3', 'H1C7', 'H2C7', 'H1C4', 'H2C4']
    atom_types = ['jxm_003', 'jxm_003', 'jxm_003', 'jxm_004', 'jxm_003', 'jxm_004', 'jxm_002', 'jxm_002', 'jxm_004', 'jxm_002', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004', 'jxm_004']
    bonds = [(0, 1), (0,2), (0,3), (1,4), (1, 5), (2, 6), (2, 7), (2, 8), (4, 7), (4,9), (4,10), (6,11), (6,12), (6,9), (9, 15), (9, 16), (7, 13), (7,14)]
    angles, dihs = DeduceTopolgy(bonds)
    head = atoms.index('C6')
    tail = atoms.index('C1') 
    _ResidueTypes[name] = residue_type(name, atoms, atom_types, bonds, angles, dihs, head, tail)

    return _ResidueTypes


def DefineResidueSeq():
    '''
    define user residue sequence and number here
    '''
    #  resname = ["ETH", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETM", "NB", "ETT"]
    #  nres  =   [1, 1, 1, 1, 1, 1, 1, 1, 9, 1, 9, 1, 9, 1, 9, 1, 9,1, 1,1,1,1,1,1, 1]
    #  resname = ['ETH', 'NB', 'ETT']
    #  nres = [1, 1, 1]

    try:
        info = sys.argv[1]
        with open(info, 'r') as fp:
            resname = []
            nres = []
            for line in fp:
                sp = line.split()
                if len(sp) == 2:
                    resname.append(sp[0])
                    nres.append(int(sp[1]))
    except:
        print("ERROR! Residue Input Info Is Wrong!")
        print("PLEASE ENSURE THE RESIDUE INPUT FILE IS AVILABLE AND THE FORMAT IS CORRENT")
        print("EXAMPLE:")
        print("  ETH   1")
        print("  ETM   1")
        print("  NB    1")
        print("  ETT   1")
        sys.exit()

    _AllResname = [[resname[i]] * nres[i] for i in range(len(nres))]
    _AllResname = [item for sub in _AllResname for item in sub]
    
    _natoms = np.array([ResidueTypes[_AllResname[idx]].natoms for idx in range(len(_AllResname))]).astype(np.int).sum()

    # dump info
    print("SYSTEM INFO: >>>>>>>>")
    print("    Total Residues: %d "%(len(_AllResname)))
    print("    Total Residues Types: %d "%(len(set(_AllResname))))
    print("    Total Atoms: %d "%(_natoms))

    return _AllResname

def BuildTopology():
    '''
    build topology based on 
    1. Force field
    2. Residue type
    3. User input residue sequence info
    '''

    from datetime import datetime
    # make top - headers 
    o = open("topol.top", 'w')
    o.write("\n")
    o.write("; GENERATED at  %s\n"%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    o.write("; Author: Jia Xiang Meng @ Jilin University\n")
    o.write(";\n")
    o.write("\n")
    o.write("[ defaults  ]\n")
    o.write("; nbfunc    comb-rule   gen-pairs   fudgeLJ fudgeQQ\n")
    o.write("1       3       yes     0.5 0.5\n")
    o.write(";\n")

    # make top - [ atomtypes ]
    o.write("\n")
    o.write("[ atomtypes ] \n")
    for k, v in F.atomtypes.name.items():
        name = k
        bond_type = v['bond_type']
        mass      = v['mass']
        charge    = v['charge']
        sigma     = v['sigma']
        epsilon   = v['epsilon']
        o.write("  %s  %s  %s  %s  %s  %s  %s\n"%(name, bond_type, mass, charge, 'A', sigma, epsilon))

    # make top - [ moleculetype ]
    o.write("\n")
    o.write("\n")
    o.write("[ moleculetype ]\n")
    o.write("; Name               nrexcl\n")
    o.write("NB                   3\n")

    # write [ atoms ]
    o.write("[ atoms ]\n")
    o.write(";   nr       type  resnr residue  atom   cgnr     charge       mass\n")
    counter = 0
    for i, res in enumerate(AllResname):
        resnr = i+1
        R = ResidueTypes[res]
        residue = res 
        natoms = R.natoms
        o.write("; residue  %d %s\n"%(resnr, res))
        
        for j in range(natoms):
            nr =  counter + 1
            type = R.atom_types[j]
            ff = F.atomtypes.name[type]
            mass = ff['mass']
            charge = ff['charge']
            atom = R.atoms[j]
            cgnr = i+1
            o.write("    %d  %s  %d  %s  %s  %d  %s  %s\n"%(nr, type, resnr, residue, atom, cgnr, charge, mass))

            bond_type = ff['bond_type']

            counter += 1

    # write [ bonds ]
    o.write("\n")
    o.write("\n")
    o.write("[ bonds ]\n")
    o.write(";  ai    aj funct            c0            c1            c2            c3\n")
    counter = 0
    former_tail = None
    for i, res in enumerate(AllResname):
        resnr = i+1
        R = ResidueTypes[res]
        bonds = R.bonds
        R.resnr = resnr
        o.write("; residue  %d %s\n"%(resnr, res))
        for bond in bonds:
            ai = bond[0]
            aj = bond[1]
            ai_type = R.atom_types[ai]
            aj_type = R.atom_types[aj]
            ai_bond_type = F.atomtypes.name[ai_type]['bond_type']
            aj_bond_type = F.atomtypes.name[aj_type]['bond_type']
            bond_type = ai_bond_type + '-' + aj_bond_type
            r_bond_type = aj_bond_type + '-' + ai_bond_type

            # write bond
            aii = ai + counter + 1
            ajj = aj + counter + 1
            if bond_type in F.bondtypes.name:
                func = F.bondtypes.name[bond_type]['func']
                parameter = F.bondtypes.name[bond_type]['parameter']
                o.write("   %d   %d  %d  %s\n"%(aii, ajj, func, parameter))
            elif r_bond_type in F.bondtypes.name:
                func = F.bondtypes.name[r_bond_type]['func']
                parameter = F.bondtypes.name[r_bond_type]['parameter']
                o.write("   %d   %d  %d  %s\n"%(aii, ajj, func, parameter))
            else:
                print("ERROR! %s (or %s)not found in force field"%(bond_type, r_bond_type))
                sys.exit()
        

        # add inter-residue linking bonds
        if R.head != None:
            ai = former_R.tail 
            aj = R.head
            ai_type = R.atom_types[ai]
            aj_type = R.atom_types[aj]
            ai_bond_type = F.atomtypes.name[ai_type]['bond_type']
            aj_bond_type = F.atomtypes.name[aj_type]['bond_type']
            bond_type = ai_bond_type + '-' + aj_bond_type
            r_bond_type = aj_bond_type + '-' + ai_bond_type

            # write bond
            aii = ai + former_R.counter + 1
            ajj = aj + counter + 1
            if bond_type in F.bondtypes.name:
                func = F.bondtypes.name[bond_type]['func']
                parameter = F.bondtypes.name[bond_type]['parameter']
                o.write("   %d   %d  %d  %s  "%(aii, ajj, func, parameter))
            elif r_bond_type in F.bondtypes.name:
                func = F.bondtypes.name[r_bond_type]['func']
                parameter = F.bondtypes.name[r_bond_type]['parameter']
                o.write("   %d   %d  %d  %s"%(aii, ajj, func, parameter))
            else:
                print("ERROR! %s (or %s)not found in force field"%(bond_type, r_bond_type))
                sys.exit()
            o.write(" ; >>>  inter-residue linking bond\n")


        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write [ angles ]
    o.write("\n\n")
    o.write("[ angles ]\n")
    o.write(";  ai    aj    ak funct            c0            c1            c2            c3\n")
    counter = 0
    for i, res in enumerate(AllResname):
        R = ResidueTypes[res]
        angles = R.angles
        resnr = i+1
        o.write("; residue  %d %s\n"%(resnr, res))
        for angle in angles:
            ai = angle[0]
            aj = angle[1]
            ak = angle[2]
            ai_type = R.atom_types[ai]
            aj_type = R.atom_types[aj]
            ak_type = R.atom_types[ak]
            ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
            aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
            ak_angle_type = F.atomtypes.name[ak_type]['bond_type']
            angle_type = ai_angle_type + '-' + aj_angle_type + '-' + ak_angle_type
            r_angle_type = ak_angle_type + '-' + aj_angle_type + '-' +  ai_angle_type
            aii = ai + counter + 1
            ajj = aj + counter + 1
            akk = ak + counter + 1
            if angle_type in F.angletypes.name:
                func = F.angletypes.name[angle_type]['func']
                parameter = F.angletypes.name[angle_type]['parameter']
                o.write("   %d   %d  %d  %d  %s\n"%(aii, ajj, akk, func, parameter))
            elif r_angle_type in F.angletypes.name:
                func = F.angletypes.name[r_angle_type]['func']
                parameter = F.angletypes.name[r_angle_type]['parameter']
                o.write("   %d   %d  %d  %d  %s\n"%(aii, ajj, akk, func, parameter))
            else:
                print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                sys.exit()

        # add inter-residdue angles
        if R.head != None:
            ai = former_R.tail
            ai_type = former_R.atom_types[ai]
            ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
            ai_bond = former_R.get_linkbond(ai)
            aii = ai + former_R.counter + 1
            former_third = [_[0] if (_.tolist().index(ai))== 1 else _[1] for _ in ai_bond]
            former_third_type = [former_R.atom_types[_] for _ in former_third]
            former_third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in former_third_type]
                
            aj = R.head
            aj_type = R.atom_types[aj]
            aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
            aj_bond = R.get_linkbond(aj)
            ajj = aj + counter + 1
            third = [_[0] if (_.tolist().index(aj))== 1 else _[1] for _ in aj_bond]
            third_type = [R.atom_types[_] for _ in third]
            third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in third_type]
            
            for j, p in enumerate(former_third_angle_type):
                angle_type = p + '-' + ai_angle_type + '-' + aj_angle_type
                r_angle_type = aj_angle_type + '-' + ai_angle_type + '-' + p
                akk = former_R.counter + former_third[j] + 1
                if angle_type in F.angletypes.name:
                    func = F.angletypes.name[angle_type]['func']
                    parameter = F.angletypes.name[angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(akk, aii, ajj, func, parameter))
                elif r_angle_type in F.angletypes.name:
                    func = F.angletypes.name[r_angle_type]['func']
                    parameter = F.angletypes.name[r_angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(akk, aii, ajj, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                    sys.exit()
                o.write(" ; >>>  inter-residue linking angle\n")

            for j, p in enumerate(third_angle_type):
                angle_type = ai_angle_type + '-' + aj_angle_type + '-' + p
                r_angle_type = p + '-' + aj_angle_type + '-' + ai_angle_type
                akk = counter + third[j] + 1
                if angle_type in F.angletypes.name:
                    func = F.angletypes.name[angle_type]['func']
                    parameter = F.angletypes.name[angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(aii, ajj, akk, func, parameter))
                elif r_angle_type in F.angletypes.name:
                    func = F.angletypes.name[r_angle_type]['func']
                    parameter = F.angletypes.name[r_angle_type]['parameter']
                    o.write("   %d   %d  %d  %d  %s "%(aii, ajj, akk, func, parameter))
                else:
                    print("ERROR! %s (or %s)not found in force field"%(angle_type, r_angle_type))
                    sys.exit()
                o.write(" ; >>>  inter-residue linking angle\n")
                
        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write [ dihedrals ]
    o.write("\n\n")
    o.write("[ dihedrals ]\n")
    o.write(";  ai    aj    ak    al      funct            c0            c1            c2            c3            c4            c5\n")
    counter = 0
    for i, res in enumerate(AllResname):
        R = ResidueTypes[res]
        dihs = R.dihs
        resnr = i+1
        o.write("; residue  %d %s\n"%(resnr, res))
        for dih in dihs:
            ai = dih[0]
            aj = dih[1]
            ak = dih[2]
            al = dih[3]
            ai_type = R.atom_types[ai]
            aj_type = R.atom_types[aj]
            ak_type = R.atom_types[ak]
            al_type = R.atom_types[al]
            ai_dih_type = F.atomtypes.name[ai_type]['bond_type']
            aj_dih_type = F.atomtypes.name[aj_type]['bond_type']
            ak_dih_type = F.atomtypes.name[ak_type]['bond_type']
            al_dih_type = F.atomtypes.name[al_type]['bond_type']
            dih_type = ai_dih_type + '-' + aj_dih_type + '-' + ak_dih_type + '-' + al_dih_type
            r_dih_type = al_dih_type + '-' + ak_dih_type + '-' + aj_dih_type + '-' +  ai_dih_type 
            aii = ai + counter + 1
            ajj = aj + counter + 1
            akk = ak + counter + 1
            all = al + counter + 1
            if dih_type in F.dihedraltypes.name:
                func = F.dihedraltypes.name[dih_type]['func']
                parameter = F.dihedraltypes.name[dih_type]['parameter']
                o.write("   %d   %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
            elif r_dih_type in F.dihedraltypes.name:
                func = F.dihedraltypes.name[r_dih_type]['func']
                parameter = F.dihedraltypes.name[r_dih_type]['parameter']
                o.write("   %d %d   %d  %d  %d %s\n"%(aii, ajj, akk, all, func, parameter))
            else:
                print("ERROR! %s (or %s)not found in force field"%(dih_type, r_dih_type))
                sys.exit()
        
        if R.head != None:
            ai = former_R.tail
            ai_type = former_R.atom_types[ai]
            ai_angle_type = F.atomtypes.name[ai_type]['bond_type']
            ai_bond = former_R.get_linkbond(ai)
            aii = ai + former_R.counter + 1
            former_third = [_[0] if (_.tolist().index(ai))== 1 else _[1] for _ in ai_bond]
            former_third_type = [former_R.atom_types[_] for _ in former_third]
            former_third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in former_third_type]
                
            aj = R.head
            aj_type = R.atom_types[aj]
            aj_angle_type = F.atomtypes.name[aj_type]['bond_type']
            aj_bond = R.get_linkbond(aj)
            ajj = aj + counter + 1
            third = [_[0] if (_.tolist().index(aj))== 1 else _[1] for _ in aj_bond]
            third_type = [R.atom_types[_] for _ in third]
            third_angle_type = [F.atomtypes.name[_]['bond_type'] for _ in third_type]

            for j, p in enumerate(former_third_angle_type):
                for k, q in enumerate(third_angle_type):
                    dih_type = p + '-' + ai_angle_type + '-' + aj_angle_type + '-' + q
                    r_dih_type = q + '-' + aj_angle_type + '-' + ai_angle_type + '-' + p
                    akk = former_R.counter + former_third[j] + 1
                    all = counter + third[k] + 1
                    if dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[dih_type]['func']
                        parameter = F.dihedraltypes.name[dih_type]['parameter']
                        o.write("   %d   %d   %d  %d  %d %s"%(akk, aii, ajj, all, func, parameter))
                    elif r_dih_type in F.dihedraltypes.name:
                        func = F.dihedraltypes.name[r_dih_type]['func']
                        parameter = F.dihedraltypes.name[r_dih_type]['parameter']
                        o.write("   %d %d   %d  %d  %d %s"%(akk, aii, ajj, all, func, parameter))
                    else:
                        print("ERROR! %s (or %s)not found in force field"%(dih_type, r_dih_type))
                        sys.exit()
                    o.write(" ; >>>  inter-residue linking dihedral\n")

        R.counter = counter
        if R.tail != None:
            former_R = R

        # end
        counter += R.natoms

    # write info tails
    o.write("\n\n")
    o.write("[ system  ]\n")
    o.write("; Name\n")
    o.write("Polymer\n")
    o.write("\n")
    o.write("[ molecules  ]\n")
    o.write("; Compound        #mols\n")
    o.write("NB        1\n")

# main module
F = DefineForceField()
ResidueTypes = DefineResidueTypes()
AllResname = DefineResidueSeq()
BuildTopology()
