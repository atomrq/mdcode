import numpy as np

def Write2Xml(filename, S):
    o = open(filename, 'w')
    timestep, dimensions, natoms = S.configure
    # 'write headers'
    o.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    o.write('<hoomd_xml version="1.6">\n')
    o.write('<configuration time_step="%d" dimensions="%d" natoms="%d">\n'%(timestep, dimensions, natoms))
    o.write('<box lx="%f" ly="%f" lz="%f" xy="0" xz="0" yz="0"/>\n'%(S.box[0], S.box[1], S.box[2]))

    # 'write position'
    if hasattr(S, "Position"):
        o.write('<position num="%d">\n'%(S.Position.shape[0]))
        for p in S.Position:
            o.write('%-18.11f%-18.11f%-18.11f\n'%(p[0], p[1], p[2]))
        o.write('</position>\n')
    else:
        pass

    # 'write image'
    if hasattr(S, "Image"):
        o.write('<image num="%d">\n'%(S.Image.shape[0]))
        for p in S.Image:
            o.write('%d\t%d\t%d\n'%(int(p[0]), int(p[1]), int(p[2])))
        o.write('</image>\n')
    else:
        pass

    # 'write type'
    if hasattr(S, "Type"):
        o.write('<type num="%d">\n'%(S.Type.shape[0]))
        for p in S.Type:
            #  o.write('%s\n'%(chr(int(p))))
            o.write('%s\n'%(p))
        o.write('</type>\n')
    else:
        pass

    # 'write body'
    if hasattr(S, "Body") and S.Body is not None:
        o.write('<body num="%d">\n'%(S.Body.shape[0]))
        for p in S.Body:
            o.write("%d\n"%(p))
        o.write('</body>\n')
    else:
        pass

    # 'write mass'
    if hasattr(S, "Mass") and S.Mass != None:
        o.write('<mass num="%d">\n'%(S.Mass.shape[0]))
        for p in S.Mass:
            o.write("%d\n"%(p))
        o.write('</mass>\n')
    else:
        pass
    # 'write bonds'
    if hasattr(S, "Bond"):
        o.write('<bond num="%d">\n'%(len(S.Bond)))
        if len(S.Bond) == 3:
            for p in S.Bond:
                o.write('%s\t%s\t%s\n'%(p[0], p[1], p[2]))
        elif S.Bond.shape[1] == 2:
            for j, p in enumerate(S.Bond):
                bond_name = chr(S.Type[p[0]]) + '-' + chr(S.Type[p[1]]) 
                o.write('%s\t%s\t%s\n'%(bond_name, p[0], p[1]))
        o.write('</bond>\n')
    else:
        pass
    # write angles
    if hasattr(S, "Angle") and S.Angle is not None:
        o.write('<angle num="%d">\n'%(S.Angle.shape[0]))
        if S.Angle.shape[1] == 4:
            for p in S.Angle:
                o.write('%s\t%s\t%s\t%s\n'%(p[0], p[1], p[2], p[3]))
        elif S.Angle.shape[1] == 3:
            for p in S.Angle:
                angle_name = 'polymer' 
                o.write('%s\t%s\t%s\t%s\n'%(angle_name, p[0], p[1], p[2]))
        # for p in S.Angle:
            # a, b, c = S.Type[p[0]], S.Type[p[1]], S.Type[p[2]]
            # angle_name = chr(a) + chr(b) + chr(c)
            # # o.write("%s %s %s %s\n"%(angle_name, p[0], p[1], p[2]))
            # o.write("%s %s %s %s\n"%('polymer', p[0], p[1], p[2]))
        o.write('</angle>\n')
    else:
        pass

    # 'write tails'
    o.write('</configuration>\n')
    o.write('</hoomd_xml>\n')

# write atoms info to .gro file.
def Write2gro(SS):
    '''
    note1. needed attributes: box, atp, atp_id, rtp, rtp_id, x, y, z
    note2. output file name: struc.gro 
    note3. For gromacs can not directly use pdb/gro file with 100000 atoms,
            so the atom_id in gro file is set a constant 100, and gromacs 
            will make the atom id itself. However, this kind of setting will 
            cause errors in visualization with vmd for wrong atom-id numbers.
    '''
    box = SS.box
    ofile = open("struc.gro",'w')
    ofile.write("NPC t=   0.00000\n")
    ofile.write("  %d\n"%(len(SS.pos)))
    #'write atoms  to gro file.'
    for i in range(len(SS.pos)):
        atp = SS.atp[i].strip() 
        atp_id = SS.atp_id[i] + 1
        rtp = SS.rtp[i].strip()
        rtp_id = SS.rtp_id[i] + 1
        x, y, z = SS.pos[i] 
        ofile.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"%(rtp_id, rtp, atp, rtp_id, x/10.0, y/10.0, z/10.0))
    ofile.write("  %8.5f %8.5f  %8.5f\n"%(box[0]/10.0, box[1]/10.0, box[2]/10.0))
