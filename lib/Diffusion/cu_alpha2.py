import gsd.hoomd
import numpy as np
from numba import cuda, float32
import math
import sys
from pytool.cython import cfun
from pytool import msd_fft
import time


def msd_numpy(_a):
    r"""MSD
    :param a:  positions of a np.ndarray, (n_f, n_p, n_d)
    :return msd
    """

    return np.asarray([((_a[inv:] - _a[:-inv])**2).sum(axis=-1).mean() if inv != 0 else 0.0 for inv in range(_a.shape[0])])


# define a device function
@cuda.jit('float32(float32, float32, float32, float32, float32, float32)', device=True, inline=True)
def cu_device_fn(ax, ay, az, bx, by, bz):
    return (ax - bx)**2 + (ay - by)**2 + (az - bz)**2  

@cuda.jit('void(float32[:, :, :], float32[:, :], int32)')
def cu_timepair(_a, _res, _N):
    i = cuda.grid(1)
    if i >= _N:
        return
    px = _a[i][:, 0]
    py = _a[i][:, 1]
    pz = _a[i][:, 2]

    counter = 0
    for j in range(1, px.shape[0]):
        for k in range(j, px.shape[0]):
            dx = px[k - j] - px[k]
            dy = py[k - j] - py[k]
            dz = pz[k - j] - pz[k]
            r2 = dx**2 + dy**2 + dz**2 #msd
            _res[i][counter] = r2 
            counter += 1
    
def _call(a, gpu=0):

    nt, na, nd = a.shape
    
    atoms = np.ascontiguousarray(a.swapaxes(0, 1).astype(np.float32))

    cuda.select_device(gpu)
    device = cuda.get_current_device()

    #  tpb = device.WARP_SIZE
    tpb = 512 
    bpg = math.ceil(na / tpb)

    MEMORY = cuda.current_context().get_memory_info()[0]
    ndata = np.arange(1, nt).sum()
    trunksize = np.ceil(MEMORY * 0.9  / ndata / 4).astype(np.int)

    res = cuda.device_array((trunksize, ndata), dtype=np.float32) 
    atoms_trunk = cuda.device_array((trunksize, nt, 3), dtype=np.float32)

    nloops = np.ceil(na / trunksize).astype(np.int)

    _rr = []

    for i in range(nloops):

        start = trunksize * i
        if i == nloops - 1:
            itrunk = atoms[start:]
            print("Trunk-%d: %d -- natoms: %d/%d"%(i, itrunk.shape[0], na, na))
        else:
            end = trunksize * (i + 1)
            itrunk = atoms[start:end] 
            print("Trunk-%d: %d -- natoms: %d/%d"%(i, itrunk.shape[0], (i + 1) * trunksize, na))

        natoms_itrunk = itrunk.shape[0]
        subarray = np.zeros((trunksize, nt, 3), dtype=np.float32)
        subarray[:natoms_itrunk] = itrunk
        atoms_trunk.copy_to_device(subarray.astype(np.float32))
        cu_timepair[bpg, tpb](atoms_trunk, res, natoms_itrunk)
        cuda.synchronize()
        _r = res.copy_to_host()[:natoms_itrunk]

        # post analyze
        #  _rr.append(cu_msd(_r, nt))
        _rr.append(cu_alpha2(_r, nt))
    _rr = np.concatenate(_rr, axis=1).mean(axis=-1)

    return _rr 

def cu_alpha2(_x, _nt):
    index = np.cumsum(np.arange(1, _nt)[::-1])

    a2 = [np.zeros(_x.shape[0])]
    for i in range(index.shape[0]):
        if i == 0:
            r2 = _x[:, :index[0]].mean(axis=-1)
            r4 = (_x[:, :index[0]]**2).mean(axis=-1)
            ia2 = 3.0 / 5.0 * r4 / r2**2 - 1
            a2.append(ia2)
        else:
            start = index[i - 1]
            end = index[i] 
            r2 = _x[:, start:end].mean(axis=-1)
            r4 = (_x[:, start:end]**2).mean(axis=-1)
            ia2 = 3.0 / 5.0 * r4 / r2**2 - 1
            a2.append(ia2)
    a2 = np.asarray(a2, dtype=np.float32)

    return a2

def cu_msd(_x, _nt):
    """
    : param _x (na, nttt)
    """

    _msd = [np.zeros(_x.shape[0])]
    index = np.cumsum(np.arange(1, _nt)[::-1])
    
    for i in range(index.shape[0]):
        if i == 0:
            r = _x[:, :index[0]].mean(axis=-1)
            _msd.append(r)
        else:
            start = index[i - 1]
            end = index[i] 
            r = _x[:, start:end].mean(axis=-1)
            _msd.append(r)
    _msd = np.asarray(_msd).astype(np.float32)

    return _msd 


def main():
    traj = gsd.hoomd.open(sys.argv[1], mode='rb')
    time_scale = int(traj[1].configuration.step - traj[0].configuration.step) * 0.001 # dt = 0.001
    nfrms = len(traj)
    box = traj[0].configuration.box[:3]

    pos = np.asarray([_.particles.position + box * _.particles.image for _ in traj])
    print(pos.shape)

    result = _call(pos, 2)
    #  mm = np.asarray([msd_fft(_) for _ in pos.swapaxes(0, 1)]).mean(axis=0).astype(np.float32)
    #  np.savetxt('msd.dat', np.c_[np.arange(nfrms), result, mm], fmt='%.6f')


# main
main()
