import gsd.hoomd
import numpy as np
from numba import cuda, float32
from math import floor, sqrt, ceil
import sys
from pytool.cython import cfun
from pytool import msd_fft
import time


@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:], float32, float32, float32[:, :])')
def cu_timepair(_a, _b, _box, _bs, _rc, res):
    row, col = cuda.grid(2)
    if row > _a.shape[0] or col > _a.shape[1]:
        return
    ix, iy, iz = _a[row, col]
    b = _b[row, :]
    for j in range(b.shape[0]):
        jx = b[j, 0]  
        jy = b[j, 1]  
        jz = b[j, 2]  
        x = ix - jx
        y = iy - jy
        z = iz - jz
        x -= _box[0] * floor(x / _box[0] + 0.5)
        y -= _box[1] * floor(y / _box[1] + 0.5)
        z -= _box[2] * floor(z / _box[2] + 0.5)
        r = sqrt(x * x + y * y + z * z)
        if 0 < r < _rc:
            idxij = floor(r / _bs)
            idxij = int(idxij)
            cuda.atomic.add(res, (row, idxij), 1)

def _call(a, b, box, bs, rc, bins):
    TPB = 32
    tpb = (TPB, TPB)
    bpg = (ceil(a.shape[0] / tpb[0]), ceil(a.shape[1] / tpb[1]))
    #  print(tpb, bpg)

    nt, na, nd = pos.shape
    ndata = np.arange(1, nt).sum()

    cuda.select_device(1)
    device = cuda.get_current_device()

    res = cuda.device_array((nt, bins), dtype=np.float32) 
    device_pos = cuda.device_array(pos.shape, dtype=np.float32)
    device_pos.copy_to_device(pos)


    cu_timepair[bpg, tpb](device_pos, device_pos, box, bs, rc, res)
    cuda.synchronize()
    
    rr = res.copy_to_host()
    print(rr.shape)

    return rr


# main
#  pos = np.random.random((4000, 12000, 3)).astype(np.float32)
traj = gsd.hoomd.open(sys.argv[1], mode='rb')
box = traj[0].configuration.box[:3]
pos = np.asarray([_.particles.position + box * _.particles.image for _ in traj], dtype=np.float32)
pos = np.random.random((10000, 12000, 3)).astype(np.float32)
bs = 0.02
rc = box[0] / 2.0
bins = ceil(rc / bs)
rho = pos.shape[1] / box[0] / box[1] / box[2]
r = np.arange(bins) * bs + bs / 2 
vol = 4.0 / 3.0 * np.pi * ((r + bs / 2)**3 - (r - bs / 2)**3)

result = _call(pos, pos, box, bs, rc, bins)
result = result.mean(axis=0) / vol / rho / pos.shape[1]
np.savetxt("rdf.dat", np.c_[r, result], fmt='%.6f')