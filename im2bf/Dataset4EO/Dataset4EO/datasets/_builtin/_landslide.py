import h5py
from torchdata.datapipes.map import SequenceWrapper, Mapper
from torch.utils.data import DataLoader2
import time

import Compose

A = Compose([RandomResizedCrop(),
         Pad,
        ]
        )


A.transform

class LandsLide():
    pass
    1. resources() +dowload decompress

    def _datapipe():
        dp = SequenceWrapper(range(1,3800))
        ndp = Mapper(dp, _prepare_sample)
        ndp = ndp.map(A.transform)
        return ndp


def _prepare_sample(data):
    i = data
    iname = "train/img/image_{}.h5".format(i)
    mname = "train/mask/mask_{}.h5".format(i)
    img = h5py.File(iname, 'r')['img'][()]
    mask = h5py.File(mname, 'r')['mask'][()]
    return (iname, mname, img, mask)

datas = DataLoader2(ndp.shuffle(), batch_size=4, num_workers=4, shuffle=True)

for epoch in range(20):
    t1 = time.time()
    for it in datas:
        print(it[0], it[1])
        pass
    t2 = time.time()
    print(t2-t1)

dp = landslide4sense.Landslide4Sense(datasets_dir)
data = next(iter(dp))
pdb.set_trace()
