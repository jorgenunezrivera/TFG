import numpy as np
import threading
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import os

def get_filenames(folder):
    return list(sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.JPEG')]))

class FromDiskGenerator(Sequence):
    # Class is a dataset wrapper for better training performance
    def __init__(
            self, x_set, batch_size=256,
    ):
        self.x = x_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x)).astype(int)
        self.memory_batch = {}
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idxs):
        if isinstance(idxs, slice):
            idxs = np.arange(0 if idxs.start is None else idxs.start, idxs.stop, 1 if idxs.step is None else idxs.step)
        for idx in idxs:
            if idx not in self.memory_batch:
                self.lock.acquire()
                memory_batch_idx = idx // self.batch_size
                self.__load_batch_data(memory_batch_idx)
                self.lock.release()
        target_size = self.memory_batch[idxs[0]].shape
        output = np.zeros(((len(idxs), ) + target_size))
        for i, idx in enumerate(idxs):
            output[i] = self.memory_batch[idx]
            del self.memory_batch[idx]
        return output

    def __load_batch_data(self, memory_batch_idx):
        inds = self.indices[
            memory_batch_idx * self.batch_size:min(len(self.x), (memory_batch_idx + 1) * self.batch_size)
        ]
        memory_batch_x = np.array([image.img_to_array(image.load_img(self.x[ind])) for ind in inds])
        memory_batch_dict = dict(zip(inds, memory_batch_x))
        self.memory_batch = {**self.memory_batch, **memory_batch_dict}