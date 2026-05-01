# noise2image.h5 — Prophesee HDF5 event reader / writer, as used by the
# upstream Noise2Image codebase.  Included verbatim for parity with
# `noise2image/utils.py`, which imports `H5EventsReader` from here.
# The Noise2Params public dataset ships only .parquet recordings, so
# this module is inactive on the published pipeline.  See the licence
# header below for the upstream (Prophesee) terms.
# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# pylint: disable=W0611
#
# with modifications made by Ruiming Cao and Dekel Galor
"""
h5 io for event storage
you can use 2 compression backends:
    - zlib (fast read, slow write)
    - zstandard (fast read, fast write, but you have to install it)
"""

import warnings
import h5py
try:
    import zstandard
except BaseException:
    pass
import numpy as np
from zlib_ng import zlib_ng


class H5EventsWriter(object):
    """
    Compresses & Writes Event Packets as they are read

    Args:
        out_name (str): destination path
        height (int): height of recording
        width (int): width of recording
        compression_backend (str): compression api to be called, defaults to zlib.
        If you can try to use zstandard which is faster at writing.
    """

    def __init__(self, out_name, height, width, compression_backend="zlib"):
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(out_name, "w")
        self.dataset_size_increment = 1000
        shape = (self.dataset_size_increment,)
        self.dset = self.f.create_dataset("event_buffers", shape, maxshape=(None,), dtype=dt)
        self.ts = self.f.create_dataset("event_buffers_start_times", shape, maxshape=(None,), dtype=dt2)

        if compression_backend != "zlib":
            warnings.warn(f"{compression_backend} is not supported yet, using zlib-ng instead")
        self.dset.attrs["compression_backend"] = 'zlib'
        self.dset.attrs["height"] = height
        self.dset.attrs["width"] = width

        self.index = 0
        self.is_close = False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def write(self, events):
        """
        Writes event buffer into a compressed packet

        Args:
            events (ndarray): events of type EventCD, which is defined  as {'names': ['x', 'y', 'p', 't'],
            'formats': ['<u2', '<u2', '<i2', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}.
        """
        if not len(events):
            return
        if self.index >= len(self.dset):
            new_len = self.dset.shape[0] + self.dataset_size_increment
            self.dset.resize((new_len,))
            self.ts.resize((new_len,))
        self.ts[self.index] = events['t'][0]

        zipped_data = zlib_ng.compress(events)
        zipped_data = np.frombuffer(zipped_data, dtype="uint8")

        self.dset[self.index] = zipped_data
        self.index += 1

    def close(self):
        if not self.is_close:
            self.dset.resize((self.index,))
            self.ts.resize((self.index,))
            self.f.close()
            self.is_close = True

    def __del__(self):
        self.close()


class H5EventsReader(object):
    """
    Reads & Seeks into a h5 file of compressed event packets.

    Args:
        src_name (str): input path
    """

    def __init__(self, path):
        self.path = path
        dt = h5py.vlen_dtype(np.dtype("uint8"))
        dt2 = np.int64
        self.f = h5py.File(path, "r")
        self.len = len(self.f["event_buffers"])
        self.height = self.f["event_buffers"].attrs["height"]
        self.width = self.f["event_buffers"].attrs["width"]
        self.start_times = self.f['event_buffers_start_times'][...]
        self.start_index = 0
        self.sub_start_index = 0

    def __len__(self):
        return len(self.start_times)

    def seek_in_buffers(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        return idx

    def seek_time(self, ts):
        idx = np.searchsorted(self.start_times, ts, side='left')
        self.start_index = max(0, idx - 1)
        events = self.load_buffer(self.start_index)
        if self.start_index > 0:
            assert events['t'][0] <= ts
        self.sub_start_index = np.searchsorted(events["t"], ts)
        self.sub_start_index = max(0, self.sub_start_index)

    def load_buffer(self, idx):
        zipped_data = self.f["event_buffers"][idx]

        unzipped_data = zlib_ng.decompress(zipped_data.data)
        events = np.frombuffer(unzipped_data, dtype={'names': ['x', 'y', 'p', 't'],
                                                     'formats': ['<u2', '<u2', '<i2', '<i8'],
                                                     'offsets': [0, 2, 4, 8], 'itemsize': 16})
        return events

    def read_interval(self, time_start_us, time_end_us):
        assert time_start_us < time_end_us, "time_start_us must be strictly smaller than time_end_us"

        idx_start = max(0, np.searchsorted(self.start_times, time_start_us, side='left') - 1)
        idx_end = np.searchsorted(self.start_times, time_end_us, side='right')

        if idx_end == 0:
            return np.array([], dtype={'names': ['x', 'y', 'p', 't'],
                                        'formats': ['<u2', '<u2', '<i2', '<i8'],
                                        'offsets': [0, 2, 4, 8], 'itemsize': 16})

        list_events = []
        for i in range(idx_start, idx_end):
            list_events.append(self.load_buffer(i))

        events = np.concatenate(list_events)

        event_idx_start = np.searchsorted(events["t"], time_start_us, side='left')
        event_idx_end = np.searchsorted(events["t"], time_end_us, side='left')
        return events[event_idx_start:event_idx_end]

    def get_size(self):
        return (self.height, self.width)

    def __iter__(self):
        for i in range(self.start_index, len(self.f["event_buffers"])):
            events = self.load_buffer(i)
            if i == self.start_index and self.sub_start_index > 0:
                events = events[self.sub_start_index:]
            yield events
