#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess

def find_gpu(n=1, verbose=False):
    if verbose:
        print(subprocess.check_output(['nvidia-smi']))
    import pynvml as nvml
    nvml.nvmlInit()
    gpus = []
    for i in range(nvml.nvmlDeviceGetCount()):
        gpu = nvml.nvmlDeviceGetHandleByIndex(i)
        mem = nvml.nvmlDeviceGetMemoryInfo(gpu)
        # gpus.append(pynvml.nvmlDeviceGetUtilizationRates(gpu))
        gpus.append((i, mem.free))
    nvml.nvmlShutdown()
    return [x[0] for x in sorted(gpus, key=lambda x: x[1], reverse=True)][0:n]

def structuralize(name, **entries):
    '''Construct a anonymous object from a dict.
    '''
    class _struct:
        def __init__(self, **entries):
            for k, v in entries.items():
                self.__dict__[k] = _struct(**v) if isinstance(v, dict) else v
    _struct.__name__ = name
    return _struct(**entries)
