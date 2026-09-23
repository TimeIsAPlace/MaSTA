# Copyright (c) 2023 Amphion.
# Licensed under the MIT license in the repository root.
"""Configuration subset used by the released MaskGCT training pipeline."""
from pathlib import Path
import json5


class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = JsonHParams(**v) if isinstance(v, dict) else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return repr(self.__dict__)


def load_config(config_fn, lowercase=False):
    with Path(config_fn).open(encoding='utf-8') as source:
        config = json5.load(source)
    if 'base_config' in config:
        raise ValueError('Use a self-contained config for this release.')
    if lowercase:
        def lower(value):
            return {k.lower(): lower(v) for k, v in value.items()} if isinstance(value, dict) else value
        config = lower(config)
    return JsonHParams(**config)
