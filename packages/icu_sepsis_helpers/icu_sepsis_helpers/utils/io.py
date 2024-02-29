import json
import logging
import pickle
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path


class PersistentDict(MutableMapping):

    DATETIME_FORMAT = '%Y-%m-%d_%H-%M-%S'
    
    def __init__(self, *levels, dts:str=None, data:dict=None):
        assert len(levels) > 0, 'Must provide at least one level'
        
        dts = datetime.now().strftime(self.DATETIME_FORMAT) if dts is None else dts
        self._data = {} if data is None else data

        self._out_dir = Path(*levels, dts)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def _load(self, path:Path):
        # check extension of the path
        if path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f'Unknown file extension `{path.suffix}`')
        
    def _save(self, key, value)->Path:
        try:
            json_path = Path.joinpath(self._out_dir, f'{key}.json')
            data = json.dumps(value, indent=4)
            with open(json_path, 'w') as f:
                f.write(data)
            return json_path
        except TypeError:
            logging.warning(f'Could not save `{key}` as json, saving as pickle instead')
            pkl_path = Path.joinpath(self._out_dir, f'{key}.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(value, f)
            return pkl_path
        
    def _del(self, path:Path, missing_ok:bool=False):
        path.unlink(missing_ok=missing_ok)

    def __getitem__(self, key):
        return self._load(self._data[key])
    
    def __setitem__(self, key, value):
        self._data[key] = self._save(key, value)

    def __delitem__(self, key):
        self._del(self._data[key])
        del self._data[key]

    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __repr__(self):
        return f'PersistentDict({self._out_dir})'
    
    def __str__(self):
        return repr(self)

    @classmethod
    def load(cls, parent_path:Path, dts:str=None):
        if dts is None:
            dts = datetime.strftime(
                max(
                    [
                        datetime.strptime(p.stem, cls.DATETIME_FORMAT) 
                            for p in parent_path.iterdir() if p.is_dir()
                    ]
                ),
                cls.DATETIME_FORMAT)
        path = Path.joinpath(parent_path, dts)
        data = {}
        for p in path.iterdir():
            data[p.stem] = p
        obj = cls(parent_path, dts=dts, data=data)
        return obj
    
    @property
    def out_dir(self):
        return self._out_dir

