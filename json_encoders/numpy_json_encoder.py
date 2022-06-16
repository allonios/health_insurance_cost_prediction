from json import JSONEncoder
import numpy as np
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float):
            return float(obj)
        if isinstance(obj, np.int):
            return int(obj)
        if isinstance(obj, range):
            return list(obj)
        if np.isnan(obj):
            return 0
        
        
        return JSONEncoder.default(self, obj)