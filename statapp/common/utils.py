import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types. Useful for logging numpy arrays.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def add_to_log(entry, path=None, auto_add=["id", "date"]):
    """Add to the training log file the specified entry.
    By default, adds a few automatically generated fields to the entry.
    
    Args:
        entry (dict): Entry to add to the log.
        path (str): Path to the log file.
            Default: "folder_containing_statapp_package/logs/tensorflow_transformer/log.json"
        auto_add (list of str): Keys to automatically add to the entry before logging.
            Supported keys:
                "id": A number equal to 1 + the maximum id in the current log.
                "date": A string representing the current date and time.
            Default: ["id", "date"].
        add_id (bool): Whether to add "id" key to entry.
            Default: True.
    """
    # Implement default path
    if path is None:
        path = os.path.join(os.path.dirname(statapp.__name__), "logs",  "tensorflow_transformer", "log.json")
    
    # Read log file
    with open(path, "r") as file:
        log = json.load(file)
    
    # Add auto fields
    if "id" in auto_add:
        current_max_id = max([el.get("id", 0) for el in log])
        entry["id"] = current_max_id + 1
    if "date" in auto_add:
        entry["date"] = datetime.datetime.today().strftime("%Y-%m-%dT%H:%M:%S")
    
    log.append(entry)
    
    # Write log file
    with open(path, "w") as file:
        json.dump(log, file, indent=4, sort_keys=True, cls=NumpyEncoder)

