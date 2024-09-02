import abc
import json

from mlworkflow import Dataset

def export_dataset(dataset: Dataset, prefix: str, keys=None, verbose: bool=True):
    """ Export a dataset to disk by saving
        - serialized dataset items in their original database format in a json file,
        - and a list of necessary files in a txt file.
    """
    files = []
    items = []

    keys = keys or dataset.keys

    for key in keys:
        item = dataset.query_item(key)
        files = files + list(item.files)
        items.append(item.db_item)
    filename = prefix + 'dataset.json'
    with open(filename, 'w') as fd:
        json.dump(items, fd)
    if verbose:
        print(f"Dataset index successfully created in '{filename}'")

    filename = prefix + 'files.txt'
    with open(filename, 'w') as fd:
        files = map(lambda x:x+'\n', files)
        fd.writelines(files)
    if verbose:
        print(f"Dataset file list successfully created in '{filename}'")
        print(f"You can zip them by running\n$ zip -r <filename>.zip `cat {filename}`")


def import_dataset(dataset_type: Dataset, filename: str, **dataset_config):
    """ Import a dataset exported by `export_dataset` by providing the
        - dataset Type,
        - the serialized dataset items in their original database format in a json file,
        - and the config dictionary required to build each dataset item (dataset dependant)
    """
    return ImportedDataset(filename=filename, dataset_type=dataset_type, **dataset_config)


def serialize_keys(keys):
    return list(map(tuple, keys)) # keys should already be tuples by design, but here we remove any reference to possible NamedTuple


def deserialize_keys(keys, type):
    return list(map(lambda k: type(*k), keys))


class ImportedDataset(Dataset):
    def __init__(self, filename, dataset_type, **dataset_config):
        with open(filename, "r") as fd:
            self.cache = json.load(fd)
        self._lookuptable = {}
        self.dataset_type = dataset_type
        self.dataset_config = dataset_config # I know it's a little bit ugly… but I need to move on to other things
        for name, value in dataset_config.items():
            setattr(self, name, value)
    def yield_keys(self):
        for db_item in self.cache:
            item = self.dataset_type.items_type(db_item, **self.dataset_config)
            self._lookuptable[item.key] = db_item
            yield item.key
    def query_item(self, key):
        try:
            db_item = self._lookuptable[key]
        except KeyError as e:
            if key not in list(self.keys):
                raise KeyError("Key '{}' not found. Did you call yield_keys() method?".format(key)) from e
            raise e
        return self.dataset_type.items_type(db_item, **self.dataset_config)

class GenericItem(metaclass=abc.ABCMeta):
    """ Python object describing dataset item.

        .. important::
            Attributes that require files to be downloaded (like images) should
            be decorated with `functools.cached_property` to prevent being read before they get
            downloaded.
    """
    @abc.abstractproperty
    def key(self):
        """ Generates the key associated to the Item.
            Key should to be immutable (eg: NamedTuple).
        """
        raise NotImplementedError
    @property
    def db_item(self):
        """ Returns the db_item that creates the python object
        """
        raise NotImplementedError
    @abc.abstractproperty
    def files(self):
        """ List files stored on remote storage that belong to the object
        """
        raise NotImplementedError

