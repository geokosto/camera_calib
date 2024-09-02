from collections import defaultdict
import dataclasses
import random
import numpy as np
from deepsport_utilities.dataset import split_equally, Subset, SubsetType

@dataclasses.dataclass
class DeepSportDatasetSplitter: # pylint: disable=too-few-public-methods
    validation_pc: int = 15
    additional_keys_usage: str = "skip"
    folds: str = "ABCDEF"
    split = {
        "A": ['KS-FR-CAEN', 'KS-FR-LIMOGES', 'KS-FR-ROANNE'],
        "B": ['KS-FR-NANTES', 'KS-FR-BLOIS', 'KS-FR-FOS'],
        "C": ['KS-FR-LEMANS', 'KS-FR-MONACO', 'KS-FR-STRASBOURG'],
        "D": ['KS-FR-GRAVELINES', 'KS-FR-STCHAMOND', 'KS-FR-POITIERS'],
        "E": ['KS-FR-NANCY', 'KS-FR-BOURGEB', 'KS-FR-VICHY'],
        # F is a fictive fold for which all keys are used for training
    }
    repetitions: dict = None
    def __post_init__(self):
        self.repetitions = self.repetitions or {}

    def split_keys(self, keys, fold=0):
        fold_index = fold
        assert 0 <= fold_index <= len(self.folds)-1, "Invalid fold index"

        testing_fold = self.folds[fold_index]
        self.testing_arena_labels = set(self.split.get(testing_fold, []))
        remaining_arena_labels = [label for f in self.folds.replace(testing_fold, "") for label in self.split.get(f, [])]
        testing_keys = []
        remaining_keys = defaultdict(list)
        additional_keys = []
        additional_arena_labels = set()
        for key in keys:
            if key.arena_label in remaining_arena_labels:
                remaining_keys[key.instant_key].append(key)
            elif key.arena_label in self.testing_arena_labels:
                testing_keys.append(key)
            else:
                additional_keys.append(key)
                additional_arena_labels.add(key.arena_label)

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        total_length = len(remaining_keys)
        validation_keys, training_keys = [], []
        indices = np.zeros(total_length, dtype=np.int32) # a vector of 1s for validation keys
        indices[np.random.choice(total_length, total_length*self.validation_pc//100, replace=False)] = 1
        for i, instant_key in zip(indices, remaining_keys):
            (training_keys, validation_keys)[i].extend(remaining_keys[instant_key])

        # Restore random seed
        random.setstate(random_state)
        testing2_keys = None
        if additional_keys:
            if self.additional_keys_usage == "testing":
                testing_keys += additional_keys
                self.testing_arena_labels = self.testing_arena_labels.union(additional_arena_labels)
            elif self.additional_keys_usage == "training":
                print(f"INFO: adding {len(additional_keys)} keys to the training set")
                training_keys += additional_keys
            elif self.additional_keys_usage == "validation":
                validation_keys += additional_keys
            elif self.additional_keys_usage in ["none", "skip"]:
                pass
            elif self.additional_keys_usage == "testing2":
                testing2_keys = additional_keys
            else:
                raise ValueError("They are additional arena labels that I don't know what to do with. Please tell me the 'additional_keys_usage' argument")

        self.testing_arena_labels = list(self.testing_arena_labels)

        return training_keys, validation_keys, testing_keys, testing2_keys

    def __call__(self, dataset, fold=0):
        training_keys, validation_keys, testing_keys, testing2_keys = self.split_keys(dataset.keys, fold)
        subsets = [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset, repetitions=self.repetitions.get('validation', 1)),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset, repetitions=self.repetitions.get('testing', 1)),
        ]
        if testing2_keys:
            subsets.append(Subset(name="testing2", subset_type=SubsetType.EVAL, keys=testing2_keys, dataset=dataset, repetitions=1))

        return subsets

@dataclasses.dataclass
class ArenaLabelFoldsDatasetSplitter(DeepSportDatasetSplitter):
    folds: str = "ABCDE"
    test_fold: str = "A"
    def __post_init__(self):
        assert self.test_fold in self.split, f"Requested test_fold ({self.test_fold}) doesn't exist. Choose among {list(self.split)}."
        assert all([fold in self.split for fold in self.folds]), f"One of the selected folds ({self.folds}) don't exist. Choose among {list(self.split)}."
        self.folds = self.folds.replace(self.test_fold, "") # make sure test_fold is not used at training or validation
    def __call__(self, dataset, fold=0):
        assert 0 <= fold < len(self.folds)
        keys = list(dataset.keys.all())

        self.testing_arena_labels = self.split[self.test_fold]
        testing_keys = [k for k in keys if k.arena_label in self.testing_arena_labels]

        validation_arena_labels = self.split[self.folds[fold]]
        validation_keys = [k for k in keys if k.arena_label in validation_arena_labels]

        training_arena_labels = [arena_label for i in range(len(self.folds)) if i != fold for arena_label in self.split[self.folds[i]]]
        training_keys = [k for k in keys if k.arena_label in training_arena_labels]

        return [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset),
        ]

@dataclasses.dataclass
class OfficialFoldsDatasetSplitter(DeepSportDatasetSplitter):
    folds: str = "ABCDE"
    eval_folds: str = "DE"
    def __post_init__(self):
        assert all([fold in self.split for fold in self.eval_folds]), f"Requested evaluation folds ({self.eval_folds}) doesn't exist. Choose among {list(self.split)}."
        assert all([fold in self.split for fold in self.folds]), f"One of the selected folds ({self.folds}) don't exist. Choose among {list(self.split)}."
    def __call__(self, dataset, fold=0):
        dataset_keys = list(dataset.keys.all())
        subset_type = lambda n: SubsetType.EVAL if n in self.eval_folds else SubsetType.TRAIN
        keys = lambda n: [k for k in dataset_keys if k.arena_label in self.split[n]]
        raise NotImplementedError("Subsets order should be checked")
        return [
            Subset(name=n, subset_type=subset_type(n), keys=keys(n), dataset=dataset) for n in self.folds
        ]

def count_keys_per_arena_label(keys):
    """returns a dict of (arena_label: number of keys of that arena)"""
    bins = {}
    for key in keys:
        bins[key.arena_label] = bins.get(key.arena_label, 0) + 1
    return bins

class KFoldsArenaLabelsTestingDatasetSplitter(DeepSportDatasetSplitter):
    def __init__(self, fold_count=8, validation_pc=15, evaluation_sets_repetitions=5):
        self.fold_count = fold_count
        self.validation_pc = validation_pc
        self.evaluation_sets_repetitions = evaluation_sets_repetitions

    def __call__(self, dataset, fold=0):
        keys = list(dataset.keys.all())
        assert fold >= 0 and fold < self.fold_count

        keys_dict = count_keys_per_arena_label(keys)
        keys_lists = split_equally(keys_dict, self.fold_count)

        self.testing_arena_labels = keys_lists[fold]
        testing_keys = [k for k in keys if k.arena_label in self.testing_arena_labels]
        remaining_keys = [k for k in keys if k not in testing_keys]

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        validation_keys = random.sample(remaining_keys, len(keys)*self.validation_pc//100)

        # Restore random seed
        random.setstate(random_state)

        training_keys = [k for k in remaining_keys if k not in validation_keys]
        r = self.evaluation_sets_repetitions
        return [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset, repetitions=r),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset, repetitions=r),
        ]

def count_keys_per_game_id(keys):
    """returns a dict of (game_id: number of keys of that game)"""
    bins = {}
    for key in keys:
        bins[key.game_id] = bins.get(key.game_id, 0) + 1
    return bins

class SingleArenaDatasetSplitter(DeepSportDatasetSplitter):
    def __init__(self, specific_arena_label):
        self.specific_arena_label = specific_arena_label
        self.fold_count = 5
    def __call__(self, dataset, fold=0):
        keys = list(dataset.keys.all())
        specific_keys = [k for k in keys if k.arena_label == self.specific_arena_label]
        d = count_keys_per_game_id(specific_keys)
        s = split_equally(d, K=self.fold_count)

        testing_keys = [k for k in specific_keys if k.game_id in s[(fold+0)%self.fold_count]]
        validation_keys = [k for k in specific_keys if k.game_id in s[(fold+1)%self.fold_count]]
        training_keys = [k for k in specific_keys if k not in testing_keys and k not in validation_keys]

        return [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset, repetitions=5),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset, repetitions=5),
        ]

class TestingArenaLabelsDatasetSplitter():
    def __init__(self, testing_arena_labels, validation_pc=15, repetitions=1):
        self.testing_arena_labels = testing_arena_labels
        self.validation_pc = validation_pc
        self.repetitions = repetitions
        assert isinstance(self.testing_arena_labels, (list, tuple))

    def __call__(self, dataset, fold=0):
        testing_keys, remaining_keys = [], []
        for key in dataset.keys:
            (remaining_keys, testing_keys)[key.arena_label in self.testing_arena_labels].append(key)

        # Backup random seed
        np_random_state = np.random.get_state()
        np.random.seed(fold)

        total_length = len(remaining_keys)
        validation_keys, training_keys = [], []
        validation_indices = np.zeros(total_length, dtype=np.int32) # a vector of 1s for validation keys
        validation_indices[np.random.choice(total_length, total_length*self.validation_pc//100, replace=False)] = 1
        for i, key in zip(validation_indices, remaining_keys):
            (training_keys, validation_keys)[i].append(key)

        # Restore random seed
        np.random.set_state(np_random_state)

        subsets = [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset, repetitions=self.repetitions),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset, repetitions=self.repetitions),
        ]
        return [s for s in subsets if len(s.keys) > 0]



class TestingValidationArenaLabelsDatasetSplitter():
    def __init__(self, testing_arena_labels, validation_arena_labels):
        if validation_arena_labels is None:
            self.__class__ = TestingArenaLabelsDatasetSplitter
            self.__init__(testing_arena_labels=testing_arena_labels)
            return
        assert isinstance(testing_arena_labels, (list, tuple))
        assert isinstance(validation_arena_labels, (list, tuple))
        self.testing_arena_labels = testing_arena_labels
        self.validation_arena_labels = validation_arena_labels
        assert len(set(self.testing_arena_labels).intersection(set(self.validation_arena_labels))) == 0

    def __call__(self, dataset, fold=0):
        validation_keys, testing_keys, training_keys = [], [], []
        for key in dataset.keys:
            if key.arena_label in self.testing_arena_labels:
                testing_keys.append(key)
            elif key.arena_label in self.validation_arena_labels:
                validation_keys.append(key)
            else:
                training_keys.append(key)
        subsets = [
            Subset(name="training", subset_type=SubsetType.TRAIN, keys=training_keys, dataset=dataset),
            Subset(name="validation", subset_type=SubsetType.EVAL, keys=validation_keys, dataset=dataset, repetitions=1),
            Subset(name="testing", subset_type=SubsetType.EVAL, keys=testing_keys, dataset=dataset, repetitions=1),
        ]
        return [s for s in subsets if len(s.keys) > 0]
