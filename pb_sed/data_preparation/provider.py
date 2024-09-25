
import math
import numpy as np
import dataclasses
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union, Optional

import lazy_dataset
from lazy_dataset.database import JsonDatabase
from paderbox.utils.random_utils import LogTruncatedNormal, Uniform
from padertorch import Configurable
from padertorch.utils import to_list
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MultiHotAlignmentEncoder
)
from pb_sed.data_preparation.mix import MixtureDataset, SuperposeEvents
from pb_sed.data_preparation.fetcher import DataFetcher
from pb_sed.data_preparation.transform import Transform
import logging


def to_repetition_list(collection, filter_empty=True):
    """given a string, a list, tuple or a repetition_dict, return a valid repetition_dict
    filter_empty discards elements with zero repetitions"""
    if isinstance(collection, str):
        # not an actual collection, single item
        return [(collection, 1)]
    elif isinstance(collection, (list, tuple)):
        # elements should have the structure (str, int) (unrolled repetiticion_dict)
        repetition_list = []
        for element in collection:
            if isinstance(element, (list, tuple)):
                assert len(element) == 2, "Only dataset, repetition tuples are allowed"
                assert isinstance(element[1], int), "Repetitions must be an integer"
                _, reps = element
                if reps > 0:
                    repetition_list.append(element)
            else:
                assert not isinstance(element, dict), "No nesting allowed, manually flatten the data structure"
                # element should be a dataset now, which is repeated once
                repetition_list.append((element, 1))
        return repetition_list
    elif isinstance(collection, dict):
        repetition_list = []
        for dataset, repetition in collection.items():
            assert isinstance(repetition, int), "No nesting allowed, manually flatten the data structure"
            assert not isinstance(dataset, (list, tuple)), "No nesting allowed, manually flatten the data structure"
            if repetition > 0:
             repetition_list.append((dataset, repetition))
        return repetition_list 
    else:
        raise ValueError(f"Unknown collection type {type(collection)}")


@dataclasses.dataclass
class DataProvider(Configurable):
    json_path: str

    audio_reader: Callable

    train_set: Dict[str, int]
    validate_set: str = None

    cached_datasets: list = None

    min_audio_length: float = 1.

    train_segmenter: float = None
    test_segmenter: float = None
    train_transform: Callable = None
    test_transform: Callable = None
    train_fetcher: Callable = None
    test_fetcher: Callable = None

    label_key: str = 'events'
    discard_labelless_train_examples: bool = True

    storage_dir: str = None
    # augmentation
    min_class_examples_per_epoch: int = 0
    scale_sampling_fn: Callable = None
    mix_interval: float = 1.5
    mix_fn: Callable = None
    rng_seed: Optional[int] = None

    def __post_init__(self):
        assert self.json_path is not None
        self.db = JsonDatabase(json_path=self.json_path)
        self.rng = np.random.RandomState(self.rng_seed)

    def get_train_set(self, filter_example_ids=None):
        return self.get_dataset(self.train_set, train=True, filter_example_ids=filter_example_ids)

    def get_validate_set(self, filter_example_ids=None):
        if self.validate_set is None:
            return None
        return self.get_dataset(self.validate_set, train=False, filter_example_ids=filter_example_ids)

    def get_dataset(self, dataset_names_or_raw_datasets, train=False, filter_example_ids=None):
        repetition_list = to_repetition_list(dataset_names_or_raw_datasets)
        ds = self.prepare_audio(repetition_list, train=train, filter_example_ids=filter_example_ids)
        ds = self.segment_transform_and_fetch(ds, train=train)
        return ds

    def prepare_audio(self,
                      repetition_list: List[Tuple[lazy_dataset.Dataset, int]],
                      train: bool=False,
                      filter_example_ids: Optional[List[str]]=None) -> lazy_dataset.Dataset:
        individual_audio_datasets = self._load_audio(
            repetition_list, train=train, filter_example_ids=filter_example_ids)
        # enforce a single output structure
        assert isinstance(individual_audio_datasets, list)
        assert isinstance(individual_audio_datasets[0], tuple)
        assert isinstance(individual_audio_datasets[0][0], lazy_dataset.Dataset)
        assert isinstance(individual_audio_datasets[0][1], int)
        
        if train and self.min_class_examples_per_epoch > 0:
            # repeat examples to meet minimum class examples per epoch
            # compute label counts on raw_datasets (no audio/data loaded)
            assert self.label_key is not None
            raw_datasets = []
            for ds, reps in repetition_list:
                raw_dataset = self.get_raw(
                    ds,
                    discard_labelless_examples=self.discard_labelless_train_examples,
                    filter_example_ids=filter_example_ids,
                )
                raw_datasets.append((raw_dataset, reps))
                    
            label_counts, labels = DataProvider._count_labels(
                raw_datasets, self.label_key)
            label_reps = DataProvider._compute_label_repetitions(
                label_counts, min_counts=self.min_class_examples_per_epoch)
            # apply label count information to datasets with loaded audio
            repetition_groups = DataProvider._build_repetition_groups(
                individual_audio_datasets, labels, label_reps)
            logging.info(f"provider num repetition_groups {len(repetition_groups)}")
        else:
            # tile/intersperse according to original repetitions
            repetition_groups = individual_audio_datasets
        # save repetition groups as datasets
        dataset = self._tile_and_intersperse(
            repetition_groups, shuffle=train)

        if train:
            dataset = self.scale_and_mix(dataset, dataset)
        logging.info(f'Total data set length: {len(dataset)}')
        return dataset

    def _load_audio(self,
                    repetition_list: List[Tuple[lazy_dataset.Dataset, int]],
                    train: bool=False,
                    filter_example_ids: Optional[List[str]]=None,
                    idx: Optional[str]=None) -> List[Tuple[lazy_dataset.Dataset, int]]:
        """Load audio from the given datasets.

        Args:
            dataset_names_or_raw_datasets: A list of dataset names or raw datasets.
            train: Whether to load the training or validation set.
            filter_example_ids: A list of example ids to filter the dataset.
            idx: String appended to the dataset name.

        Returns:
            A list of tuples of the form (dataset, repetitions).
        """
        loaded_repetition_list = []
        for dataset_handle, repetition in repetition_list:
            ds = self.get_raw(dataset_handle,
                discard_labelless_examples=(
                    train and self.discard_labelless_train_examples
                ),
                filter_example_ids=filter_example_ids,
            )
            ds = ds.map(self.audio_reader)
            cache = (
                self.cached_datasets is not None
                and isinstance(dataset_handle, str)
                and dataset_handle in self.cached_datasets
            )
            if cache:
                ds = ds.cache(lazy=False)

            if isinstance(dataset_handle, str):
                ds_name = " " + dataset_handle
            else:
                ds_name = ""
            if idx is not None:
                ds_name += f" [{idx}]"
            logging.info(f'Single data set length{ds_name}: {len(ds)}')
            loaded_repetition_list.append((ds, repetition))
        return loaded_repetition_list


    def get_raw(
            self,
            dataset_handle: Union[str, lazy_dataset.Dataset],
            discard_labelless_examples: bool=False,
            filter_example_ids: Optional[List[str]]=None,
    ) -> lazy_dataset.Dataset:
        """	Returns a dataset with the raw examples from the given datasets.

        Raw refers to examples for which no audio has been loaded yet.

        Args:
            dataset_handle: The name of a dataset or a lazy_dataset.Dataset.
            discard_labelless_examples: If True, examples without a label are discarded.
            filter_example_ids: If given, only examples with an ID in this list are returned.

        Returns:
            A lazy dataset with the raw examples.
        """
        if isinstance(dataset_handle, str):
            # simple: dataset is given by its name
            ds = self.db.get_dataset(dataset_handle)
        else:
            assert isinstance(dataset_handle, lazy_dataset.Dataset), type(dataset_handle)
            # dataset is given as lazy_dataset -> why?
            ds = dataset_handle
        ds = self._apply_filters(ds,
                                discard_labelless_examples=discard_labelless_examples,
                                filter_example_ids=filter_example_ids)
        return ds
    
    def _apply_filters(self,
                        ds: lazy_dataset.Dataset,
                        discard_labelless_examples: bool=True,
                        filter_example_ids: Optional[List[str]]=None) -> lazy_dataset.Dataset:
        # filtering: discard labelless examples and/or filter by example ID and by audio length
        if discard_labelless_examples:
            ds = ds.filter(
                lambda ex: self.label_key in ex and ex[self.label_key],
                lazy=False
            )
        if filter_example_ids is not None:
            ds = ds.filter(
                lambda ex: ex['example_id'] not in filter_example_ids, lazy=False
            )
        return ds.filter(
            lambda ex: 'audio_length' in ex and ex['audio_length'] > self.min_audio_length, lazy=False
        )

    def _tile_and_intersperse(self, datasets: List[Tuple[lazy_dataset.Dataset, int]],
                              shuffle: bool=False) -> lazy_dataset.Dataset:
        """ Tiles and intersperses datasets.

        Args:
            datasets: A list of datasets and their repetition factors.
            shuffle: If True, the datasets are shuffled before interspersing.

        Returns:
            A lazy dataset with the tiled and interspersed examples.
        """
        if shuffle:
            datasets = [
                (ds.shuffle(reshuffle=True, rng=self.rng), reps) for ds, reps in datasets
            ]
        return lazy_dataset.intersperse(
            *[ds.tile(reps) for ds, reps in datasets]
        )

    def scale_and_mix(self,
                      dataset: lazy_dataset.Dataset,
                      mixin_dataset: Optional[lazy_dataset.Dataset]=None) -> lazy_dataset.Dataset:
        """ Scale audio data and mix datasets.

        Args:
            dataset: The dataset to scale and mix.
            mixin_dataset: The dataset to mix with. If None, the dataset is
                mixed with itself.
        
        Returns:
            The scaled and mixed dataset.
        """
        if mixin_dataset is None:
            mixin_dataset = dataset
        if self.scale_sampling_fn is not None:
            def scale(example):
                w = self.scale_sampling_fn()
                example['audio_data'] = example['audio_data'] * w
                return example
            dataset = dataset.map(scale)
            mixin_dataset = mixin_dataset.map(scale)

        # mix audio, combine examples -> effect of mixing with itself? deterministic shuffle?
        if self.mix_interval is not None:
            assert self.mix_fn is not None
            dataset = MixtureDataset(
                dataset, mixin_dataset,
                mix_interval=self.mix_interval,
                mix_fn=self.mix_fn
            )
        return dataset

    @staticmethod
    def _count_labels(raw_datasets: List[Tuple[lazy_dataset.Dataset, int]],
                      label_key,
                      reps=1) -> Tuple[Dict[str, int], List[List[List[str]]]]:
        """Count the labels in a dataset.

        Args:
            raw_datasets: A list of tuples containing a dataset
            and its number of repetitions.
            label_key: The key of the label in the dataset.
            reps: The number of repetitions of the dataset.

        Returns:
            A tuple of the updated label counts and a list with entries per dataset, each a list with all labels for each example.
        """
        label_counts = defaultdict(lambda: 0)
        labels = []
        for ds, ds_reps in raw_datasets:
            cur_labels = []
            for example in ds:
                ex_labels = sorted(set(to_list(example[label_key])))
                cur_labels.append(ex_labels)
                for label in ex_labels:
                    label_counts[label] += ds_reps * reps
            labels.append(cur_labels)
        return label_counts, labels

    @staticmethod
    def _compute_label_repetitions(label_counts: Dict[str, int],
    min_counts: Union[int, float]) -> Dict[str, int]:
        """Compute label repetitions for a dataset.

        Args:
            label_counts: A dictionary mapping labels to their counts.
            min_counts: The minimum number of repetitions for each label.
                If a float in (0, 1), it is interpreted as a fraction of the
                maximum label count.
        
        Returns:
            A dictionary mapping labels to their repetitions.
        """
        max_count = max(label_counts.values())
        if isinstance(min_counts, float):
            assert 0. < min_counts < 1., min_counts
            min_counts = math.ceil(max_count * min_counts)
        assert isinstance(min_counts, int) and min_counts > 1, min_counts
        assert min_counts - 1 <= 0.9 * max_count, (f"The minimum number of label repetitions should be "
                                                  f"less than 90 percent of the dataset length {(min_counts, max_count)}")
        
        base_rep = 1 // (1 - (min_counts-1)/max_count)
        min_counts *= base_rep
        label_repetitions = {
            label: math.ceil(min_counts / count)
            for label, count in label_counts.items()
        }
        return label_repetitions

    @staticmethod
    def _build_repetition_groups(
    dataset: List[Tuple[lazy_dataset.Dataset, int]],
    labels: List[List[List[str]]],
    label_repetitions) -> List[Tuple[lazy_dataset.Dataset, int]]:
        """	Build repetition groups for a dataset.

        Args:
            dataset: A dataset or a list of datasets.
            labels: A list of lists of labels.
            label_repetitions: A dict mapping labels to repetitions.
        
        Returns:
            A list of (dataset, repetitions) pairs.
        """
        # new datasets of examples with same repetitions
        datasets = []
        for dataset_repetitions, ds_labels in zip(dataset, labels):
            ds, ds_reps = dataset_repetitions
            # maximum of label repetitions (labels-to-be-repeated) per example
            idx_reps = [
                max([label_repetitions[label] for label in idx_labels])
                for idx_labels in ds_labels
            ]
            rep_groups = {}
            # group by number of repetitions
            for n_reps in set(idx_reps):
                rep_groups[n_reps] = np.argwhere(
                    np.array(idx_reps) == n_reps
                ).flatten().tolist()
            # new datasets of examples with same repetitions
            datasets = []
            for n_reps, indices in sorted(
                    rep_groups.items(), key=lambda x: x[0]
            ):
                datasets.append((ds[sorted(indices)], n_reps*ds_reps))
        return datasets

    def segment_transform_and_fetch(
            self, dataset, segment=True, transform=True, fetch=True,
            train=False,
    ):
        segmenter = self.train_segmenter if train else self.test_segmenter
        segment = segment and segmenter is not None
        if segment:
            dataset = dataset.map(segmenter)
        if transform:
            transform = self.train_transform if train else self.test_transform
            assert transform is not None
            if segment:
                dataset = dataset.batch_map(transform)
            else:
                dataset = dataset.map(transform)
        if fetch:
            fetcher = self.train_fetcher if train else self.test_fetcher
            assert fetcher is not None
            dataset = fetcher(dataset, batched_input=segment)
        return dataset

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['audio_reader'] = {
            'factory': AudioReader,
            'source_sample_rate': None,
            'target_sample_rate': 16000,
            'average_channels': True,
            'normalization_domain': 'instance',
            'normalization_type': 'max',
            'alignment_keys': ['events'],
        }
        config['train_transform'] = {
            'factory': Transform,
            'stft': {
                'factory': STFT,
                'shift': 320,
                'window_length': 960,
                'size': 1024,
                'fading': 'half',
                'pad': True,
                'alignment_keys': ['events'],
            },
            'label_encoder': {
                'factory': MultiHotAlignmentEncoder,
                'label_key': 'events',
                'storage_dir': config['storage_dir'],
            },
            'anchor_sampling_fn': {
                'factory': Uniform,
                'low': 0.4,
                'high': 0.6,
            },
            'anchor_shift_sampling_fn': {
                'factory': Uniform,
                'low': -0.1,
                'high': 0.1,
            },
        }
        config['test_transform'] = {
            'factory': Transform,
            'stft': config['train_transform']['stft'],
            'label_encoder': config['train_transform']['label_encoder'],
            'provide_boundary_targets': config['train_transform']['provide_boundary_targets'],
            'provide_strong_targets': config['train_transform']['provide_strong_targets'],
        }
        config['train_fetcher'] = {
            'factory': DataFetcher,
            'prefetch_workers': 16,
            'batch_size': 16,
            'max_padding_rate': .05,
            'max_bucket_buffer_size': 2000,
            'drop_incomplete': True,
            'global_shuffle': False,  # already shuffled in prepare_audio
        }
        config['test_fetcher'] = {
            'factory': DataFetcher,
            'prefetch_workers': config['train_fetcher']['prefetch_workers'],
            'batch_size': 2 * config['train_fetcher']['batch_size'],
            'max_padding_rate': config['train_fetcher']['max_padding_rate'],
            'bucket_expiration': config['train_fetcher']['bucket_expiration'],
            'max_bucket_buffer_size': config['train_fetcher']['max_bucket_buffer_size'],
            'drop_incomplete': False,
            'global_shuffle': False,
        }
        config['scale_sampling_fn'] = {
            'factory': LogTruncatedNormal,
            'loc': 0.,
            'scale': 1.,
            'truncation': np.log(3.),
        }
        if config['mix_interval'] is not None:
            config['mix_fn'] = {
                'factory': SuperposeEvents,
            }
            if config['mix_fn']['factory'] == SuperposeEvents:
                config['mix_fn'].update({
                    'min_overlap': 1.,
                    'fade_length': config['train_transform']['stft']['window_length'],
                    'label_key': 'events',
                })
