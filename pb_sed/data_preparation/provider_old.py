
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


@dataclasses.dataclass
class DataProvider(Configurable):
    json_path: str

    audio_reader: Callable

    train_set: dict
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
        # need: dataset_names_or_raw_datasets string or dictionary (list maybe?), but no nesting
        ds = self.prepare_audio(dataset_names_or_raw_datasets, train=train, filter_example_ids=filter_example_ids)
        ds = self.segment_transform_and_fetch(ds, train=train)
        return ds

    def prepare_audio(self, dataset_names_or_raw_datasets, train=False, filter_example_ids=None):
        individual_audio_datasets = self._load_audio(
            dataset_names_or_raw_datasets, train=train, filter_example_ids=filter_example_ids)
        # why this, just enforce a single output type
        if not isinstance(individual_audio_datasets, list):
            assert isinstance(individual_audio_datasets, lazy_dataset.Dataset), type(individual_audio_datasets)
            individual_audio_datasets = [(individual_audio_datasets, 1)]
        # tile and intersperse cannot handle nested datasets!!!
        combined_audio_dataset = self._tile_and_intersperse(
            individual_audio_datasets, shuffle=train)

        if train and self.min_class_examples_per_epoch > 0:
            assert self.label_key is not None
            logging.info(f"provider: {dataset_names_or_raw_datasets}")
            raw_datasets = self.get_raw(
                dataset_names_or_raw_datasets,
                discard_labelless_examples=self.discard_labelless_train_examples,
                filter_example_ids=filter_example_ids,
            )
            # repeating examples for class balancing
            label_counts, labels = self._count_labels(
                raw_datasets, self.label_key)
            label_reps = self._compute_label_repetitions(
                label_counts, min_counts=self.min_class_examples_per_epoch)
            repetition_groups = self._build_repetition_groups(
                individual_audio_datasets, labels, label_reps)
            
            logging.info(f"provider num repetition_groups {len(repetition_groups)}")
            logging.info(f"provider repetitions {[(len(ds), reps) for (ds, reps) in repetition_groups]}")
            dataset = self._tile_and_intersperse(
                repetition_groups, shuffle=train)
        else:
            dataset = combined_audio_dataset
        # 
        if train:
            # dataset = self.scale_and_mix(dataset, combined_audio_dataset)
            dataset = self.scale_and_mix(dataset, dataset)
        print(f'Total data set length:', len(dataset))
        return dataset

    def _load_audio(self,
                    dataset_names_or_raw_datasets: Union[str, List[str], Tuple[str], Dict[str, int], lazy_dataset.Dataset],
                    train=False,
                    filter_example_ids=None,
                    idx=None):
        if isinstance(dataset_names_or_raw_datasets, (dict, list, tuple)):
            ds = []
            for i, name_or_ds in enumerate(dataset_names_or_raw_datasets):
                num_reps = (
                    dataset_names_or_raw_datasets[name_or_ds]
                    if isinstance(dataset_names_or_raw_datasets, dict)
                    else name_or_ds[1] if isinstance(name_or_ds, (list, tuple))
                    else 1
                )
                if num_reps == 0:
                    continue
                ds.append((
                    self._load_audio(
                        name_or_ds[0] if isinstance(name_or_ds, (list, tuple))
                        else name_or_ds,
                        train=train, filter_example_ids=filter_example_ids, idx=i,
                    ),
                    num_reps
                ))
            # may be nested repetition object
            return ds
        # possible types for dataset_names_or_raw_datasets: str, lazy_dataset.Dataset
        ds = self.get_raw(
            dataset_names_or_raw_datasets,
            discard_labelless_examples=(
                train and self.discard_labelless_train_examples
            ),
            filter_example_ids=filter_example_ids,
        ).map(self.audio_reader)
        cache = (
            self.cached_datasets is not None
            and isinstance(dataset_names_or_raw_datasets, str)
            and dataset_names_or_raw_datasets in self.cached_datasets
        )
        if cache:
            ds = ds.cache(lazy=False)

        if isinstance(dataset_names_or_raw_datasets, str):
            ds_name = " " + dataset_names_or_raw_datasets
        else:
            ds_name = ""
        if idx is not None:
            ds_name += f" [{idx}]"
        print(f'Single data set length{ds_name}:', len(ds))
        return ds

    def get_raw(
            self, dataset_names_or_raw_datasets: Union[str, List[str], Dict[str, int], List[Tuple[str, int]]],
            discard_labelless_examples=False,
            filter_example_ids=None,
    ):
        """	Returns a dataset with the raw examples from the given datasets.

        Args:
            dataset_names_or_raw_datasets: A list of dataset names or raw datasets.
            discard_labelless_examples: If True, examples without a label are discarded.
            filter_example_ids: If given, only examples with an ID in this list are returned.

        Returns:
            A lazy dataset with the raw examples.
        """
        if isinstance(dataset_names_or_raw_datasets, (dict, list, tuple)):
            if isinstance(dataset_names_or_raw_datasets, dict):
                dataset_names_or_raw_datasets = list(dataset_names_or_raw_datasets.items())
            if (
                isinstance(dataset_names_or_raw_datasets, (list, tuple))
                and not isinstance(dataset_names_or_raw_datasets[0], (list, tuple))
            ):
                dataset_names_or_raw_datasets = [
                    (ds, 1) for ds in dataset_names_or_raw_datasets]
            # dataset_names_or_raw_datasets: List[Tuple[str, int]]
            dataset_names_or_raw_datasets = list(filter(
                lambda x: x[1] > 0, dataset_names_or_raw_datasets
            ))
            # recursive application of this function
            # allows for nested lists of dataset names -> why?
            return [
                (
                    self.get_raw(
                        name_or_ds[0] if isinstance(name_or_ds, (list, tuple))
                        else name_or_ds,
                        discard_labelless_examples=discard_labelless_examples,
                        filter_example_ids=filter_example_ids,
                    ),
                    name_or_ds[1],
                )
                for name_or_ds in dataset_names_or_raw_datasets
            ]
        elif isinstance(dataset_names_or_raw_datasets, str):
            # simple: dataset is given by its name
            ds = self.db.get_dataset(dataset_names_or_raw_datasets)
        else:
            assert isinstance(dataset_names_or_raw_datasets, lazy_dataset.Dataset), type(dataset_names_or_raw_datasets)
            # dataset is given as lazy_dataset -> why?
            ds = dataset_names_or_raw_datasets
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

    def _tile_and_intersperse(self, datasets, shuffle=False):
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

    def scale_and_mix(self, dataset, mixin_dataset=None):
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

    def _count_labels(self, raw_datasets, label_key, label_counts=None, reps=1):
        if label_counts is None:
            label_counts = defaultdict(lambda: 0)
        if isinstance(raw_datasets, list):
            labels = []
            for ds, ds_reps in raw_datasets:
                label_counts, cur_labels = self._count_labels(
                    ds, label_key, label_counts=label_counts, reps=ds_reps*reps
                )
                labels.append(cur_labels)
            return label_counts, labels

        labels = []
        for example in raw_datasets:
            cur_labels = sorted(set(to_list(example[label_key])))
            labels.append(cur_labels)
            for label in cur_labels:
                label_counts[label] += reps
        return label_counts, labels

    @staticmethod
    def _compute_label_repetitions(label_counts, min_counts):
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
        assert min_counts - 1 <= 0.9 * max_count, (min_counts, max_count)
        base_rep = 1 // (1 - (min_counts-1)/max_count)
        min_counts *= base_rep
        label_repetitions = {
            label: math.ceil(min_counts / count)
            for label, count in label_counts.items()
        }
        return label_repetitions

    def _build_repetition_groups(self, dataset, labels, label_repetitions):
        """	Build repetition groups for a dataset.

        Args:
            dataset: A dataset.
            labels: A list of lists of labels.
            label_repetitions: A dict mapping labels to repetitions.
        
        Returns:
            A list of (dataset, repetitions) pairs.
        """
        assert len(dataset) == len(labels), (len(dataset), len(labels))
        if isinstance(dataset, list):
            return [
                (group_ds, ds_reps*group_reps)
                for (ds, ds_reps), cur_labels in zip(dataset, labels)
                for group_ds, group_reps in self._build_repetition_groups(
                    ds, cur_labels, label_repetitions
                )
            ]
        idx_reps = [
            max([label_repetitions[label] for label in idx_labels])
            for idx_labels in labels
        ]
        rep_groups = {}
        for n_reps in set(idx_reps):
            rep_groups[n_reps] = np.argwhere(
                np.array(idx_reps) == n_reps
            ).flatten().tolist()
        datasets = []
        for n_reps, indices in sorted(
                rep_groups.items(), key=lambda x: x[0]
        ):
            datasets.append((dataset[sorted(indices)], n_reps))
        # ds = lazy_dataset.intersperse(*datasets)
        return datasets

    def segment_transform_and_fetch(
            self, dataset, segment=True, transform=True, fetch=True,
            train=False,
    ):
        print(f"length before segmentation, provider: {len(dataset)}")
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
                'min_overlap': 1.,
                'fade_length': config['train_transform']['stft']['window_length'],
                'label_key': 'events',
            }
