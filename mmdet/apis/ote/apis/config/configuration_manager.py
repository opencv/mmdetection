# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import copy
import tempfile
import torch

from typing import Optional, List

from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.logging import logger_factory

from ..detection import MMDetectionParameters
from .config_mapper import ConfigMappings
from .task_types import MMDetectionTaskType

from mmcv import Config
from mmdet.apis import set_random_seed

logger = logger_factory.get_logger("MMDetectionTask")


class MMDetectionConfigManager(object):
    def __init__(self, task_environment: TaskEnvironment, task_type: MMDetectionTaskType, scratch_space: str,
                 random_seed: Optional[int] = 42):
        """
        Class that configures an mmdetection model and training configuration. Initializes the task-specific
        configuration. Sets the work_dir for mmdetection and the number of classes in the model. Also seeds random
        generators.

        :param task_environment: Task environment for the task, containing configurable parameters, labels, etc.
        :param task_type: MMDetectionTaskType of the task at hand
        :param scratch_space: Path to working directory
        :param random_seed: Optional int to seed random generators.
        """
        # Initialize configuration mappings for the task type and get configurable parameters
        self.config_mapper = ConfigMappings()
        conf_params = task_environment.get_configurable_parameters(instance_of=MMDetectionParameters)

        # Build the config
        template = conf_params.algo_backend.template.value
        self.model_name = conf_params.algo_backend.model_name.value
        base_dir = os.path.abspath(os.path.dirname(template))
        model_config = os.path.join(base_dir, conf_params.algo_backend.model.value)
        data_pipeline = os.path.join(base_dir, conf_params.algo_backend.data_pipeline.value)

        self.custom_lr_schedule = self._get_custom_lr_schedule(model_config)

        logger.warning(f'model config {model_config}')
        logger.warning(f'data pipeline {data_pipeline}')

        self._compose_config(
            model_file=model_config,
            schedule_file=None,
            dataset_file=data_pipeline,
            runtime_file=self.config_mapper.get_runtime_file('default')
        )

        # Fix config.
        if hasattr(self.config, 'total_epochs'):
            self.config.runner.max_epochs = self.config.total_epochs

        # Assign additional parameters
        # FIXME.
        self.config.gpu_ids = range(1)

        # this controls the maximum number of ground truth bboxes per image that will be processed on the gpu. If an
        # image contains more gt bboxes than this, they will be moved to the cpu for processing. It is set to avoid
        # gpu oom errors
        self._max_number_gt_bbox_per_image_on_gpu = 100

        # mmdetection training needs workdir to store logs and checkpoints
        self.config.work_dir = scratch_space
        self.config.seed = random_seed
        set_random_seed(random_seed)

        # Specify label names in config
        labels = task_environment.labels
        self.label_names = [lab.name for lab in labels]
        # FIXME. What for?
        self.config.labels = labels
        self.set_data_classes()

        # Finally, update the config to make sure the model heads have the correct number of classes, and the values
        # set in the configurable parameters are reflected in the config
        self._update_model_classification_heads()
        self.update_project_configuration(conf_params)

    def _get_custom_lr_schedule(self, model_file: str):
        schedule_sections = ('optimizer', 'optimizer_config', 'lr_config', 'momentum_config')
        model_config = Config.fromfile(model_file)
        schedule_config = dict()
        for section in schedule_sections:
            if section in model_config:
                schedule_config[section] = model_config[section]
        return schedule_config
    def _compose_config(self, model_file: str, schedule_file: str, dataset_file: str, runtime_file: str):
        """
        Constructs the full mmdetection configuration from files containing the different config sections

        :param model_file: Path to the model config file
        :param schedule_file: Path to the learning rate schedule file
        :param dataset_file: Path to the dataset config file
        :param runtime_file: Path to the runtime config file
        """
        config_file_list = [model_file, schedule_file, dataset_file, runtime_file]
        config = dict()
        for filename in config_file_list:
            if filename is None:
                continue
            update_config = Config.fromfile(filename)
            config = Config._merge_a_into_b(update_config, config)
        self.config = Config(config)

    def set_data_classes(self):
        """ Sets the label names for the different subsets """
        subsets = ['train', 'val', 'test']
        for subset in subsets:
            self.config.data[subset].classes = self.label_names

    def update_project_configuration(self, configurable_parameters: MMDetectionParameters):
        """
        Update the mmdetection model configuration according to the configurable parameters.

        :param configurable_parameters: Parameters to set

        """
        learning_rate_schedule_name = configurable_parameters.learning_parameters.learning_rate_schedule.value
        learning_rate_warmup_iters = configurable_parameters.learning_parameters.learning_rate_warmup_iters.value
        self._update_learning_rate_schedule(learning_rate_schedule_name, learning_rate_warmup_iters)
        self.config.runner.max_epochs = int(configurable_parameters.learning_parameters.num_epochs.value)
        self.config.optimizer.lr = float(configurable_parameters.learning_parameters.learning_rate.value)
        self.config.data.samples_per_gpu = int(configurable_parameters.learning_parameters.batch_size.value)

    def update_dataset_subsets(self, dataset: Dataset, model: torch.nn.Module = None):
        """
        Set the correct dataset subsets in an mmdetection configuration

        :param dataset: Dataset that defines the subsets
        :param model: If a model is passed, the config of that model will be updated instead of the config maintained
            by the config_manager.

        :return: model with updated data configuration
        """
        if model is None:
            cfg_to_change = self.config.data
        else:
            cfg_to_change = model.cfg.data

        cfg_to_change.test.ote_dataset = dataset.get_subset(Subset.TESTING)
        cfg_to_change.train.ote_dataset = dataset.get_subset(Subset.TRAINING)
        cfg_to_change.val.ote_dataset = dataset.get_subset(Subset.VALIDATION)
        return model

    @property
    def config_copy(self):
        """
        Return a copy of the config, for passing to certain mmdetection methods that modify config in place, such
        as train_detector

        :return Config:
        """
        return copy.deepcopy(self.config)

    @staticmethod
    def config_to_string(config: Config) -> str:
        """
        Convert a full mmdetection config to a string.

        :param config: configuration object to convert
        :return str: string representation of the configuration
        """
        config_copy = copy.deepcopy(config)
        # Clean config up by removing dataset and label entities as this causes the pretty text parsing to fail
        config_copy.data.test.ote_dataset = None
        config_copy.data.train.ote_dataset = None
        config_copy.data.val.ote_dataset = None
        config_copy.labels = [label.name for label in config.labels]
        return Config(config_copy).pretty_text

    @staticmethod
    def config_from_string(config_string: str) -> Config:
        """
        Generate an mmdetection config dict object from a string.

        :param config_string: string to parse
        :return config: configuration object
        """
        with tempfile.NamedTemporaryFile('w', suffix='.py') as temp_file:
            temp_file.write(config_string)
            temp_file.flush()
            return Config.fromfile(temp_file.name)

    def save_config_to_file(self):
        """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """
        filepath = os.path.join(self.config.work_dir, 'config.py')
        config_string = self.config_to_string(self.config)
        with open(filepath, 'w') as f:
            f.write(config_string)

    def _replace_config_section_from_file(self, file) -> Config:
        """
        Replace part of the configuration by a config file.

        :param file: Config file containing the config section to update
        :return Config: config section that was updated
        """
        config = self.config_copy
        config_section = Config.fromfile(file)
        new_config = Config._merge_a_into_b(config_section, config)
        self.config = Config(new_config)
        return config_section

    def _update_learning_rate_schedule(self, schedule_name: str, warmup_iters: int):
        """
        Update the learning rate scheduling config section in the current configuration

        :param schedule_file: Path to the learning rate schedule file containing the desired schedule
        """

        # remove old optimizer and lr config sections
        sections_to_pop = ('optimizer', 'optimizer_config', 'lr_config', 'momentum_config')
        for section in sections_to_pop:
            if section in self.config:
                self.config.pop(section)

        if schedule_name == 'custom':
            for section in sections_to_pop:
                if section in self.custom_lr_schedule:
                    self.config[section] = self.custom_lr_schedule[section]
        else:
            schedule_file = self.config_mapper.get_schedule_file(schedule_name)
            logger.warning(f'Update LR schedule from {schedule_file}')
            self._replace_config_section_from_file(schedule_file)

        # Set gradient clipping if required for the model in config
        # self._update_gradient_clipping()

        # Set learning rate warmup settings.
        if warmup_iters > 0:
            self.config.lr_config.warmup = 'linear'
            self.config.lr_config.warmup_ratio = 1.0 / 3
            self.config.lr_config.warmup_iters = warmup_iters

    def _update_model_classification_heads(self):
        """ Modify the number of classes of the model in the box heads """
        if 'roi_head' in self.config.model.keys():
            if isinstance(self.config.model.roi_head.bbox_head, List):
                for head in self.config.model.roi_head.bbox_head:
                    head.num_classes = len(self.label_names)
            else:
                self.config.model.roi_head.bbox_head.num_classes = len(self.label_names)
        elif 'bbox_head' in self.config.model.keys():
            self.config.model.bbox_head.num_classes = len(self.label_names)

    def get_lr_schedule_friendly_name(self, lr_policy_type: str):
        """
        Gives the user friendly name of the learning rate schedule associated with lr_policy_type

        :param lr_policy_type: name of the learning rate policy type
        :return: str, friendly name of this learning rate schedule
        """
        return self.config_mapper.get_schedule_friendly_name(lr_policy_type)

    def _search_in_config_dict(self, config_dict, key_to_search, prior_keys=None, results=None):
        """
        Recursively searches a config_dict for all instances of key_to_search and returns the key path to them
        :param config_dict: dict to search
        :param key_to_search: dict key to look for
        :return (value_at_key_to_search, key_path_to_key_to_search)
        """
        if prior_keys is None:
            prior_keys = list()
        if results is None:
            results = []
        if isinstance(config_dict, List):
            dict_to_search_in = {k: v for k, v in enumerate(config_dict)}
        else:
            dict_to_search_in = config_dict
        if not isinstance(dict_to_search_in, dict):
            return results
        for key, value in dict_to_search_in.items():
            current_key_path = prior_keys + [key]
            if key == key_to_search:
                results.append([value, prior_keys])
            self._search_in_config_dict(value, key_to_search, current_key_path, results)
        return results
