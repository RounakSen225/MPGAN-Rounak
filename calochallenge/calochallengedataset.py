from typing import List, Union, Optional

import torch
from torch import Tensor
import numpy as np

import logging
import h5py

from calochallenge.xmlhandler import XMLHandler
from calochallenge.highlevelfeatures import HighLevelFeatures as HLF
import math
# from os.path import exists

energy_cutoff = 1e-18

class CaloChallengeDataset(torch.utils.data.Dataset):
    _num_non_mask_features = 4

    def __init__(
        self,
        data_dir: str = "./datasets/",
        num_particles: int = 30,
        normalize: bool = True,
        feature_norms: List[float] = [1.0, 1.0, 1.0, 1.0],
        feature_shifts: List[float] = [-0.5, -0.5, -0.5, -0.5],
        particle: str = 'photon',
        num_features: int = 4,
        inc: List[int] = [],
        train_fraction: float = 0.7,
        logE: bool = True,
        logR: bool = True,
        use_mask: bool = False,
        train: bool = False,
        ignore_layer_12: bool = True
    ):
        self.data_dir = data_dir
        self.feature_norms = feature_norms
        self.feature_shifts = feature_shifts
        self.feature_maxes = None
        self.feature_mins = None
        self.normalize = normalize
        self.num_particles = num_particles
        self.particle = particle
        self.num_features = num_features
        self.logE = logE
        self.logR = logR
        self.use_mask = use_mask
        self.inc = inc
        self.ignore_layer_12 = ignore_layer_12

        #only considering photon data for now        
        self.HLF_1_photons = HLF('photon', filename= self.data_dir + 'binning_dataset_1_photons.xml')  
        if train:
            self.photon_file = h5py.File(self.data_dir + 'dataset_1_photons_1.hdf5', 'r')
        else:
            self.photon_file = h5py.File(self.data_dir + 'dataset_1_photons_2.hdf5', 'r')
        self.HLF_1_photons.CalculateFeatures(self.photon_file["showers"][:])

        dataset = self.format_data()

        if self.num_particles != -1:
            dataset = np.array(list(map(lambda x: x[x[:, 3].argsort()][-self.num_particles:], dataset)))

        if self.use_mask:
            mask = (dataset[:, :, 3:] != 0).astype(float)
            dataset = np.concatenate((dataset, mask), axis=2)

        dataset = torch.from_numpy(dataset)

        if self.logE:
            dataset[:, :, 3] = torch.log(dataset[:, :, 3] + energy_cutoff)

        if self.logR:
            dataset[:, :, 2] = torch.log(dataset[:, :, 2])

        self.num_features = dataset.shape[2]
        feature_maxes = [float(torch.max(dataset[:, :, i])) for i in range(self.num_features)]
        print('Max features: ', feature_maxes)
        feature_mins = [float(torch.min(dataset[:, :, i])) for i in range(self.num_features)]
        print('Min features: ', feature_mins)

        if self.normalize:
            self.normalize_features(dataset, feature_shifts=-0.5)
            print('\nAfter normalization: \n')
            feature_maxes = [float(torch.max(dataset[:, :, i])) for i in range(self.num_features)]
            print('Max features: ', feature_maxes)
            feature_mins = [float(torch.min(dataset[:, :, i])) for i in range(self.num_features)]
            print('Min features: ', feature_mins)

        if len(self.inc) != 0:
            data_inc = self.photon_file['incident_energies'][:]
            data_inc_sorted = np.sort(data_inc, axis=0).flatten()
            energies = data_inc_sorted[self.inc]
            data_inc = data_inc.flatten()
            indices = [np.where(data_inc == element)[0] for element in energies]
            dataset = torch.from_numpy(np.concatenate([dataset[idx] for idx in indices]))

        self.dataset = dataset
        jet_features = self.get_jet_features(dataset)

        tcut = int(len(dataset) * train_fraction)
        self.data = dataset[:tcut] if train else dataset[tcut:]
        self.jet_features = jet_features[:tcut] if train else jet_features[tcut:]
    
        print('Data shape: ', self.data.shape)
        logging.info("Dataset processed")

    def format_data(self):
        """
        Formats calochallenge dataset 1 based on the data given in .hdf5 and binning files
        """
        filename = self.data_dir + 'binning_dataset_1_' + self.particle + 's.xml'
        xml = XMLHandler(self.particle, filename=filename)
        data = self.photon_file['showers'][:]
        Ne = data.shape[0]
        coordinates = np.empty((0, self.num_features - 1))
        layer_count = 0
        for layer in range(len(xml.GetTotalNumberOfRBins())):
            r_list, a_list = xml.fill_r_a_lists(layer)
            stack = np.vstack(([layer_count]*len(r_list),a_list,r_list))
            coordinates = np.vstack((coordinates, stack.T))
            if len(r_list) > 0 or not self.ignore_layer_12:
                layer_count += 1
        coordinates = np.tile(coordinates, (Ne, 1, 1))
        data_tile = np.tile(data.T, (1, 1, 1))
        point_cloud = np.vstack((coordinates.T, data_tile)).T
        return point_cloud


    def get_jet_features(self, dataset: Tensor) -> Tensor:
        """
        Returns jet-level features. `Will be expanded to include jet pT and eta.`

        Args:
            dataset (Tensor):  dataset tensor of shape [N, num_particles, num_features],
              where the last feature is the mask.
            use_num_particles_jet_feature (bool): `Currently does nothing,
              in the future such bools will specify which jet features to use`.

        Returns:
            Tensor: jet features tensor of shape [N, num_jet_features].

        """
        jet_num_particles = (torch.sum(dataset[:, :, -1], dim=1) / self.num_particles).unsqueeze(1)
        logging.debug("{num_particles = }")
        return jet_num_particles

    #@classmethod
    def normalize_features(
        self,
        dataset: Tensor,
        feature_norms: Union[float, List[float]] = 1.0,
        feature_shifts: Union[float, List[float]] = 0.0,
    ) -> Optional[List]:
        """
        Normalizes dataset features (in place),
        by scaling to ``feature_norms`` maximum and shifting by ``feature_shifts``.

        If the value in the List for a feature is None, it won't be scaled or shifted.

        If ``fpnd`` is True, will normalize instead to the same scale as was used for the
        ParticleNet training in https://arxiv.org/abs/2106.11535.

        Args:
            dataset (Tensor): dataset tensor of shape [N, num_particles, num_features].
            feature_norms (Union[float, List[float]]): max value to scale each feature to.
              Can either be a single float for all features, or a list of length ``num_features``.
              Defaults to 1.0.
            feature_shifts (Union[float, List[float]]): after scaling, value to shift feature by.
              Can either be a single float for all features, or a list of length ``num_features``.
              Defaults to 0.0.
            fpnd (bool): Normalize features for ParticleNet inference for the
              Frechet ParticleNet Distance metric. Will override `feature_norms`` and
              ``feature_shifts`` inputs. Defaults to False.

        Returns:
            Optional[List]: if ``fpnd`` is False, returns list of length ``num_features``
            of max absolute values for each feature. Used for unnormalizing features.

        """
        num_features = dataset.shape[2]

        feature_mins = [float(min(torch.min(dataset[:, :, i]), 0)) for i in range(num_features)]
       
        if isinstance(feature_norms, float):
            feature_norms = np.full(num_features, feature_norms)

        if isinstance(feature_shifts, float):
            feature_shifts = np.full(num_features, feature_shifts)

        for i in range(num_features):
            dataset[:, :, i] -= feature_mins[i]

        feature_maxes = [float(torch.max(torch.abs(dataset[:, :, i]))) for i in range(num_features)]
        self.feature_maxes = feature_maxes
        self.feature_mins = feature_mins
        self.feature_norms = feature_norms
        self.feature_shifts = feature_shifts

        logging.debug(f"{feature_maxes = }")

        for i in range(num_features):
            if feature_maxes[i] == 0: continue
            if feature_norms[i] is not None:
                dataset[:, :, i] /= feature_maxes[i]
                dataset[:, :, i] *= feature_norms[i]

            if feature_shifts[i] is not None:
                dataset[:, :, i] += feature_shifts[i]

        return feature_maxes

    #@classmethod
    def unnormalize_features(
        self,
        dataset: Union[Tensor, np.ndarray],
        ret_mask_separate: bool = True,
        is_real_data: bool = False,
        zero_mask_particles: bool = True,
        zero_neg_pt: bool = True,
    ):
        """
        Inverts the ``normalize_features()`` function on the input ``dataset`` array or tensor,
        plus optionally zero's the masked particles and negative pTs.
        Only applicable if dataset was normalized first
        i.e. ``normalize`` arg into JetNet instance is True.

        Args:
            dataset (Union[Tensor, np.ndarray]): Dataset to unnormalize.
            ret_mask_separate (bool): Return the jet and mask separately. Defaults to True.
            is_real_data (bool): Real or generated data. Defaults to False.
            zero_mask_particles (bool): Set features of zero-masked particles to 0.
              Not needed for real data. Defaults to True.
            zero_neg_pt (bool): Set pT to 0 for particles with negative pt.
              Not needed for real data. Defaults to True.

        Returns:
            Unnormalized dataset of same type as input. Either a tensor/array of shape
            ``[num_jets, num_particles, num_features (including mask)]`` if ``ret_mask_separate``
            is False, else a tuple with a tensor/array of shape
            ``[num_jets, num_particles, num_features (excluding mask)]`` and another binary mask
            tensor/array of shape ``[num_jets, num_particles, 1]``.
        """
        if self.normalize:
            #raise RuntimeError("Can't unnormalize features if dataset has not been normalized.")

            num_features = dataset.shape[2]

            for i in range(num_features):
                if self.feature_shifts[i] is not None and self.feature_shifts[i] != 0:
                    dataset[:, :, i] -= self.feature_shifts[i]

                if self.feature_norms[i] is not None:
                    dataset[:, :, i] /= self.feature_norms[i]
                    dataset[:, :, i] *= self.feature_maxes[i]

            for i in range(num_features):
                dataset[:, :, i] += self.feature_mins[i]

        if self.logE:
            dataset[:, :, 3] = torch.exp(dataset[:, :, 3]) - energy_cutoff

        if self.logR:
            dataset[:, :, 2] = torch.exp(dataset[:, :, 2])

        mask = dataset[:, :, -1] >= 0.5 if self.use_mask else None

        if not is_real_data and zero_mask_particles and self.use_mask:
            dataset[~mask] = 0

        if not is_real_data and zero_neg_pt:
            dataset[:, :, 2][dataset[:, :, 2] < 0] = 0
            dataset[:, :, 0][dataset[:, :, 0] < 0] = 0

        return dataset[:, :, :self._num_non_mask_features], mask if ret_mask_separate else dataset
    
    def get_boundaries(self):
        filename = self.data_dir + 'binning_dataset_1_' + self.particle + 's.xml'
        xml = XMLHandler(self.particle, filename=filename)
        l_list = []
        r_list = []
        alpha_list = []
        layer_count = 0
        for l in range(len(xml.GetTotalNumberOfRBins())):
            r, alpha = xml.fill_r_a_lists(l)
            r = list(set(r))
            alpha = list(set(alpha))
            if len(r) > 0:
                l_list.append(layer_count)
                if self.logR: r_list.append(torch.log(torch.Tensor(sorted(r))))
                else: r_list.append(torch.Tensor(sorted(r)))
                alpha_list.append(torch.Tensor(sorted(alpha)))
            if len(r) > 0 or not self.ignore_layer_12: layer_count += 1
        l_list = torch.Tensor(sorted(l_list))
        l_list, alpha_list, r_list = self._normalize(l_list=l_list, alpha_list=alpha_list, r_list=r_list, feature_shifts=-0.5)
        l_boundaries = []
        alpha_boundaries = []
        r_boundaries = []
        l_boundaries = (l_list[:-1]+l_list[1:])/2
        for alpha_layer in alpha_list:
            alpha_boundaries.append((alpha_layer[:-1]+alpha_layer[1:])/2)
        for r_layer in r_list:
            r_boundaries.append((r_layer[:-1] + r_layer[1:])/2)
        return l_boundaries, alpha_boundaries, r_boundaries
    
    def _normalize(self, 
        l_list, 
        alpha_list, 
        r_list,
        feature_norms = 1.0,
        feature_shifts = 0.0,
    ):
        if not self.normalize: return l_list, alpha_list, r_list
        l_norm, alpha_norm, r_norm = [], [], []
        if l_list.nelement() != 0 and torch.max(l_list) != 0:
            l_norm = torch.Tensor(l_list / torch.max(torch.abs(l_list)) * feature_norms + feature_shifts)
        for i in range(len(r_list)):
            if r_list[i].nelement() != 0 and torch.max(r_list[i]) != 0: 
                r_norm.append(torch.Tensor(r_list[i] / torch.max(torch.abs(r_list[i])) * feature_norms + feature_shifts))
        for i in range(len(alpha_list)):
            if alpha_list[i].nelement() != 0:
                alpha_shifted = alpha_list[i] - torch.min(alpha_list[i])
                if torch.max(alpha_shifted) != 0:
                    alpha_norm.append(torch.Tensor(alpha_shifted / torch.max(torch.abs(alpha_shifted)) * feature_norms + feature_shifts))
                else:
                    alpha_norm.append(alpha_shifted)
            else:
                alpha_norm.append(alpha_list[i])
        return l_norm, alpha_norm, r_norm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.jet_features[idx]
