import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch13_raw')

NoduleInfoTuple = namedtuple('NoduleInfoTuple', 'isMalignant_bool',
                             'diameter_mm', 'series_uid', 'center_xyz')

# Decorator to wrap a function with a memoizing callable that saves up
# to the maxsize most recent calls. It can save time when an expensive
# or I/O bound function is periodically called with the same arguments.
# In computing, memoization or memoisation is an optimization technique
# used primarily to speed up computer programs by storing the results of
# expensive function calls and returning the cached result when the same
# inputs occur again
@functools.lru_cache(1)
def getNoduleInfoList(requireDataOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all
    # of the subsets yet
    mhd_list = glob.glob("data-unversioned/part2/luna/subset*/*.mhd")
    dataPresentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open("data/part2/luna/annotations.csv", 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm))

    noduleInfo_list = []
    with open("data/part2/luna/candidates.csv", 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in dataPresentOnDisk_set and \
                    requireDataOnDisk_bool:
                continue

            isMalignant_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            # If a match is found, candidateDiameter_mm =
            # annotationDiameter_mm, else candidateDiameter_mm = 0.
            # A match is when the candidate and the annotation centers
            # have a Manhattan distance lower or equal to 1/4 of the
            # annotation diameter
            candidateDiameter_mm = 0.0
            for annotationDiameter_mm, annotationDiameter_mm in \
                    diameter_dict.get(series_uid, []):
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] -
                                   annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            noduleInfo_list.append(NoduleInfoTuple(isMalignant_bool,
                                                   candidateDiameter_mm,
                                                   series_uid,
                                                   candidateCenter_xyz))
    # Put first the malignant nodules sorted from the smallest to the
    # largest
    noduleInfo_list.sort(reverse=True)
    return noduleInfo_list


# Ct is used to load individual CT scans
class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob("data-unversioned/part2/luna/subset*/{}.mhd"
                             .format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in
        # https://en.wikipedia.org/wiki/Hounsfield_scale HU are scaled
        # oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc
        # (water) being 0.
        # This gets rid of negative density stuff used to indicate
        # out-of-FOV
        ct_a[ct_a < -1000] = -1000

        # This nukes any weird hotspots and clamps bone down
        ct_a[ct_a > 1000] = 1000

        self.series_uid = series_uid
        self.hu_a = ct_a

        # Origin: coordinates of the pixel/voxel with index (0,0,0) in
        # physical units (i.e. mm). Default is zero i.e. origin of
        # physical space.
        # Spacing: distance between adjacent pixels/voxels in each
        # dimension given in physical units. Default is one i.e.
        # (1 mm, 1 mm, 1 mm).
        # Direction Matrix: mapping/rotation between direction of the
        # pixel/voxel axes and physical directions. Default is identity
        # matrix. The matrix is passed as a 1D array in row-major form.
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_tup = tuple(int(round(x)) for x in
                                   ct_mhd.GetDirection())

    def getRawNodule(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_tup)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz,
                self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


# If typed is set to true, function arguments of different types will be
# cached separately. For example, f(3) and f(3.0) will be treated as
# distinct calls with distinct results.
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)
    return ct_chunk, center_irc


def getCtAugmentedNodule(augmentation_dict, series_uid, center_xyz, width_irc,
                         use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawNodule(series_uid, center_xyz,
                                              width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    trasform_t = torch.eye(4).to(torch.float64)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                trasform_t[i, i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            trasform_t[3, i] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            trasform_t[i, i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_red = random.random() * math.pi * 2
        s = math.sin(angle_red)
        c = math.cos(angle_red)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float64)

        trasform_t @= rotation_t

    # Generates a 3D flow field (sampling grid)
    affine_t = F.affine_grid(
        # Input batch of affine matrices
        trasform_t[:3].unsqueeze(0).to(torch.float32),
        # Target output image size
        ct_t.size()
    )
    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode='border'
    ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmented_chunk['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None,
                 sortby_str='random', ratio_int=0, augmentation_dict=None,
                 noduleInfo_list=None):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if noduleInfo_list:
            self.noduleInfo_list = copy.copy(noduleInfo_list)
            self.use_cache = False
        else:
            self.noduleInfo_list = copy.copy(getNoduleInfoList())
            self.use_cache = True

        if series_uid:
            self.noduleInfo_list = [x for x in self.noduleInfo_list
                                    if x[2] == series_uid]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.noduleInfo_list = self.noduleInfo_list[::val_stride]
            assert self.noduleInfo_list
        elif val_stride > 0:
            del self.noduleInfo_list[::val_stride]
            assert self.noduleInfo_list

        if sortby_str == 'random':
            random.shuffle(self.noduleInfo_list)
        elif sortby_str == 'series_uid':
            self.noduleInfo_list.sort(key=lambda x: (x.series_uid,
                                                     x.center_xyz))
        elif sortby_str == 'malignancy_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.benign_list = [nt for nt in self.noduleInfo_list if not
                            nt.isMalignant_bool]
        self.malignant_list = [nt for nt in self.noduleInfo_list if
                               nt.isMalignant_bool]

        log.info("{!r}: {} {} samples, {} ben, {} mal, {} ratio".format(
            self,
            len(self.noduleInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.benign_list),
            len(self.malignant_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.benign_list)
            random.shuffle(self.malignant_list)

    def __len__(self):
        if self.ratio_int:
            return 200000
        else:
            return len(self.noduleInfo_list) // 20

    def __getitem__(self, ndx):
        if self.ratio_int:
            malignant_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                benign_ndx = ndx - 1 - malignant_ndx
                benign_ndx %= len(self.benign_list)
                nodule_tup = self.benign_list[benign_ndx]
            else:
                malignant_ndx %= len(self.malignant_list)
                nodule_tup = self.malignant_list[malignant_ndx]
        else:
            nodule_tup = self.noduleInfo_list[ndx]

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            nodule_t, center_irc = getCtAugmentedNodule(
                self.augmentation_dict,
                nodule_tup.series_uid,
                nodule_tup.center_xyz,
                width_irc,
                self.use_cache
            )
        elif self.use_cache:
            nodule_a, center_irc = getCtRawNodule(
                nodule_tup.series_uid,
                nodule_tup.center_xyz,
                width_irc
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
        else:
            ct = getCt(nodule_tup.series_uid)
            nodule_a, center_irc = ct.getRawNodule(
                nodule_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        malignant_t = torch.tensor([
            not nodule_tup.isMalignant_bool,
            nodule_tup.isMalignant_bool
            ], dtype=torch.long)


        return nodule_t, malignant_t, nodule_tup.series_uid, \
            torch.tensor(center_irc)
