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

NoduleInfoTuple = namedtuple(
    'NoduleInfoTuple',
    "isMalignant_bool, diameter_mm, series_uid, center_xyz"
)
MaskTuple = namedtuple(
    'MaskTuple',
    ("raw_dense_mask, body_mask, air_mask, raw_nodule_mask, nodule_mask, "
     "lung_mask, mal_mask"))


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
    def __init__(self, series_uid, buildMasks_bool=True):
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

        noduleInfo_list = getNoduleInfoList()
        # List of benign nodules for a specific CT scan
        self.benignInfo_list = [ni_tup for ni_tup in noduleInfo_list
                                if not ni_tup.isMalignant_bool
                                and ni_tup.series_uid == self.series_uid]
        self.benign_mask = self.buildAnnotationMask(self.benignInfo_list)[0]
        self.benign_indexes = sorted(set(self.benign_mask.nonzero()[0]))

        # List of malignant nodules for a specific CT scan
        self.malignantInfo_list = [ni_tup for ni_tup in noduleInfo_list
                                   if ni_tup.isMalignant_bool
                                   and ni_tup.series_uid == self.series_uid]
        self.malignant_mask = self.buildAnnotationMask(
            self.malignantInfo_list)[0]
        self.malignant_indexes = sorted(set(self.malignant_mask.nonzero()[0]))

    def buildAnnotationMask(self, noduleInfo_list, threshold_hu=-500):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for noduleInfo_tup in noduleInfo_list:
            center_irc = xyz2irc(
                noduleInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_tup
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)
            radius = 2

            # Index
            ci_min = ci - radius
            ci_max = ci + radius
            try:
                while self.hu_a[ci_max, cr, cc] > threshold_hu and \
                        self.hu_a[ci_min, cr, cc] > threshold_hu:
                    ci_min -= 1
                    ci_max += 1
            except IndexError:
                ci_min += 1
                ci_max -= 1

            # Row
            cr_min = ci - radius
            cr_max = ci + radius
            try:
                while self.hu_a[ci, cr_max, cc] > threshold_hu and \
                        self.hu_a[ci, cr_max, cc] > threshold_hu:
                    cr_min -= 1
                    cr_max += 1
            except IndexError:
                cr_min += 1
                cr_max -= 1

            # Column
            cc_min = ci - radius
            cc_max = ci + radius
            try:
                while self.hu_a[ci, cr, cc_max] > threshold_hu and \
                        self.hu_a[ci, cr, cc_min] > threshold_hu:
                    cc_min -= 1
                    cc_max += 1
            except IndexError:
                cc_min += 1
                cc_max -= 1

            slice_tup = (
                slice(ci - radius, ci + radius + 1),
                slice(ci - radius, ci + radius + 1),
                slice(ci - radius, ci + radius + 1),
            )
            boundingBox_a[slice_tup] = True

        thresholded_a = boundingBox_a & (self.hu_a > threshold_hu)
        # Every "False" next to a "True" is converted to "True".
        # iterations=2 means that the process is repeted two times.
        mask_a = morph.binary_dilation(thresholded_a, iterations=2)

        return mask_a, thresholded_a, boundingBox_a

    def build2dLungMask(self, mask_ndx):
        """Return a MaskTuple with different masks."""
        raw_dense_mask = self.hu_a[mask_ndx] > -300
        # The closing of an input image by a structuring element is the
        # erosion of the dilation of the image by the structuring
        # element.
        # This operation try to fill holes.
        dense_mask = morph.binary_closing(raw_dense_mask, iterations=2)
        # The opening of an input image by a structuring element is the
        # dilation of the erosion of the image by the structuring
        # element.
        # This operation try to remove small isolated blocks.
        dense_mask = morph.binary_opening(dense_mask, iterations=2)

        body_mask = morph.binary_fill_holes(dense_mask)
        air_mask = morph.binary_fill_holes(body_mask & ~dense_mask)
        air_mask = morph.binary_erosion(air_mask, iterations=1)

        lung_mask = morph.binary_dilation(air_mask, iterations=5)

        raw_nodule_mask = self.hu_a[mask_ndx] > -600
        raw_nodule_mask &= air_mask
        nodule_mask = morph.binary_opening(raw_nodule_mask, iterations=1)

        ben_mask = morph.binary_dilation(nodule_mask, iterations=1)
        ben_mask &= ~self.malignant_mask[mask_ndx]

        mal_mask = self.malignant_mask[mask_ndx]

        return MaskTuple(raw_dense_mask,
                         dense_mask,
                         body_mask,
                         air_mask,
                         raw_nodule_mask,
                         nodule_mask,
                         lung_mask,
                         ben_mask,
                         mal_mask)

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
# See also this: https://www.cameronmacleod.com/blog/python-lru-cache
ctCache_depth = 5
@functools.lru_cache(ctCache_depth, typed=True)
def getCt(series_uid):

    return Ct(series_uid)


# raw_cache = getCache('part2ch13_raw')
# See: http://www.grantjenks.com/docs/diskcache/tutorial.html
@raw_cache.memoize(typed=True)
def getCtRawNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)

    return ct_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid, buildMasks_bool=False)

    return len(ct.benign_indexes)


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
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(noduleInfo_tup.series_uid for
                                          noduleInfo_tup in
                                          getNoduleInfoList()))

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.noduleInfo_list = [x for x in self.noduleInfo_list if x.series_uid
                                in series_set]

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

        return len(self.noduleInfo_list)

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
                width_irc
            )
            nodule_t = torch.from_numpy(nodule_a).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        malignant_t = torch.tensor([not nodule_tup.isMalignant_bool,
                                    nodule_tup.isMalignant_bool],
                                   dtype=torch.long)


        return nodule_t, malignant_t, nodule_tup.series_uid, \
            torch.tensor(center_irc)


class PrepcacheLunaDataset(LunaDataset):
    def __getitem__(self, ndx):
        nodule_t, malignant_t, series_uid, center_t = super().__getitem__(ndx)
        getCtSampleSize(series_uid)

        return nodule_t, malignant_t, series_uid, center_t


class Luna2dSegmentationDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None,
                 contextSlices_count=2, augmentation_dict=None,
                 fullCt_bool=False):
        self.contextSlices_count = contextSlices_count
        self.augmentation_dict = augmentation_dict

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(noduleInfo_tup.series_uid for
                                          noduleInfo_tup in
                                          getNoduleInfoList()))

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            if fullCt_bool:
                self.sample_list.extend([
                    (series_uid, ct_ndx) for ct_ndx in
                    range(getCt(series_uid).hu_a.shape[0])
                ])
            else:
                self.sample_list.extend([
                    (series_uid, ct_ndx) for ct_ndx in
                    range(getCtSampleSize(series_uid))
                ])

        # !r calls repr()
        log.info("{!r}: {} {} series, {} slices".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation',
             False: 'training'}[isValSet_bool],
            len(self.sample_list)
        ))

    def __len__(self):

        return len(self.sample_list)

    def __getitem__(self, ndx):
        if isinstance(ndx, int):
            series_uid, sample_ndx = \
                self.sample_list[ndx % len(self.sample_list)]
            ct = getCt(series_uid)
            ct_ndx = self.sample_list[sample_ndx][1]
            useAugmentation_bool = False
        else:
            series_uid, ct_ndx, useAugmentation_bool = ndx
            ct = getCt(series_uid)

        ct_t = torch.zeros((self.contextSlices_count * 2 + 1 + 1,
                            512, 512))

        start_ndx = ct_ndx - self.contextSlices_count
        end_ndx = ct_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)

            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t /= 1000

        mask_tup = ct.build2dLungMask(ct_ndx)

        ct_t[-1] = torch.from_numpy(mask_tup.lung_mask.astype(np.float32))

        nodule_t = torch.from_numpy((mask_tup.mal_mask | mask_tup.ben_mask)
                                    .astype(np.float32)).unsqueeze(0)
        ben_t = torch.from_numpy(mask_tup.ben_mask.astype(np.float32))\
            .unsqueeze(0)
        mal_t = torch.from_numpy(mask_tup.mal_mask.astype(np.float32))\
            .unsqueeze(0)
        label_int = mal_t.max() + ben_t.max() * 2

        if self.augmentation_dict and useAugmentation_bool:
            if 'rotate' in self.augmentation_dict:
                if random.random() > 0.5:
                    ct_t = ct_t.rot90(1, [1, 2])
                    nodule_t = nodule_t.rot90(1, [1, 2])

            if 'flip' in self.augmentation_dict:
                dims = [d + 1 for d in range(2) if random.random() > 0.5]

                if dims:
                    ct_t = ct_t.flip(dims)
                    nodule_t = nodule_t.flip(dims)

            if 'noise' in self.augmentation_dict:
                noise_t = torch.randn_like(ct_t)
                noise_t *= self.augmentation_dict['noise']

                ct_t += noise_t

        return ct_t, nodule_t, label_int, ben_t, mal_t, ct.series_uid, ct_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, batch_size=80, **kwargs):
        self.needsShuffle_bool = True
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)

    def __len__(self):

        return 50000

    def __getitem__(self, ndx):
        if self.needsShuffle_bool:
            random.shuffle(self.series_list)
            self.needsShuffle_bool = False

        if isinstance(ndx, int):
            if ndx % self.batch_size == 0:
                self.series_list.append(self.series_list.pop(0))

            series_uid = self.series_list[ndx % ctCache_depth]
            ct = getCt(series_uid)

            if ndx % 3 == 0:
                ct_ndx = random.choice(ct.malignant_indexes or
                                       ct.benign_indexes)
            elif ndx % 3 == 1:
                ct_ndx = random.choice(ct.benign_indexes)
            elif ndx % 3 == 2:
                ct_ndx = random.choice(list(range(ct.hu_a.shape[0])))

            useAugmentation_bool = True
        else:
            series_uid, ct_ndx = useAugmentation_bool = ndx

        return super.__getitem__((series_uid, ct_ndx, useAugmentation_bool))
