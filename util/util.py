import collections
import copy
import datetime
import gc
import time

import numpy as np

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Irc stay for index, row and column. Xyz are the patient
# coordinates. x represent the right to left direction, y the
# anterior to posterior direction and z the inferior to superior
# direction.
# Usually the row and column dimensions have voxel sizes that are
# the same, and the index dimension has a larger value.
# Commonly, CTs are 512 rows by 512 columns, with the index
# dimension ranging from around 100 total slices up to perhaps 250
# slices.
IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

# The following function apply a scaling factor to produce images with
#  realistic proportions.
def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_tup):
    if direction_tup == (1, 0, 0, 0, 1, 0, 0, 0, 1):
        direction_ary = np.ones((3,))
    elif direction_tup == (-1, 0, 0, 0, -1, 0, 0, 0, 1):
        direction_ary = np.array((-1, -1, 1))
    else:
        raise Exception("Unsupported direction_tup: {}".format(direction_tup))

    coord_cri = (np.array(coord_xyz) - np.array(origin_xyz)) / \
        np.array(vxSize_xyz)
    coord_cri *= direction_ary

    return IrcTuple(*list(reversed(coord_cri.tolist())))


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_tup):
    coord_cri = np.array(list(reversed(coord_irc)))

    if direction_tup == (1, 0, 0, 0, 1, 0, 0, 0, 1):
        direction_ary = np.ones((3,))
    elif direction_tup == (-1, 0, 0, 0, -1, 0, 0, 0, 1):
        direction_ary = np.array((-1, -1, 1))
    else:
        raise Exception("Unsupported direction_tup: {}".format(direction_tup))

    coord_xyz = coord_cri * direction_ary * np.array(vxSize_xyz) \
        + np.array(origin_xyz)

    return XyzTuple(*coord_xyz.tolist())


def importstr(module_str, from_=None):
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        # In this case rsplit is equivalment to split
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        # getattr(module, sub_str) is equivalent to module.sub_str
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))

    return module


def prhist(ary, prefix_str=None, **kwargs):
    if prefix_str is None:
        prefix_str = ''
    else:
        prefix_str += ' '

    count_ary, bins_ary = np.histogram(ary, **kwargs)
    for i in range(count_ary.shape[0]):
        print("{}{:-8.2f}".format(prefix_str, bins_ary[i]),
              "{:-10}".format(count_ary[i]))

    print("{}{:-8.2f}".format(prefix_str, bins_ary[-1]))


def enumerateWithEstimate(iter, desc_str, start_ndx=0, print_ndx=4,
                          backoff=2, iter_len=None):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            duration_sec = ((time.time() - start_ts) /
                            (current_ndx - start_ndx + 1)
                            * (iter_len - start_ndx))

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
