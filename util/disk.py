import gzip

from diskcache import FanoutCache, Disk
from diskcache.core import BytesType, MODE_BINARY, BytesIO

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


# diskcache.Disk objects are responsible for serializing and
# deserializing data stored in the cache
class GzipDisk(Disk):
    def store(self, value, read, key=None):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large
            buffers (typically 2 or 4 GiB). Previously, inputs were
            limited to 2 GiB, compression and decompression operations
            did not properly handle results of 2 or 4 GiB.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :return: (size, mode, filename, value) tuple for Cache table
        """
        # pylint: disable=unidiomatic-typecheck
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2 ** 30):
                # Chunking
                gz_file.write(value[offset:offset + 2 ** 30])
            gz_file.close()

            value = str_io.getvalue()

        return super().store(value, read)

    def fetch(self, mode, filename, value, read):
        """
        Override from base class diskcache.Disk.

        Chunking is due to needing to work on pythons < 2.7.13:
        - Issue #27130: In the "zlib" module, fix handling of large
            buffers (typically 2 or 4 GiB). Previously, inputs were
            limited to 2 GiB, compression and decompression operations
            did not properly handle results of 2 or 4 GiB.

        :param int mode: value mode raw, binary, text or pickle
        :param str filename: filename or corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        """
        value = super().fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()

            while True:
                # Note: 2 ** 30 = 1 GB
                uncompressed_data = gz_file.read(2 ** 30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break

            value = read_csio.getvalue()

        return value


def getCache(scope_str):
    # Built atop Cache is diskcache.FanoutCache which automatically
    # shards the underlying database. Sharding is the practice of
    # horizontally partitioning data. Here it is used to decrease
    # blocking writes. While readers and writers do not block each
    # other, writers block other writers. Therefore a shard for every
    # concurrent writer is suggested. This will depend on your scenario.
    # The default value is 8.
    # timeout sets a limit on how long to wait for database
    # transactions.
    # size_limit is used as the total size of the cache. The size limit
    # of individual cache shards is the total size divided by the number
    # of shards.
    return FanoutCache("data-unversioned/cache/" + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=2e11)
