# Sample code for writing a transformation and writing output to JSON with Apache Beam.
 
import os
import json

import apache_beam as beam
from apache_beam import coders
from apache_beam.io.iobase import Write
from apache_beam.transforms import PTransform   

class _JsonSink(beam.io.FileSink):
    """A Dataflow sink for writing JSON files."""

    def __init__(self,
                 file_path_prefix,
                 file_name_suffix='',
                 num_shards=0,
                 shard_name_template=None,
                 coder=coders.ToStringCoder(),
                 compression_type=beam.io.CompressionTypes.AUTO):

        super(_JsonSink, self).__init__(
            file_path_prefix,
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            shard_name_template=shard_name_template,
            coder=coder,
            mime_type='text/plain',
            compression_type=compression_type)
        self.last_rows = dict()

    def open(self, temp_path):
        """ Open file and initialize it w opening a list."""
        file_handle = super(_JsonSink, self).open(temp_path)
        file_handle.write('[\n')
        return file_handle

    def write_record(self, file_handle, value):
        """Writes a single encoded record converted to JSON and terminates the
        line w a comma."""
        if self.last_rows.get(file_handle, None) is not None:
            file_handle.write(self.coder.encode(
                json.dumps(self.last_rows[file_handle])))
            file_handle.write(',\n')

        self.last_rows[file_handle] = value

    def close(self, file_handle):
        """Finalize the JSON list and close the file handle returned from
        ``open()``. Called after all records are written.
        """
        if file_handle is not None:
            # Write last row without a comma
            file_handle.write(self.coder.encode(
                json.dumps(self.last_rows[file_handle])))

            # Close list and then the file
            file_handle.write('\n]\n')
            file_handle.close()


class WriteToJson(PTransform):
    """PTransform for writing to JSON files."""

    def __init__(self,
                 file_path_prefix,
                 file_name_suffix='',
                 num_shards=0,
                 shard_name_template=None,
                 coder=coders.ToStringCoder(),
                 compression_type=beam.io.CompressionTypes.AUTO):

        self._sink = _JsonSink(file_path_prefix, file_name_suffix, num_shards,
                               shard_name_template, coder, compression_type)
