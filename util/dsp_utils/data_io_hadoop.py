import io
import json
import os
import sys
import csv
import logging
import gzip
import pickle
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict

from bloomberg.ds.katiehdfs.hadoopidentityclient import HadoopIdentityClient
import pandas as pd


logger = logging.getLogger(__name__)

class DataIO: 

    def __init__(self, local=True): 

        self.client = HadoopIdentityClient()    
        self.logger = logging.getLogger(__name__)

    def download(self, hdfs_path, local_file_path, overwrite=True, n_threads=-1):
        self.logger.debug("Downloading file to %s from hdfs path %s" % (local_file_path, hdfs_path))
        self.client.download(hdfs_path, local_file_path, overwrite=overwrite, n_threads=n_threads)

    def upload(self, hdfs_path, local_file_path, overwrite=True, n_threads=-1):
        self.logger.debug("Uploading file %s to hdfs path %s" % (local_file_path, hdfs_path))
        self.client.upload(hdfs_path, local_file_path, overwrite=overwrite, n_threads=n_threads)

    def read_hdfs_data(self, fname): 
        with self.client.read(fname) as reader:
            return json.load(reader)

    def read_pkl_data(self, fname, is_pandas=True):
        with self.client.read(fname) as reader:

            if is_pandas:
                return pd.read_pickle(reader, compression=None)
            else:
                return pickle.load(reader) 

    def read_text(self, fname):
        with self.client.read(fname, encoding="utf-8") as reader:
            text = reader.read() 

        return text


    def write_figure(self, pplot, output_file, hdfs=True):
        """
        Write plot to hdfs file.

        Args:
            pplot (matplotlib.pyplot): plot to be saved
            output_file: path to hdfs to store the figure.

        """

        with io.BytesIO() as buf:
            pplot.savefig(buf, format='png')
            buf.seek(0)
            with self.client.write(output_file, overwrite=True):
                writer.write(buf.getvalue())
