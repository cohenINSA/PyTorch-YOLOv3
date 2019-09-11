import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #self.writer = tf.compat.v1.summary.FileWriter(log_dir)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        #summary = tf.compat.v1.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, global_tag, tag_value_pairs, step):
        """Log scalar variables."""
        #summary = tf.compat.v1.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        #self.writer.add_summary(summary, step)
        self.writer.add_scalars(global_tag, {tag: val for tag, val in tag_value_pairs}, step)

    def close(self):
        self.writer.close()


class DictSaver:
    """
    Saves key-value pairs in a pandas DataFrame. The keys are the column labels, each new added data is a new row.
    Data must be added one by one.
    """
    def __init__(self):
        self.data = None

    def add_data(self, data, index):
        """
        :param data: dict with the new values to add and their labels.
        :param index:
        :return:
        """
        assert type(data) == dict, "DataSaver requires data as dictionnaries."
        for key, val in data.items():
            if not type(val) == list:
                data[key] = [val.item()]

        if self.data is None:
            # create new DataFrame
            self.data = pd.DataFrame.from_dict(data)
            self.data.index = [index] if not type(index) == list else index
        else:
            # add to existing dataframe
            new_df = pd.DataFrame.from_dict(data)
            new_df.index = [index]
            if index in self.data.index:
                print("Index %s already existing. Replacing old values." % str(index))
                self.data.drop(index)
            self.data = self.data.append(new_df)

    def save(self, path=None):
        if self.data is not None:
            self.data.to_csv(path)
