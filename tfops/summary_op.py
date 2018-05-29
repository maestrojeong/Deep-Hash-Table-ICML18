import tensorflow as tf

class SummaryWriter:
    def __init__(self, save_path):
        self.writer = tf.summary.FileWriter(save_path)

    def add_summary(self, tag, simple_value, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)])
        self.writer.add_summary(summary, global_step)

