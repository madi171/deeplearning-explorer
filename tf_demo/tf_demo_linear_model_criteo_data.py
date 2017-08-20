"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API. with Criteo dataset"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import pandas as pd
import tensorflow as tf

# Enable INFO log level for step debug
tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["Id", "Label", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13",
           "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
           "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

LABEL_COLUMN = "Label"

CATEGORICAL_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
                       "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]

CONTINUOUS_COLUMNS = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]


def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.
    sparse_columns = dict()
    for col_name in CATEGORICAL_COLUMNS:
        sparse_columns[col_name] = tf.contrib.layers.sparse_column_with_hash_bucket(col_name, hash_bucket_size=1000)

    # Continuous base columns.
    continue_columns = dict()
    for col_name in CONTINUOUS_COLUMNS:
        continue_columns[col_name] = tf.contrib.layers.real_valued_column(col_name)

    # Wide columns and deep columns. (Only consider sparse features)
    # Building wide features
    wide_columns = []
    # Add all sparse features to wide feature
    wide_columns.extend(sparse_columns.values())
    # Add crossing features
    wide_columns.append(
        tf.contrib.layers.crossed_column([sparse_columns["C2"], sparse_columns["C6"]], hash_bucket_size=int(1e4)))
    wide_columns.append(
        tf.contrib.layers.crossed_column([sparse_columns["C3"], sparse_columns["C16"]], hash_bucket_size=int(1e4)))

    # Building deep features
    deep_columns = []
    # Add all continue features to deep feature
    deep_columns.extend(continue_columns.values())
    # Add all embedded sparse feature
    deep_columns.extend([tf.contrib.layers.embedding_column(col, dimension=8) for col in sparse_columns.values()])

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[256, 128, 64], dropout=0.1)
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[256, 128, 64],
            dnn_dropout=0.1,
            fix_global_step_increment_bug=True)
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    train_file_name, test_file_name = "../datasets/criteo/train.tiny.csv", "../datasets/criteo/test.tiny.csv"
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skiprows=1,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    # Remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    print("\n------------")
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


def main():
    # train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
    #                   FLAGS.train_data, FLAGS.test_data)
    train_and_eval("", "wide", 1, "", "")


if __name__ == "__main__":
    main()
