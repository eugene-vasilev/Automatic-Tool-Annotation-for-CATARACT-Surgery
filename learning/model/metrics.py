from keras import backend as K
import tensorflow as tf


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def f1(y_true, y_pred):
    """F1 metric.

    Only computes a batch-wise average of F1-score.

    Computes the F1-score, a metric for classification,
    weighted average of the precision and recall.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    pre = precision(y_true_f, y_pred_f)
    rec = recall(y_true_f, y_pred_f)

    return 2 * ((pre * rec) / (pre + rec + K.epsilon()))


def auc(y_true, y_pred):
    """ROC AUC metric.

    Only computes a batch-wise average of ROC AUC score.

    For more information see https://www.tensorflow.org/api_docs/python/tf/metrics/auc
    """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
