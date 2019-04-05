import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """Applie weight sharing to the given model."""
    for num, layer in enumerate(model.layers):
        weights_array = layer.get_weights()
        result_weights = []
        for weights in weights_array:
            if np.unique(weights).shape[0] > 1:
                start_shape = weights.shape
                if len(start_shape) >= 2:
                    weights = np.reshape(weights, (np.prod(weights.shape[0:-1]), weights.shape[-1]))
                elif len(start_shape) == 1:
                    weights = np.expand_dims(weights, -1)

                shape = weights.shape
                mat = csr_matrix(weights) if shape[0] < shape[1] else csc_matrix(weights)
                min_ = np.min(mat.data)
                max_ = np.max(mat.data)
                fitting_data = mat.data.reshape(-1, 1)
                space = np.linspace(min_, max_, num=np.min([2**bits, len(fitting_data)]))
                kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1,
                                precompute_distances=True, algorithm="full", n_jobs=-1)
                kmeans.fit(fitting_data)
                mat.data = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)

                result_weights.append(np.reshape(mat.toarray(), start_shape))
            else:
                result_weights.append(weights)

        model.layers[num].set_weights(result_weights)

    return model
