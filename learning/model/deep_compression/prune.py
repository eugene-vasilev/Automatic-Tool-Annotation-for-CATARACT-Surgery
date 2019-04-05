import numpy as np


def prune(model, std=0.25, quite=True):
    if not quite:
        print_compressing_result(model)
        print('')
        print('')

    for num, layer in enumerate(model.layers):
        weights_array = layer.get_weights()
        result_weights = []
        for input_num, weights in enumerate(weights_array):
            threshold = np.std(weights) * std
            print('Pruning with threshold : {} for layer {} , input num {}'
                  .format(threshold, layer.name, input_num))
            mask = np.ones_like(weights)
            new_mask = np.where(abs(weights) < threshold, 0, mask)
            result_weights.append(weights * new_mask)

        model.layers[num].set_weights(result_weights)

    if not quite:
        print('')
        print('')
        print_compressing_result(model)

    return model


def print_compressing_result(model):
    nonzero = total = 0
    for layer in model.layers:
        weights_array = layer.get_weights()
        for input_num, weights in enumerate(weights_array):
            nonzero_count = np.count_nonzero(weights)
            total_params = np.prod(weights.shape)
            nonzero += nonzero_count
            total += total_params
            print('''{} : {} | nonzeros = {} / {} ({}%) | total pruned = {} | shape = {}'''
                  .format(layer.name, input_num, nonzero_count, total_params, 100 * nonzero_count / total_params,
                          total_params - nonzero_count, weights.shape))

    print('''alive : {}, pruned : {}, total : {}, Compression rate : x{} | {}% pruned'''
          .format(nonzero, total - nonzero, total, total / nonzero, 100 * (total - nonzero) / total))
