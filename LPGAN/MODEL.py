import tensorflow as tf

from .DATA import *
from .CONVNET import *

def conv_net_block(conv_net, net_info, tensor_list, is_first, is_training, act_o, dilation_rate):
    seed = FLAGS['process_random_seed']
    trainable = conv_net['trainable']
    tensor = tensor_list[conv_net['input_index']]
    if is_first:
        layer_name_format = '%12s'
        net_info.architecture_log.append('========== net_name = %s ==========' % conv_net['net_name'])
        net_info.architecture_log.append('[%s][%4d] : (%s)' % (layer_name_format % 'input', tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list())))
        if FLAGS['mode_use_debug']:
            print(net_info.architecture_log[-2])
            print(net_info.architecture_log[-1])
    with tf.compat.v1.variable_scope(conv_net['net_name']):
        for l_index, layer_o in enumerate(conv_net['layers']):
            layer = layer_o['name']
            if layer == "relu":
                tensor = exe_relu_layer(tensor)
            elif layer == "prelu":
                tensor = exe_prelu_layer(tensor, net_info, l_index, is_first, act_o)
            elif layer == "lrelu":
                tensor = exe_lrelu_layer(tensor, layer_o)
            elif layer == "bn":
                tensor = exe_bn_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, act_o)
            elif layer == "in":
                tensor = exe_in_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "ln":
                tensor = exe_ln_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "conv":
                tensor = exe_conv_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, seed, dilation_rate)
            elif layer == "conv_res":
                tensor = exe_conv_res_layer(tensor, layer_o, tensor_list, net_info, l_index, is_first, is_training, trainable, seed)
            elif layer == "res":
                tensor = exe_res_layer(tensor, layer_o, tensor_list)
            elif layer == "max_pool":
                tensor = exe_max_pool_layer(tensor, layer_o)
            elif layer == "avg_pool":
                tensor = exe_avg_pool_layer(tensor, layer_o)
            elif layer == "resize":
                tensor = exe_resize_layer(tensor, layer_o)
            elif layer == "concat":
                tensor = exe_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "g_concat":
                tensor = exe_global_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "reshape":
                tensor = exe_reshape_layer(tensor, layer_o)
            elif layer == "clip":
                tensor = exe_clip_layer(tensor, layer_o)
            elif layer == "sigmoid":
                tensor = exe_sigmoid_layer(tensor)
            elif layer == "softmax":
                tensor = exe_softmax_layer(tensor)
            elif layer == "squeeze":
                tensor = exe_squeeze_layer(tensor, layer_o)
            elif layer == "abs":
                tensor = exe_abs_layer(tensor)
            elif layer == "tanh":
                tensor = exe_tanh_layer(tensor)
            elif layer == "inv_tanh":
                tensor = exe_inv_tanh_layer(tensor)
            elif layer == "add":
                tensor = exe_add_layer(tensor, layer_o)
            elif layer == "mul":
                tensor = exe_mul_layer(tensor, layer_o)
            elif layer == "reduce_mean":
                tensor = exe_reduce_mean_layer(tensor, layer_o)
            elif layer == "null":
                tensor = exe_null_layer(tensor)
            elif layer == "selu":
                tensor = exe_selu_layer(tensor)
            else:
                assert False, 'Error layer name = %s' % layer
            tensor_list.append(tensor)

            if is_first:
                info = '[%s][%4d] : (%s)'% (layer_name_format % layer, tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list()))
                if 'index' in layer_o:
                    info = info + ', use index [%4d] : (%s)' % (layer_o['index'], ', '.join('%4d' % (-1 if v is None else v) for v in tensor_list[layer_o['index']].get_shape().as_list()))
                net_info.architecture_log.append(info)
                if FLAGS['mode_use_debug']:
                    print(info)

    return tensor

def model(net_info, tensor, global_tensor, is_training, act_o, dilation_rate, is_first=False):
    tensor_list = [tensor, global_tensor]
    if net_info.name == "netD":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o, dilation_rate)
        result = tensor_list[-1]
    elif net_info.name == "netG":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o, dilation_rate)
        result = tensor_list[-1]
    elif (net_info.name).startswith('netG'):
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o, dilation_rate)
        result = tensor_list[-1]
    else:
        assert False, 'net_info.name ERROR = %s' % net_info.name
    return result, tensor_list
