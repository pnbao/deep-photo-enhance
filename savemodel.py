import os
import tensorflow as tf
from LPGAN import DATA, MODEL

trained_checkpoint_prefix = 'LPGAN/model/27.500-new.ckpt'
export_dir = os.path.join('models', '27500') # IMPORTANT: each model folder must be named '0', '1', ... Otherwise it will fail!
netG_act_o = dict(size=1, index=0)

test_df = DATA.DataFlow()
netG = DATA.NetInfo('netG-604', test_df)

with tf.name_scope(netG.name):
    with tf.compat.v1.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.compat.v1.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_test_output1, netG_test_list = MODEL.model(netG, test_df.input1, test_df.input2, False, netG_act_o, None, is_first=True)
            netG_test_gfeature1 = netG_test_list[25]
            scopeA.reuse_variables()
            netG_test_dilation_list = []
            for dilation in range(10):
                netG_test_dilation_output, _ = MODEL.model(netG, test_df.input1, test_df.input2, False, netG_act_o, dilation+1)
                netG_test_dilation_list.append(netG_test_dilation_output)


loader = tf.compat.v1.train.Saver(var_list=netG.weights, max_to_keep=None)

with tf.compat.v1.Session() as sess:
    # Restore from checkpoint
    # loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    loader.restore(sess, trained_checkpoint_prefix)
    
    # Export checkpoint to SavedModel
    # builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    # builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
    # builder.save()
    output_node_names = ['netG/netG_var_scope/netG_var_scopeA/netG_3_1/Add']

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names)

    # Save the frozen graph
    with open('frozen_model.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
