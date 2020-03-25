import os
import tensorflow as tf
from LPGAN import DATA_HDR, MODEL_HDR

netG_act_o = dict(size=1, index=0)

test_df = DATA_HDR.DataFlow()
netG = DATA_HDR.NetInfo('netG-%d' % DATA_HDR.FLAGS['num_exp_HDR'], test_df)
with tf.name_scope(netG.name):
    with tf.compat.v1.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.compat.v1.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_test_output1, netG_test_list = MODEL_HDR.model(netG, test_df.input1, test_df.input2, False, netG_act_o, None, is_first=True)
            netG_test_gfeature1 = netG_test_list[25]
            print("netG_test_gfeature1 ", netG_test_gfeature1)
            print("netG_test_output1 ", netG_test_output1)

saver = tf.compat.v1.train.Saver(var_list=netG.weights, max_to_keep=None)   
sess_config = tf.compat.v1.ConfigProto(log_device_placement=False)
sess = tf.compat.v1.Session(config=sess_config)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())
saver.restore(sess, DATA_HDR.FLAGS['load_model_path_new'])

output_node_names = ['netG-999/netG-999_var_scope/netG-999_var_scopeA/netG-999_3/Add']

frozen_graph_def = tf.graph_util.convert_variables_to_constants(
   sess,
   tf.get_default_graph().as_graph_def(),
   output_node_names)

#Save the frozen graph
with open('frozen_999.pb', 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
