import os
import tensorflow as tf
from LPGAN import DATA, MODEL

with tf.gfile.GFile("output_frozen_999.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    for op in graph.get_operations():
        print(op.name)
   
#with tf.compat.v1.Session() as sess:
    # Restore from checkpoint
    #loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    #sess.run(tf.compat.v1.global_variables_initializer())
    #sess.run(tf.compat.v1.local_variables_initializer())
    #loader.restore(sess, trained_checkpoint_prefix)
    
    #constant_values = {}
    #constant_ops = [op for op in sess.graph.get_operations()] # if op.type == "Const"
    #for constant_op in constant_ops:
    #    print("BBB ", constant_op.name)
    # Export checkpoint to SavedModel
    #builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    #builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
    #builder.save()
    #output_node_names = ['netG-999/netG-999_var_scope/netG-999_var_scopeA/netG-999_3/Add_48']

    #frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    #    sess,
    #    tf.get_default_graph().as_graph_def(),
    #    output_node_names)

    #Save the frozen graph
    #with open('output_frozen_999.pb', 'wb') as f:
    #  f.write(frozen_graph_def.SerializeToString())
