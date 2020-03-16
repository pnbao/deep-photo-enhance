import os
import tensorflow as tf

trained_checkpoint_prefix = 'LPGAN/model/27.500-new.ckpt'
export_dir = os.path.join('models', '27500') # IMPORTANT: each model folder must be named '0', '1', ... Otherwise it will fail!

with tf.compat.v1.Session() as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    loader.restore(sess, trained_checkpoint_prefix)
    
    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess, ["train", "serve"], strip_default_attrs=True)
    builder.save()
