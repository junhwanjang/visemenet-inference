# Be Aware: Only supports in Tensorflow 1.1.0
import sys
sys.path.append("./VisemeNet_tensorflow/")

import tensorflow as tf
from src.model import model
from src.utl.load_param import model_dir


def freeze_visemenet_graph(out_path):
    model_name='pretrain_biwi'

    with tf.Graph().as_default() as graph:

        init, net1_optim, net2_optim, all_optim, x, x_face_id, y_landmark, \
        y_phoneme, y_lipS, y_maya_param, dropout, cost, tensorboard_op, pred, \
        clear_op, inc_op, avg, batch_size_placeholder, phase = model()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        max_to_keep = 20
        saver = tf.train.Saver(max_to_keep=max_to_keep)

        OLD_CHECKPOINT_FILE = model_dir + model_name + '/' + model_name +'.ckpt'

        saver.restore(sess, OLD_CHECKPOINT_FILE)
        print("Model loaded: " + model_dir + model_name)

        ## For debugging
        # node_names = [node.name for node in sess.graph_def.node]
        # for node_name in node_names:
        #     if node_name.find("net2_output") != -1:
        #         print(node_name)
        
        output_names = ['net2_output/add_1', 'net2_output/add_4', 'net2_output/add_6']
        frozen_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)

        with tf.gfile.GFile(out_path, 'w') as f:
            f.write(frozen_def.SerializeToString())
        
        print("Save ProtoBuffer in {}".format(out_path))


if __name__ == '__main__':
    out_path = "./visemenet_frozen.pb"
    freeze_visemenet_graph(out_path=out_path)
