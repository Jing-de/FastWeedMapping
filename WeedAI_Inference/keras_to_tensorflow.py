#!/usr/bin/env python
"""
Copyright (c) 2019, by the Authors: Amir H. Abdi
This script is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model files
to the freezed .pb tensorflow weight file. The resultant TensorFlow model
holds both the model architecture and its associated weights.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json, model_from_yaml

K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', None, 'Path to the input model.')
flags.DEFINE_string('input_model_json', None, 'Path to the input model '
                                              'architecture in json format.')
flags.DEFINE_string('input_model_yaml', None, 'Path to the input model '
                                              'architecture in yaml format.')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', False,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_string('output_nodes_prefix', None,
                    'If set, the output nodes will be renamed to '
                    '`output_nodes_prefix`+i, where `i` will numerate the '
                    'number of of output nodes of the network.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('channels_first', False,
                     'Whether channels are the first dimension of a tensor. '
                     'The default is TensorFlow behaviour where channels are '
                     'the last dimension.')
flags.DEFINE_boolean('output_meta_ckpt', False,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')

flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')


def load_model(input_model_path, input_json_path=None, input_yaml_path=None):
    from predict_on_full_images import transform_highres_center_model
    from utils import generate_setup_dict
    
    if not Path(input_model_path).exists():
        raise FileNotFoundError(
            'Model file `{}` does not exist.'.format(input_model_path))
        
    model_name = input_model_path.split('/')[-1].split('.h5')[0]  
    if model_name.startswith('random_seed'):
        model_name = model_name[14:]    
        
    model = tf.keras.models.load_model(input_model_path)
    setup = generate_setup_dict()
    num_classes = 6
    print('\n##############################################################')
    print('The model being transformed is {}.'.format(model_name))
    print('##############################################################')
    flex_model = transform_highres_center_model(model, num_classes, model_name, setup)
    print('The model has been transformed, returning it to the function `load_model(..)`.')
    return flex_model


def main(args):
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)

    if FLAGS.channels_first:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    flex_model = load_model(FLAGS.input_model, FLAGS.input_model_json, FLAGS.input_model_yaml)

    orig_output_node_names = [node.op.name for node in flex_model.outputs]
    if FLAGS.output_nodes_prefix:
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                FLAGS.output_nodes_prefix, i)
            pred[i] = tf.identity(flex_model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names
    logging.info('Converted output node names are: %s',
                 str(converted_output_node_names))

    sess = K.get_session()
    if FLAGS.output_meta_ckpt:
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))

    if FLAGS.save_graph_def:
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))

    if FLAGS.quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    app.run(main)