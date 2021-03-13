import os
import tarfile
import numpy as np
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.models import (load_model, model_from_json)
from tensorflow.python.framework import graph_io


def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen


import sys
sys.path.append(os.path.join(os.getcwd(), "model-optimizer"))

from mo.utils.versions_checker import check_python_version  # pylint: disable=no-name-in-module


DEFAULT_TAR_NAME = 'model.tar.gz'

"""
  - input_dir
  - output_dir
  - model_name
  - input_model
  - input_model_as_text
"""

def main_tf():
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    """Obtain parser from OpenVino and append additional arguments

    Note on store_store: If flag is provided, it is true regardless of whatever user value provided
    """
    from mo.utils.cli_parser import get_tf_cli_parser
    parser = get_tf_cli_parser()
    parser.add_argument("--input_dir", help="Directory for inputs such as tar.gz, h5, pb", type=str,
                                       default="/opt/ml/processsing/input_data")
    args, _ = parser.parse_known_args()

    """Obtain .h5 from unzipping the tar.gz file
    """
    filepath = os.path.join(args.input_dir, DEFAULT_TAR_NAME)
    if filepath.endswith('tar.gz'):
        tar = tarfile.open(filepath, 'r:gz')
        tar.extractall(path=args.input_dir)
        tar.close()

    """Convert h5 to protobuf
    """
    tf.compat.v1.disable_eager_execution()
    # Toggle off learning parameters
    K.set_learning_phase(False)

    model = load_model(os.path.join(args.input_dir, args.model_name + '.h5'))
    model.summary()

    session = tf.compat.v1.keras.backend.get_session()
    freeze_graph(
        graph=session.graph,
        session=session,
        output=[out.op.name for out in model.outputs],
        save_pb_dir=args.input_dir,
        save_pb_name=args.model_name + '.pb',
        save_pb_as_text=args.input_model_is_text)

    """Update the parser value based on procedures above
    """
    parser.set_defaults(input_model=os.path.join(args.input_dir, args.model_name+'.pb'))

    from mo.main import main as optimizer_main
    sys.exit(optimizer_main(parser, 'tf'))


if __name__ == '__main__':
    main_tf()
