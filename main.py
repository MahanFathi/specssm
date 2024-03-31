
from typing import Sequence
from absl import app, flags, logging

import jax
import gin

from train import Trainer

flags.DEFINE_list(
    "gin_search_paths",
    ["./configs"], "List of paths where the Gin config files are located.")
flags.DEFINE_multi_string(
    "gin_file", ["base.gin"], "List of Gin config files.")
flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings.")

FLAGS = flags.FLAGS


def parse_gin_configuration():
    """Load and parse Gin configuration from command-line flags."""
    for gin_file_path in FLAGS.gin_search_paths:
        logging.info("Added Gin search path %s", gin_file_path)
        gin.add_config_file_search_path(gin_file_path)
    for gin_file in FLAGS.gin_file:
        logging.info("Loading Gin config file %s", gin_file)
    if FLAGS.gin_param:
        for gin_param in FLAGS.gin_param:
            logging.info("Overriding Gin param %s", gin_param)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    parse_gin_configuration()

    # minimal preparation for argument passing goes here

    Trainer().train()


if __name__=='__main__':
    app.run(main)
