import logging
import pprint

from absl import app

import jax

from ml_collections import config_dict
from ml_collections import config_flags


cfg = config_dict.ConfigDict()

cfg.key = 0

cfg.energy_fn = 'least_square'
cfg.target_energy_fn = 'least_square'

cfg.optimizer = 'adam'

_CONFIG = config_flags.DEFINE_config_dict('my_config', cfg)


def main(_):
  pp = pprint.PrettyPrinter(indent=2, compact=True)
  logging.info(f"JAX running on {jax.devices()[0].platform.upper()}")
    

if __name__ == '__main__':
  app.run(main)
