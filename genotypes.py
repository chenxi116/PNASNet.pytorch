from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PNASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 0),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 1),
    ('max_pool_3x3', 1),
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 1),
    ('sep_conv_3x3', 4),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('skip_connect', 1),
  ],
  reduce_concat = [2, 3, 4, 5, 6],
)

