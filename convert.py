import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import NetworkImageNet
from genotypes import PNASNet
from operations import *
from utils import preprocess_for_eval

import sys
import os
sys.path.append('../PNASNet.TF')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
from pnasnet import build_pnasnet_large, pnasnet_large_arg_scope
slim = tf.contrib.slim


class ConvertPNASNet(object):

  def __init__(self):
    self.image = Image.open('data/cat.jpg')
    self.read_tf_weight()
    self.write_pytorch_weight()

  def read_tf_weight(self):
    self.weight_dict = {}
    image_ph = tf.placeholder(tf.uint8, (None, None, 3))
    image_proc = preprocess_for_eval(image_ph, 323, 323)
    with slim.arg_scope(pnasnet_large_arg_scope()):
      logits, end_points = build_pnasnet_large(
          tf.expand_dims(image_proc, 0), num_classes=1001, is_training=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ckpt_restorer = tf.train.Saver()
    ckpt_restorer.restore(sess, '../PNASNet.TF/data/model.ckpt')

    weight_keys = [var.name[:-2] for var in tf.global_variables()]
    weight_vals = sess.run(tf.global_variables())
    for weight_key, weight_val in zip(weight_keys, weight_vals):
      self.weight_dict[weight_key] = weight_val

    self.tf_logits, self.tf_end_points, self.tf_image_proc = sess.run(
        [logits, end_points, image_proc], feed_dict={image_ph: self.image})

  def write_pytorch_weight(self):
    model = NetworkImageNet(216, 1001, 12, False, PNASNet)
    model.drop_path_prob = 0
    model.eval()

    self.used_keys = []
    self.convert_conv(model.conv0, 'conv0/weights')
    self.convert_bn(model.conv0_bn, 'conv0_bn/gamma', 'conv0_bn/beta',
        'conv0_bn/moving_mean', 'conv0_bn/moving_variance')
    self.convert_cell(model.stem1, 'cell_stem_0/')
    self.convert_cell(model.stem2, 'cell_stem_1/')

    for i in range(12):
      self.convert_cell(model.cells[i], 'cell_{}/'.format(i))
    
    self.convert_fc(model.classifier, 'final_layer/FC/weights',
        'final_layer/FC/biases')

    print('Conversion complete!')
    print('Check 1: whether all TF variables are used...')
    assert len(self.weight_dict) == len(self.used_keys)
    print('Pass!')

    model = model.cuda()
    image = self.tf_image_proc.transpose((2, 0, 1))
    image = Variable(self.Tensor(image)).cuda()
    logits, _ = model(image.unsqueeze(0))
    self.pytorch_logits = logits.data.cpu().numpy()

    print('Check 2: whether logits have small diff...')
    assert np.max(np.abs(self.tf_logits - self.pytorch_logits)) < 1e-5
    print('Pass!')

    model_path = 'data/PNASNet-5_Large.pth'
    torch.save(model.state_dict(), model_path)
    print('PyTorch model saved to {}'.format(model_path))

  def convert_cell(self, cell, name):
    # cell.preprocess0
    assert isinstance(cell.preprocess0, FactorizedReduce) or isinstance(cell.preprocess0, ReLUConvBN) or isinstance(cell.preprocess0, Identity)
    if isinstance(cell.preprocess0, FactorizedReduce):
      self.convert_conv(cell.preprocess0.conv_1, name + 'path1_conv/weights')
      self.convert_conv(cell.preprocess0.conv_2, name + 'path2_conv/weights')
      self.convert_bn(cell.preprocess0.bn, name + 'final_path_bn/gamma',
          name + 'final_path_bn/beta', name + 'final_path_bn/moving_mean',
          name + 'final_path_bn/moving_variance')
    else:
      if name + 'prev_1x1/weights' in self.weight_dict:
        self.convert_conv(cell.preprocess0.op[1], name + 'prev_1x1/weights')
        self.convert_bn(cell.preprocess0.op[2], name + 'prev_bn/gamma',
            name + 'prev_bn/beta', name + 'prev_bn/moving_mean',
            name + 'prev_bn/moving_variance')
      # else preprocess0 is Identity or = preprocess1; do nothing

    # cell.preprocess1
    assert isinstance(cell.preprocess1, ReLUConvBN)
    self.convert_conv(cell.preprocess1.op[1], name + '1x1/weights')
    self.convert_bn(cell.preprocess1.op[2], name + 'beginning_bn/gamma',
        name + 'beginning_bn/beta', name + 'beginning_bn/moving_mean',
        name + 'beginning_bn/moving_variance')

    # cell._ops
    for i in range(len(cell._ops)):
      side = 'left/' if i % 2 == 0 else 'right/'
      prefix = name + 'comb_iter_{}/'.format(i // 2) + side
      if isinstance(cell._ops[i], SepConv):
        suffix = '{0}x{0}'.format(cell._ops[i].op[1].kernel_size[0])
        
        self.convert_conv(cell._ops[i].op[1], 
            prefix + 'separable_' + suffix + '_1/depthwise_weights', sep=True)
        self.convert_conv(cell._ops[i].op[2],
            prefix + 'separable_' + suffix + '_1/pointwise_weights', sep=False)
        self.convert_bn(cell._ops[i].op[3],
            prefix + 'bn_sep_' + suffix + '_1/gamma',
            prefix + 'bn_sep_' + suffix + '_1/beta',
            prefix + 'bn_sep_' + suffix + '_1/moving_mean',
            prefix + 'bn_sep_' + suffix + '_1/moving_variance')
        self.convert_conv(cell._ops[i].op[5],
            prefix + 'separable_' + suffix + '_2/depthwise_weights', sep=True)
        self.convert_conv(cell._ops[i].op[6],
            prefix + 'separable_' + suffix + '_2/pointwise_weights', sep=False)
        self.convert_bn(cell._ops[i].op[7],
            prefix + 'bn_sep_' + suffix + '_2/gamma',
            prefix + 'bn_sep_' + suffix + '_2/beta',
            prefix + 'bn_sep_' + suffix + '_2/moving_mean',
            prefix + 'bn_sep_' + suffix + '_2/moving_variance')
      elif isinstance(cell._ops[i], ReLUConvBN):
        # skip_connect with stride > 1
        self.convert_conv(cell._ops[i].op[1], prefix + '1x1/weights')
        self.convert_bn(cell._ops[i].op[2],
            prefix + 'bn_1/gamma', prefix + 'bn_1/beta',
            prefix + 'bn_1/moving_mean', prefix + 'bn_1/moving_variance')
      elif isinstance(cell._ops[i], nn.Sequential):
        # max_pool or avg_pool with C_in != C_out
        self.convert_conv(cell._ops[i][1], prefix + '1x1/weights')
        self.convert_bn(cell._ops[i][2], 
            prefix + 'bn_1/gamma', prefix + 'bn_1/beta',
            prefix + 'bn_1/moving_mean', prefix + 'bn_1/moving_variance')

  def convert_conv(self, conv2d, weights_key, sep=False):
    weights = self.weight_dict[weights_key]
    if sep:
      # TF: [filter_height, filter_width, in_channels, channel_multiplier]
      # TF: [1, 1, channel_multiplier * in_channels, channel_multiplier]
      # PyTorch: [out_channels, in_channels // groups, *kernel_size]
      weights = np.transpose(weights, (2, 3, 0, 1))
    else:
      # TF: [filter_height, filter_width, in_channels, out_channels]
      # PyTorch: [out_channels, in_channels, *kernel_size]
      weights = np.transpose(weights, (3, 2, 0, 1))
    assert conv2d.weight.shape == self.Param(weights).shape, '{0} vs {1}'.format(conv2d.weight.shape, self.Param(weights).shape)
    conv2d.weight = self.Param(weights)
    self.used_keys += [weights_key]

  def convert_bn(self, bn, gamma_key, beta_key, moving_mean_key, moving_var_key):
    gamma = self.weight_dict[gamma_key]
    beta = self.weight_dict[beta_key]
    moving_mean = self.weight_dict[moving_mean_key]
    moving_var = self.weight_dict[moving_var_key]
    assert bn.weight.shape == self.Param(gamma).shape
    assert bn.bias.shape == self.Param(beta).shape
    assert bn.running_mean.shape == self.Tensor(moving_mean).shape
    assert bn.running_var.shape == self.Tensor(moving_var).shape
    bn.weight = self.Param(gamma)
    bn.bias = self.Param(beta)
    bn.running_mean = self.Tensor(moving_mean)
    bn.running_var = self.Tensor(moving_var)
    self.used_keys += [gamma_key, beta_key, moving_mean_key, moving_var_key]

  def convert_fc(self, fc, weights_key, biases_key):
    weights = self.weight_dict[weights_key]
    biases = self.weight_dict[biases_key]
    weights = np.transpose(weights)
    assert fc.weight.shape == self.Param(weights).shape
    assert fc.bias.shape == self.Param(biases).shape
    fc.weight = self.Param(weights)
    fc.bias = self.Param(biases)
    self.used_keys += [weights_key, biases_key]

  def Param(self, x):
    return torch.nn.Parameter(torch.from_numpy(x))

  def Tensor(self, x):
    return torch.from_numpy(x)


if __name__ == '__main__':
  ConvertPNASNet()
