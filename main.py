import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import torch
import torchvision.datasets as datasets
from torch.autograd import Variable
from model import NetworkImageNet
from genotypes import PNASNet
from utils import preprocess_for_eval

parser = argparse.ArgumentParser()
parser.add_argument('--valdir', type=str, default='data/val',
                    help='path to ImageNet val folder')
parser.add_argument('--image_size', type=int, default=331,
                    help='image size')
parser.add_argument('--num_conv_filters', type=int, default=216,
                    help='number of filters')
parser.add_argument('--num_classes', type=int, default=1001,
                    help='number of categories')
parser.add_argument('--num_cells', type=int, default=12,
                    help='number of cells')


def main():
  args = parser.parse_args()
  assert torch.cuda.is_available()

  image_ph = tf.placeholder(tf.uint8, (None, None, 3))
  image_proc = preprocess_for_eval(image_ph, args.image_size, args.image_size)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  model = NetworkImageNet(args.num_conv_filters, args.num_classes,
                          args.num_cells, False, PNASNet)
  model.drop_path_prob = 0
  model.eval()
  model.load_state_dict(torch.load('data/PNASNet-5_Large.pth'))
  model = model.cuda()

  c1, c5 = 0, 0
  val_dataset = datasets.ImageFolder(args.valdir)
  for i, (image, label) in enumerate(val_dataset):
    tf_image_proc = sess.run(image_proc, feed_dict={image_ph: image})
    image = torch.from_numpy(tf_image_proc.transpose((2, 0, 1)))
    image = Variable(image).cuda()
    logits, _ = model(image.unsqueeze(0))
    top5 = logits.data.cpu().numpy().squeeze().argsort()[::-1][:5]
    top1 = top5[0]
    if label + 1 == top1:
      c1 += 1
    if label + 1 in top5:
      c5 += 1
    print('Test: [{0}/{1}]\t'
          'Prec@1 {2:.3f}\t'
          'Prec@5 {3:.3f}\t'.format(
          i + 1, len(val_dataset), c1 / (i + 1.), c5 / (i + 1.)))


if __name__ == '__main__':
  main()
