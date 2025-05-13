import torch
import numpy as np
from PIL import Image
import math
import os

def lcm(a, b):
    """
    Function to calculate the least common multiple(最小公倍数)
    当 a 和 b 都不为零时，计算 a 和 b 的绝对值乘积，然后除以它们的最大公约数（GCD，Greatest Common Divisor），从而得到最小公倍数（LCM）。
    :param a:
    :param b:
    :return:
    """
    return abs(a * b) / math.gcd(a, b) if a and b else 0

def mkdirs(paths : list | str):
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    elif isinstance(paths, str):
        mkdir(paths)
    else:
        raise TypeError

def mkdir(path : str):
    if not os.path.exists(path):
        os.makedirs(path)

def save_network(network, save_dir, network_label, epoch_label):
    save_filename = 'epoch_%s_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.to(device='cuda')
    else:
        network.to(device='mps:0')

# helper loading function that can be used by subclasses
def load_network(network, save_dir, network_label, epoch_label):
    save_filename = 'epoch_%s_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.isfile(save_path):
        raise FileNotFoundError('%s not exists yet!' % save_path)
    else:
        network.load_state_dict(torch.load(save_path))
    print("INFO: Load network: %s, at epoch %s" % (epoch_label, network_label))

def tensor2array(image : torch.tensor, _type=np.uint8, normalize=True) -> np.ndarray:
    """
    Converts a Tensor into a Numpy array
    :param image: the input tensor, or list of tensors
    :param _type: the desired type of the converted numpy array
    :param normalize:
    :return:
    """
    if isinstance(image, list):
        image_numpy = []
        for i in range(len(image)):
            image_numpy.append(tensor2array(image[i], _type, normalize))
        return np.array(image_numpy)

    image_numpy = image.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = image_numpy * 255 if not normalize else (image_numpy + 1) / 2.0 * 255
    image_numpy = np.clip(image_numpy, 0, 255)
    # Todo: Not sure why, to be tested
    if image.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(_type)

def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")