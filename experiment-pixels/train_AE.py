# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np
import scipy.integrate
import torch, argparse
from tqdm import tqdm
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from hnn import PixelHNN
from nn_models import MLPAutoencoder
from data import get_dataset
solve_ivp = scipy.integrate.solve_ivp

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2*28**2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--latent_dim', default=2, type=int, help='latent dimension of autoencoder')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--input_noise', default=0.0, type=float, help='std of noise added to HNN inputs')
    parser.add_argument('--batch_size', default=200, type=int, help='batch size')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=10000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--name', default='pixels', type=str, help='either "real" or "sim" data')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=os.path.join(THIS_DIR, 'tar_pkl'), type=str, help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()

'''The loss for this model is a bit complicated, so we'll
    define it in a separate function for clarity.'''
def ae_loss(x, x_next, model, return_scalar=True):
  # encode pixel space -> latent dimension
  z = model.encode(x)
  z_next = model.encode(x_next)

  # autoencoder loss
  x_hat = model.decode(z)
  ae_loss = ((x - x_hat)**2).mean(1)

  # canonical coordinate loss
  # -> makes latent space look like (x, v) coordinates
  w, dw = z.split(1,1)
  w_next, _ = z_next.split(1,1)
  cc_loss = ((dw-(w_next - w))**2).mean(1)

  # sum losses and take a gradient step
  loss = ae_loss + 10 * cc_loss
  if return_scalar:
    return loss.mean(), ae_loss.mean(), cc_loss.mean()
  return loss, ae_loss, cc_loss

def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  model = MLPAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim, nonlinearity='relu')
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-5)

  # get dataset
  data = get_dataset('pendulum', args.save_dir)

  x = torch.tensor( data['pixels'], dtype=torch.float32)
  test_x = torch.tensor( data['test_pixels'], dtype=torch.float32)
  next_x = torch.tensor( data['next_pixels'], dtype=torch.float32)
  test_next_x = torch.tensor( data['test_next_pixels'], dtype=torch.float32)

  # vanilla ae train loo
  stats = {'train_loss': [], 'test_loss': []}
  for step in tqdm(range(args.total_steps+1), disable=args.verbose):
    
    # train step
    ixs = torch.randperm(x.shape[0])[:args.batch_size]
    loss = ae_loss(x[ixs], next_x[ixs], model)[0]
    loss.backward() ; optim.step() ; optim.zero_grad()

    stats['train_loss'].append(loss.item())
    if args.verbose and step % args.print_every == 0:
      # run validation
      test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
      test_loss = ae_loss(test_x[test_ixs], test_next_x[test_ixs], model)[0]
      stats['test_loss'].append(test_loss.item())

      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

  train_dist, ae_dist, cc_dist = ae_loss(x, next_x, model, return_scalar=False)
  test_dist = ae_loss(test_x, test_next_x, model, return_scalar=False)[0]
  print('Final train loss {:.4e} +/- {:.4e}'.format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0])))
  print('Final ae loss {:.4e} +/- {:.4e}'.format(ae_dist.mean().item(), ae_dist.std().item()/np.sqrt(train_dist.shape[0])))
  print('Final cc loss {:.4e} +/- {:.4e}'.format(cc_dist.mean().item(), cc_dist.std().item()/np.sqrt(train_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    path = '{}/{}-pixels-{}.tar'.format(args.save_dir, args.name, 'AE')
    torch.save(model.state_dict(), path)