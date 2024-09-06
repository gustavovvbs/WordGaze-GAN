import torch 
import argparse 

from solver import Solver

def main(args):
    solver = Solver(root=args.root, batch_size=args.batch_size, num_epoch=args.num_epoch, lr=args.lr)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        print('Invalid mode')
        exit(1)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='gestures_data.json', help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
   
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2')
    parser.add_argument('--log_step', type=int, default=100, help='Log step')
    parser.add_argument('--sample_step', type=int, default=500, help='Sample step')
    parser.add_argument('--model_save_step', type=int, default=1000, help='Model save step')
    parser.add_argument('--resume_step', type=int, default=0, help='Resume step')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Resume epoch')
    parser.add_argument('--mode', type=str, default='train', help='Mode')

    args = parser.parse_args()
    main(args)