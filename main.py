import argparse
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from models.VGG16 import VGG16
import numpy
from metann import Learner
from utils.digits_process_dataset import *
from FFTDataset import *

torch.manual_seed(0)
numpy.random.seed(0)

parser = argparse.ArgumentParser(description="Training on Digits")
parser.add_argument("--data_dir", default="data", type=str, help="dataset dir")
parser.add_argument(
    "--dataset", default="mnist", type=str, help="dataset mnist or cifar10"
)
parser.add_argument(
    "--num_iters", default=10001, type=int, help="number of total iterations to run"
)
parser.add_argument(
    "--start_iters",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=32, type=int, help="mini-batch size (default: 32)"
)
parser.add_argument(
    "--lr",
    "--min-learning-rate",
    default=0.0001,
    type=float,
    help="initial learning rate",
)
parser.add_argument(
    "--lr_max",
    "--adv-learning-rate",
    default=1,
    type=float,
    help="adversarial learning rate",
)

parser.add_argument("--K", default=3, type=int, help="number of augmented domains")
parser.add_argument(
    "--T_min", default=100, type=int, help="intervals between domain augmentation"
)
parser.add_argument(
    "--print-freq", "-p", default=1000, type=int, help="print frequency (default: 10)"
)
parser.add_argument(
    "--resume", default=None, type=str, help="path to saved checkpoint (default: none)"
)
parser.add_argument("--name", default="Digits", type=str, help="name of experiment")
parser.add_argument("--mode", default="train", type=str, help="train or test")
parser.add_argument("--GPU_ID", default=0, type=int, help="GPU_id")
parser.add_argument("--FFT_num", default=200, type=int, help="image number FFT generated")
parser.add_argument("--alpha", default=1.0, type=float, help="image alpha")


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_ID)

    exp_name = args.name

    kwargs = {"num_workers": 4}

    model = Learner(VGG16())
    model = model.cuda()
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["iter"]
            prec = checkpoint["prec"]
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (iter {})".format(
                    args.resume, checkpoint["iter"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.mode == "train":
        train(model, exp_name, kwargs)
        evaluation(model, args.data_dir, args.batch_size, kwargs)
    else:
        evaluation(model, args.data_dir, args.batch_size, kwargs)


def train(model, exp_name, kwargs):
    train_loader, val_loader = construct_datasets(
        args.data_dir, args.batch_size, kwargs
    )

    print("Training task model")

    criterion = nn.CrossEntropyLoss().cuda()
    mse_loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    only_virtual_test_images = []
    only_virtual_test_labels = []
    train_loader_iter = iter(train_loader)
    counter_k = 0

    for t in range(args.start_iters, args.num_iters): 

        batch_time = AverageMeter() 
        losses = AverageMeter() 
        top1 = AverageMeter() 
        end = time.time() 

        counter_k = 0 

        if t % args.T_min == 0 and t > 0: 
            aug_start_time = time.time()

            virtual_test_images = []
            virtual_test_labels = []
            origin_images, origin_labels = load_mnist(args.data_dir, split="test")
            for i in range(args.FFT_num):
                idx = random.randint(0, origin_images.shape[0] - 1)
                img = origin_images[idx]
                label = origin_labels[idx]
                image = createBackgroundPic(32, 32)
                img_o = np.array(img)
                img_s = np.array(image)
                post_transform = get_post_transform()
                img_s2o, img_o2s = colorful_spectrum_mix(img_o, img_s, alpha=args.alpha)
                img_s2o, img_o2s = post_transform(img_s2o), post_transform(img_o2s)
                img_s2o = np.array(img_s2o)
                img_s2o = img_s2o.astype(np.uint8)
                
                virtual_test_images = np.append(virtual_test_images, img_s2o)
                virtual_test_labels = np.append(virtual_test_labels, label)
    
                virtual_test_images = np.reshape(virtual_test_images, (-1, 3, 32, 32))
                virtual_test_labels = np.reshape(virtual_test_labels, (-1,))

            if counter_k == 0:
                only_virtual_test_images = np.copy(virtual_test_images)
                only_virtual_test_labels = np.copy(virtual_test_labels)
            else:
                only_virtual_test_images = np.concatenate(
                    [only_virtual_test_images, virtual_test_images]
                )
                only_virtual_test_labels = np.concatenate(
                    [only_virtual_test_labels, virtual_test_labels]
                )

            aug_size = len(only_virtual_test_labels)
            X_aug = torch.stack(
                [torch.from_numpy(only_virtual_test_images[i]) for i in range(aug_size)]
            )
            y_aug = torch.stack(
                [torch.from_numpy(np.asarray(i)) for i in only_virtual_test_labels]
            )
            aug_dataset = torch.utils.data.TensorDataset(X_aug, y_aug)
            aug_loader = torch.utils.data.DataLoader(
                aug_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                **kwargs
            )
            aug_loader_iter = iter(aug_loader)
            counter_k += 1
           
        model.train()
        try:
            input, target = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            input, target = next(train_loader_iter)

        input, target = (
            input.cuda(non_blocking=True).float(),
            target.cuda(non_blocking=True).long(),
        )
        params = list(model.parameters())
        output = model.functional(params, True, input)
        loss = criterion(output, target)

        if counter_k == 0:
            optimizer.zero_grad()
            loss.backward()
        else:
            grads = torch.autograd.grad(loss, params, create_graph=True)
            params = [
                (param - args.lr * grad).requires_grad_()
                for param, grad in zip(params, grads)
            ]
            try:
                input_b, target_b = next(aug_loader_iter)
            except:
                aug_loader_iter = iter(aug_loader)
                input_b, target_b = next(aug_loader_iter)

            input_b, target_b = (
                input_b.cuda(non_blocking=True),
                target_b.cuda(non_blocking=True).long(),
            )
            output_b = model.functional(params, True, input_b)
            loss_b = criterion(output_b, target_b)
            loss_combine = (loss + loss_b) / 2
            optimizer.zero_grad()
            loss_combine.backward()

        optimizer.step()
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        batch_time.update(time.time() - end)

        if t % args.print_freq == 0:
            print(
                "Iter: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    t, t, args.num_iters, batch_time=batch_time, loss=losses, top1=top1
                )
            )
            # evaluate on validation set per print_freq, compute acc on the whole val dataset
            prec1 = validate(val_loader, model)
            print("validation set acc", prec1)

            save_checkpoint(
                {
                    "iter": t + 1,
                    "state_dict": model.state_dict(),
                    "prec": prec1,
                },
                args.dataset,
                exp_name+"_"+str(t),
                filename = "iter: "+str(t)+"-T_min: "+str(args.T_min)+"-FFT_num: "+str(args.FFT_num)+"-Alpha: "+str(args.alpha)
            )


if __name__ == "__main__":
    main()
