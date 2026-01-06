import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from dataset import Dataset_
from model import Model
from trainer import Trainer

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="HW")

parser.add_argument(
    "-data",
    metavar="DIR",
    default="./datasets",
    help="path to dataset"
)

parser.add_argument(
    "-rgb-path",
    metavar="RGB_DIR",
    default="./datasets/RGB",
    help="path to RGB dataset"
)

parser.add_argument(
    "-depth-path",
    metavar="DEPTH_DIR",
    default="./datasets/Depth",
    help="path to Depth dataset"
)

parser.add_argument(
    "-dataset-name",
    default="uav",
    choices=["stl10", "cifar10", "uav"],
    help="dataset name"
)

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)"
)

parser.add_argument(
    "-j",
    "--workers",
    default=12,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)"
)

parser.add_argument(
    "--epochs",
    default=1000,
    type=int,
    metavar="N",
    help="number of total epochs to run"
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
         "batch size of all GPUs on the current node when "
         "using Data Parallel or Distributed Data Parallel"
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.0003,
    type=float,
    metavar="LR",
    dest="lr",
    help="initial learning rate"
)

parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    dest="weight_decay",
    help="weight decay (default: 1e-4)"
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training"
)

parser.add_argument(
    "--disable-cuda",
    action="store_true",
    help="Disable CUDA"
)

parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training"
)

parser.add_argument(
    "--out_dim",
    default=128,
    type=int,
    help="feature dimension (default: 128)"
)

parser.add_argument(
    "--log-dir",
    type=str,
    default="runs",
    help="Directory to save training logs and models"
)

parser.add_argument(
    "--log-every-n-steps",
    default=100,
    type=int,
    help="Log every n steps"
)

parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)"
)

parser.add_argument(
    "--n-views",
    default=2,
    type=int,
    metavar="N",
    help="Number of views for contrastive learning training"
)

parser.add_argument(
    "--gpu-index",
    default=0,
    type=int,
    help="Gpu index"
)

parser.add_argument(
    "--pretrained-model-path",
    default=None,
    help="Load pretrained model"
)

parser.add_argument(
    "--save-every",
    default=100,
    type=int,
    help="Save checkpoint"
)

parser.add_argument(
    "--img-size",
    type=int,
    nargs=2,
    default=(224, 224)
)

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    train_dataset = Dataset_(args.img_size, args.rgb_path, args.depth_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True
    )

    model = Model(base_model=args.arch, out_dim=args.out_dim, img_res=args.img_size)

    # Load pretrained weights (optional)
    if args.pretrained_model_path:
        checkpoint = torch.load(args.pretrained_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded pretrained model from {args.pretrained_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )

    #  It’s a no-op if the "gpu_index" argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        trainer = Trainer(
            model=model, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            args=args
        )
        trainer.train(train_loader)


if __name__ == "__main__":
    main()
