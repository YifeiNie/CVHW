import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from dataset import Dataset_
from model import Model
import os
from tqdm import tqdm
from torchvision.utils import save_image, make_grid

def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--rgb-path", default="./datasets/RGB", help="Path to RGB images")
    parser.add_argument("--depth-path", default="./datasets/Depth", help="Path to Depth images (optional)")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--arch", default="resnet18", help="Backbone architecture")
    parser.add_argument("--out-dim", type=int, default=128, help="Projector output dim")

    parser.add_argument("--infer-output-dir", type=str, default="./outputs", help="Directory to save reconstructed depth")
    parser.add_argument("--gpu-index", type=int, default=0)
    
    parser.add_argument(
        "--pretrained-model-path", 
        default="/home/nesc-gy/nyf/code/cvhw/runs/20260105_032016_epochs_1000/checkpoint_epoch_0999.pth",
        type=str, 
        help="Path to trained model checkpoint"
    )


    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_index}")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # Dataset
    dataset = Dataset_(args.img_size, args.rgb_path, args.depth_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Model
    model = Model(base_model=args.arch, out_dim=args.out_dim, img_res=(args.img_size, args.img_size))
    checkpoint = torch.load(args.pretrained_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    os.makedirs(args.infer_output_dir, exist_ok=True)

    cnt = 0
    with torch.no_grad():
        for idx, (rgb, depth) in enumerate(tqdm(loader)):
            if cnt % 10 != 0:
                continue
            rgb = rgb.to(device)
            depth = depth.to(device)

            # 预测 depth
            feat = model(rgb)
            pred_depth = model.forward_with_depth(feat)

            # 归一化到 [0,1]，方便保存为图片
            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            pred_depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)

            # 如果 depth 是单通道，把通道扩展成3通道方便拼图
            if depth_norm.shape[1] == 1:
                depth_norm = depth_norm.repeat(1, 3, 1, 1)
            if pred_depth_norm.shape[1] == 1:
                pred_depth_norm = pred_depth_norm.repeat(1, 3, 1, 1)

            # 拼接 RGB | GT Depth | Pred Depth
            grid = torch.cat([rgb_norm, depth_norm, pred_depth_norm], dim=3)  # 水平方向拼接 (dim=3是宽)

            save_path = os.path.join(args.infer_output_dir, f"comparison_{idx:04d}.png")
            save_image(grid, save_path)

    print(f"Inference done. Comparison images saved to {args.infer_output_dir}")


if __name__ == "__main__":
    main()
