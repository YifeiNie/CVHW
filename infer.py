import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from dataset import Dataset_
from model import Model
import os
from vis import *
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
        default="/home/nesc-gy/nyf/code/cvhw/runs/20260107_071454_epochs_1000/checkpoint_epoch_0999.pth",
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
    rgb_feat_list = []
    depth_feat_list = []

    with torch.no_grad():
        for idx, (rgb, depth) in enumerate(tqdm(loader)):
            # if idx % 2 != 0:
            #     continue

            rgb = rgb.to(device)
            depth = depth.to(device) 

            feat_rgb = model(rgb)
            feat_depth = model(depth.repeat(1, 3, 1, 1)) 

            pred_depth = model.forward_with_depth(feat_rgb)
            rgb_feat_list.append(feat_rgb.cpu())
            depth_feat_list.append(feat_depth.cpu())

            rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            pred_depth_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)

            if depth_norm.shape[1] == 1:
                depth_norm = depth_norm.repeat(1, 3, 1, 1)
            if pred_depth_norm.shape[1] == 1:
                pred_depth_norm = pred_depth_norm.repeat(1, 3, 1, 1)

            if idx % 50 == 0:
                grid = torch.cat([rgb_norm, depth_norm, pred_depth_norm], dim=3)
                save_path = os.path.join(args.infer_output_dir, f"comparison_{idx:04d}.png")
                save_image(grid, save_path)

    print(f"Inference done. Comparison images saved to {args.infer_output_dir}")

    # ===== loop 结束后，统一跑一次 t-SNE =====
    feat_rgb_all = torch.cat(rgb_feat_list, dim=0)      # (N, C)
    feat_depth_all = torch.cat(depth_feat_list, dim=0)  # (N, C)

    print("Collected RGB feats:", feat_rgb_all.shape)
    print("Collected Depth feats:", feat_depth_all.shape)

    tsne_visualization(
        feat_rgb_all,
        feat_depth_all,
        save_path="tsne_rgb_depth.png"
    )


if __name__ == "__main__":
    main()
