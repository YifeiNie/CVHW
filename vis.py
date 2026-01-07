import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
@torch.no_grad()
def extract_features(model, dataloader, device, max_batches=20):
    model.eval()

    rgb_feats = []
    depth_feats = []

    for i, (rgb, depth) in enumerate(dataloader):
        if i >= max_batches:
            break

        if depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1)

        rgb = rgb.to(device)
        depth = depth.to(device)

        f_rgb = model(rgb)       # (B, out_dim)
        f_depth = model(depth)   # (B, out_dim)

        rgb_feats.append(f_rgb.cpu())
        depth_feats.append(f_depth.cpu())

    rgb_feats = torch.cat(rgb_feats, dim=0)
    depth_feats = torch.cat(depth_feats, dim=0)

    return rgb_feats.numpy(), depth_feats.numpy()


def tsne_visualization(rgb_feats, depth_feats, save_path=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    features = np.concatenate([rgb_feats, depth_feats], axis=0)
    labels = np.array([0]*len(rgb_feats) + [1]*len(depth_feats))

    # 1. 标准化
    features = StandardScaler().fit_transform(features)

    # 2. PCA 降维
    features_pca = PCA(n_components=50).fit_transform(features)

    # 3. t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42,
        init='pca'
    )
    features_2d = tsne.fit_transform(features_pca)

    # 4. 分别取 RGB/Depth
    rgb_2d = features_2d[labels == 0]
    depth_2d = features_2d[labels == 1]

    # 5. 绘图
    plt.figure(figsize=(6,6))
    plt.scatter(rgb_2d[:,0], rgb_2d[:,1], c='red', s=8, alpha=0.7, label='RGB')
    plt.scatter(depth_2d[:,0], depth_2d[:,1], c='green', s=8, alpha=0.7, label='Depth')
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE of RGB and Depth Features")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
