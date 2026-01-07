import logging
import os
import sys
import torch
import torch.nn.functional as F

from datetime import datetime
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint 

torch.manual_seed(0)

class Trainer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(
            self.args.log_dir,
            f"{timestamp}_epochs_{self.args.epochs}"
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.output_dir)
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'training.log'),
            level=logging.DEBUG
        )

        self.align_loss = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.recon_loss = torch.nn.MSELoss().to(self.args.device)

    def info_nce_loss(self, features):
        labels = torch.arange(self.args.batch_size)
        labels = labels.repeat_interleave(2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        save_config_file(self.output_dir, self.args)

        n_iter = 0
        logging.info(f"Start training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        best_loss = float('inf')
        for epoch_counter in tqdm(range(self.args.epochs), desc="Training epochs"):
            epoch_loss = 0.0
            num_batches = 0

            for rgb_images, depth_images in train_loader:
                if depth_images.shape[1] == 1:
                    depth_images = depth_images.repeat(1, 3, 1, 1)

                rgb_images = rgb_images.to(self.args.device)
                depth_images = depth_images.to(self.args.device)

                with autocast(
                    device_type=self.args.device.type,
                    enabled=self.args.fp16_precision
                ):
                    rgb_features = self.model(rgb_images)
                    depth_features = self.model(depth_images)

                    features = torch.zeros(
                        2 * self.args.batch_size,
                        self.args.out_dim,
                        device=self.args.device
                    )
                    features[0::2] = rgb_features
                    features[1::2] = depth_features

                    logits, labels = self.info_nce_loss(features)

                    pred_depth = self.model.forward_with_depth(rgb_features)
                    align_loss = self.align_loss(logits, labels)
                    recon_loss = self.recon_loss(pred_depth, depth_images[:, 0:1, :, :])
                    loss = (
                        torch.exp(-self.model.log_sigma_align) * align_loss +
                        torch.exp(-self.model.log_sigma_recon) * recon_loss +
                        self.model.log_sigma_align + self.model.log_sigma_recon
                    )

                epoch_loss += loss.item()
                num_batches += 1

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], n_iter)
                    self.writer.add_scalar(
                        'learning_rate',
                        self.scheduler.get_last_lr()[0],
                        n_iter
                    )

                n_iter += 1

            if epoch_counter >= 10:
                self.scheduler.step()

            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            logging.info(
                f"Epoch: {epoch_counter}\t"
                f"Avg Loss: {avg_epoch_loss:.4f}\t"
                f"Top1 accuracy: {top1[0]:.2f}"
            )

            if epoch_counter % self.args.save_every == 0 or epoch_counter == self.args.epochs - 1:
                checkpoint_path = os.path.join(
                    self.output_dir,
                    f'checkpoint_epoch_{epoch_counter:04d}.pth'
                )

                save_checkpoint(
                    {
                        'epoch': epoch_counter,
                        'arch': self.args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'loss': avg_epoch_loss,
                    },
                    is_best=(avg_epoch_loss < best_loss),
                    filename=checkpoint_path
                )

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss

                logging.info(f"Checkpoint saved at {checkpoint_path}")

        final_model_path = os.path.join(self.output_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        logging.info(f"Final model saved at {final_model_path}")

        self.writer.close()
        logging.info("Training has finished.")
