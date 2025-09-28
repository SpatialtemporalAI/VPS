"""
MegaLoc + DINOv2 训练脚本
基于MegaLoc论文的多任务学习策略
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
from typing import Dict, List

# 导入自定义模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from vps.models.megaloc_dinov2 import MegaLocDINOv2, MegaLocLoss

class MegaLocTrainer:
    """MegaLoc训练器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # 初始化模型
        self.model = MegaLocDINOv2(**config['model']).to(self.device)
        
        # 损失函数
        self.criterion = MegaLocLoss(**config['loss'])
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['lr'] * 0.01
        )
        
        # 初始化wandb
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                config=config,
                name=config['logging']['run_name']
            )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_losses = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到设备
            images = batch['images'].to(self.device)
            targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in batch.items() if k != 'images'}
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(images, task="all")
            
            # 计算损失
            losses = self.criterion(predictions, targets)
            total_loss = losses['total_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # 累积损失
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # 记录到wandb
            if self.config['logging']['use_wandb'] and batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': total_loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        # 计算平均损失
        avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_losses = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['images'].to(self.device)
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in batch.items() if k != 'images'}
                
                predictions = self.model(images, task="all")
                losses = self.criterion(predictions, targets)
                
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0
                    total_losses[key] += value.item()
        
        avg_losses = {k: v / len(val_loader) for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            # 训练
            train_losses = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_losses = self.validate(val_loader, epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'train/epoch_loss': train_losses['total_loss'],
                    'val/epoch_loss': val_losses['total_loss'],
                    'epoch': epoch
                })
            
            # 保存最佳模型
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                self.save_checkpoint(epoch, best=True)
            
            # 定期保存检查点
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        save_dir = Path(self.config['training']['checkpoint_dir'])
        save_dir.mkdir(exist_ok=True)
        
        if best:
            save_path = save_dir / 'best_model.pth'
        else:
            save_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")


def create_data_loaders(config: dict):
    """创建数据加载器"""
    # 这里需要根据您的具体数据格式实现
    # 示例数据加载器结构
    
    class MegaLocDataset(torch.utils.data.Dataset):
        """MegaLoc数据集"""
        def __init__(self, data_dir: str, split: str = 'train'):
            self.data_dir = Path(data_dir)
            self.split = split
            # 加载数据索引
            self.samples = self._load_samples()
        
        def _load_samples(self):
            # 实现数据样本加载逻辑
            # 返回包含图像路径、标签等信息的列表
            pass
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            # 实现单个样本的加载逻辑
            # 返回包含images、vpr_labels、landmark_labels、pose_labels的字典
            pass
    
    # 创建数据集
    train_dataset = MegaLocDataset(config['data']['train_dir'], 'train')
    val_dataset = MegaLocDataset(config['data']['val_dir'], 'val')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config)
    
    # 创建训练器
    trainer = MegaLocTrainer(config)
    
    # 恢复训练（如果指定）
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"恢复训练从: {args.resume}")
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    print("训练完成！")


if __name__ == "__main__":
    main()
