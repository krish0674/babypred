import argparse
from .trainer import train_model
from posture.OpenPoseKeras.pose_init import pose_init

def main():
    parser = argparse.ArgumentParser(description="Train baby sleep position classifier")

    parser.add_argument('--base_dir', type=str, default='/kaggle/input/baby-sleep-dataset',
                        help='Path to dataset root directory (with train/val/test folders)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--pose_model', type=str,help='Model pose')

    args = parser.parse_args()
    
    pose_init(args.pose_model)

    train_model(
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

if __name__ == '__main__':
    main()
