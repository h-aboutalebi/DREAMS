import argparse
import random

#from https://github.com/mathiaszinnen/focal_loss_torch
# use pip install focal_loss_torch to install focal loss
from focal_loss.focal_loss import FocalLoss 
import logging
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from dataset.dino2_dataset import CustomDataset
from models.dino_v2 import DinoV2
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, device, test_loader, writer, epoch):
    model.eval()
    true_labels = []
    predictions = []
    progress_bar = tqdm(test_loader, total=len(
        test_loader), desc=f'Epoch {epoch}')

    with torch.no_grad():
        for (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.view_as(target).cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    accuracy, precision, recall, f1 = precision_recall_fscore_support(
        true_labels, predictions, average='macro')
    
    if f1 is None:
        f1 = 0
    if recall is None:
        recall = 0
    if precision is None:
        precision = 0
    if accuracy is None:
        accuracy = 0
    logging.info(f'\nTest set: Accuracy: {accuracy:.4f}, Precision: {
                 precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n')
    

    # Log metrics
    writer.add_scalar('Accuracy', accuracy, epoch)
    writer.add_scalar('Precision', precision, epoch)
    writer.add_scalar('Recall', recall, epoch)
    writer.add_scalar('F1', f1, epoch)

    return accuracy, precision, recall, f1


def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    normalizer = torch.nn.Softmax(dim=-1) 
    # Wrap `train_loader` with `tqdm` for a progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f'Epoch {epoch}')
    focal_loss = FocalLoss(gamma=0.7) #increase gamma to focus nore on hard examples

    for batch_idx, (data, target) in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = focal_loss(normalizer(output), target) 
        loss.backward()
        optimizer.step()

        # Optional: Calculate total loss and accuracy if you want to display them in the progress bar
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Update the progress bar
        progress_bar.set_postfix({'Loss': f'{loss.item():.6f}', 'Acc': f'{
                                 100. * correct / total:.2f}%'})

        # Log to TensorBoard
        if batch_idx % 10 == 0:
            writer.add_scalar('Training loss', loss.item(),
                              epoch * len(train_loader) + batch_idx)

    logging.info(f'End of Epoch {epoch}, Average Loss: {
                 total_loss / len(train_loader)}, Accuracy: {100. * correct / total}%')


def print_config(config):
    logging.info("Configuration of the running experiment:")
    config_dict = vars(config)
    logging.info(json.dumps(config_dict, indent=4, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(
        description='DREAMS Project - DinoV2 Training Script')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='batch size (default: 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='batch size (default: 4)')
    parser.add_argument('--train_path', type=str,
                        default="/storage/disk1/hossein/retina/train/train", help='path to the train dataset')
    parser.add_argument('--test_path', type=str,
                        default="/storage/disk1/hossein/retina/train/test", help='path to the test dataset')
    parser.add_argument('--label_path', type=str,
                        default="/storage/disk1/hossein/retina/train/trainLabels.csv", help='path to the labels dataset')
    parser.add_argument('--model_name', type=str,
                        default='DinoV2', help='model name')
    parser.add_argument('--output_path', type=str,
                        default='/home/hossein/results_dream', help='path for saving trained models')
    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, str(args.lr))
    try:
        os.makedirs(args.output_path)
    except OSError:
        logging.info("Creation of the directory %s failed" % args.output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    writer = SummaryWriter(f"{args.output_path}/runs")

    # Configure logging to write to a file and also to stderr
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    best_accuracy = 0

    # Format for our loglines
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # File handler for writing logs to a file
    file_handler = logging.FileHandler(f"{args.output_path}/training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler for writing logs to stderr (the terminal)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    print_config(args)  # Set the logging level

    # Initialize datasets
    # Add appropriate arguments for your dataset class
    train_dataset = CustomDataset(args.train_path, args.label_path, "/home/hossein/results_dream/label_distribution_train.png")
    # Add appropriate arguments for your dataset class
    test_dataset = CustomDataset(args.test_path, args.label_path, "/home/hossein/results_dream/label_distribution_test.png")
    logging.info(f'Train dataset size: {len(train_dataset)}')
    logging.info(f'Test dataset size: {len(test_dataset)}')

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = DinoV2().to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) #cosine annealing

    # Training loop
    for epoch in range(1, args.epochs+1):  # 10 epochs for example
        train(model, device, train_loader, optimizer, epoch, writer=writer)
        scheduler.step()
        accuracy, _, _, _ = evaluate(model, device, test_loader, writer=writer, epoch=epoch)
        # Add validation/testing here
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"{args.output_path}/{args.model_name}_best.pth")
    
    # Load the best model
    model.load_state_dict(torch.load(f"{args.output_path}/{args.model_name}_best.pth"))
    
    # Evaluate the model on the test set
    evaluate(model, device, test_loader, writer=writer, epoch=args.epochs)

    writer.close()


if __name__ == '__main__':
    main()
