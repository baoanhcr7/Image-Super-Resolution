import argparse
from math import log10
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_testing_set
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generating the Super Resolution Model')
    parser.add_argument('--upscale_factor', type=int, required=True, help="Upscale_factor by which increase resolution")
    parser.add_argument('--trainBatchSize', type=int, default=64, help='Training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='Testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='Number of Epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--cuda', action='store_true', help='To use Cuda')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='Random seed to use. Default=123')
    opt = parser.parse_args()
    
    print(opt)
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda else "cpu")
    print(device)
    print('===> Loading datasets\n')
    
    train_set = get_training_set(opt.upscale_factor)
    test_set = get_test_set(opt.upscale_factor)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.trainBatchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    
    print('===> Building model\n')
    model = Net(upscale_factor=opt.upscale_factor).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print("Criterion ==>",criterion)
    print("Optimizer ==>",optimizer)
    print()
    avg_loss_list = []
    avg_psnr_list = []
    
    def train(epoch):
        epoch_loss = 0
        for interation, batch in enumerate(training_data_loader, 1):
            
            input, target = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(input), target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, interation, len(training_data_loader), loss.item()))
        avg_loss_list.append(epoch_loss/len(training_data_loader))
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    
    def test():
        avg_psnr = 0
        with torch.no_grad():
            for batch in testing_data_loader:
                input, target = batch[0].to(device), batch[1].to(device)

                prediction = model(input)
                mse = criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        avg_psnr_list.append(avg_psnr / len(testing_data_loader))
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
        
    def checkpoint(epoch):
        model_out_path = "model_epoch_{}_upscale_{}.pth".format(epoch, opt.upscale_factor)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
        if(epoch%10 == 0):
            checkpoint(epoch)

    fig, ax = plt.subplots()
    ax.plot(epochs_list,avg_loss_list,'r--')
    plt.savefig("Average_Loss_Curve.png")
    fig, ax = plt.subplots()
    ax.plot(epochs_list, avg_psnr_list,'b--')
    plt.savefig("Average_PSNR_values.png")
        