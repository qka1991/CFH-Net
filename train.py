import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.util import confusion_matrix, getScores, tensor2labelim, tensor2im, print_current_losses
import numpy as np
import random
import torch
import cv2
from tensorboardX import SummaryWriter


if __name__ == '__main__':
    train_opt = TrainOptions().parse()

    np.random.seed(train_opt.seed)
    random.seed(train_opt.seed)
    torch.manual_seed(train_opt.seed)
    torch.cuda.manual_seed(train_opt.seed)

    train_data_loader = CreateDataLoader(train_opt)
    train_dataset = train_data_loader.load_data()
    train_dataset_size = len(train_data_loader)
    print('#training images = %d' % train_dataset_size)

    valid_opt = TrainOptions().parse()
    valid_opt.phase = 'val'
    valid_opt.batch_size = 1
    valid_opt.num_threads = 1
    valid_opt.serial_batches = True
    valid_opt.isTrain = False
    valid_data_loader = CreateDataLoader(valid_opt)
    valid_dataset = valid_data_loader.load_data()
    valid_dataset_size = len(valid_data_loader)
    print('#validation images = %d' % valid_dataset_size)

    writer = SummaryWriter()

    model = create_model(train_opt, train_dataset.dataset)
    model.setup(train_opt)
 ion"])
                t = (time.time() - iter_start_time) / train_opt.batch_size
                print_current_losses(epoch, epoch_iter, losses, t, t_data
            iter_data_time = time.time()

        mean_loss = np.mean(train_loss_iter)
        # One average training loss value in tensorboard in one epoch
        writer.add_scalar('train/mean_loss', mean_loss, epoch)

        palet_file = 'datasets/palette.txt'
        impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))
        tempDict = model.get_current_visuals()
        rgb = tensor2im(tempDict['rgb_image'])
        if train_opt.use_sne:
            another = tensor2im((tempDict['another_image']+1)/2)    # color normal images

        output = tensor2labelim(tempDict['output'], impalette)
        #image_numpy = np.concatenate((rgb, another, label, output), axis=1)
        image_numpy = np.concatenate((rgb, another), axis=1)
        image_numpy = image_numpy.astype(np.float64) / 255
        writer.add_image('Epoch' + str(epoch), image_numpy, dataformats='HWC')  # show training images in tensorboard

        print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.nepoch, time.time() - epoch_start_time))
        model.update_learning_rate()

        ### Evaluation on the validation set ###
        model.eval()
        valid_loss_iter = []
        epoch_iter = 0
        conf_mat = np.zeros((valid_dataset.dataset.num_labels, valid_dataset.dataset.num_labels), dtype=np.float)
        with torch.no_grad():
            for i, data in enumerate(valid_dataset):
                model.set_input(data)
                model.forward()
                model.get_loss()
                epoch_iter += valid_opt.batch_size
                gt = model.label.cpu().int().numpy()
                _, pred = torch.max(model.output.data.cpu(), 1)
                pred = pred.float().detach().int().numpy()

                # Resize images to the original size for evaluation
                image_size = model.get_image_oriSize()
                oriSize = (image_size[0].item(), image_size[1].item())
                gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                pred = np.expand_dims(cv2.resize(np.squeeze(pred, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)

           er, len(valid_dataset) * valid_opt.batch_size), end='\r')

        avg_valid_loss = torch.mean(torch.stack(valid_loss_iter))
        globalacc, pre, recall, F_score, iou = getScores(conf_mat)

        # Record performance on the validation set
        writer.add_scalar('valid/loss', avg_valid_loss, epoch)
        writer.add_scalar('valid/global_acc', globalacc, epoch)

        writer.add_scalar('valid/recall', recall, epoch)
        writer.add_scalar('valid/F_score', F_score, epoch)
        writer.add_scalar('valid/iou', iou, epoch)

        # Save the best model according to the F-score, and record corresponding epoch number in tensorboard
        if F_score > F_score_max:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('best')
            F_score_max = F_score
            writer.add_text('best model', str(epoch))
