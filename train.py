"""Based one CycleGAN project: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from pytorchtools import EarlyStopping
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    best_loss=float('inf')
    losses = {'total': float('inf')}
    early_stopping = EarlyStopping(patience=50,verbose=True)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # 当前 epoch 中的训练迭代次数
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # if save_result:
                    # 保存图片
                    # 注释之后display_current_results只起到一个保存图片的作用
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            # print(total_iters % opt.print_freq == 0,'if condition')False if condition
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        # print(losses,'loss')OrderedDict([('side', 0.16406187415122986), ('fused', 0.03523821383714676), 
        # ('total', 0.19930008053779602)]) loss
        # print(losses['total'])0.08352002501487732
        # print(type(model))
        # print(hasattr(model, 'state_dict'))
        # print(losses,'losses')
        if losses['total']<best_loss:
            best_loss = losses['total']
            best_model_saved = True
            print(f"New best model found at epoch {epoch} with loss {best_loss}. Saving model.")
            model.save_networks('best') 
        early_stopping(losses['total'],model)
        if early_stopping.early_stop:
            if not best_model_saved:
                print(f"Early stopping triggered. Saving the best model at epoch {epoch} with loss {best_loss}.")
                model.save_networks('best') 
            print('early stopping')
            break
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_summary())  # 打印显存使用概况