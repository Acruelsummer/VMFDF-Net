"""Based one CycleGAN project: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.cuda.set_per_process_memory_fraction(0.5)  # 设置为GPU内存的50%
from thop import profile
import torch
torch.cuda.empty_cache()


# def calculate_flops(model, input_size=None):
#     """计算FLOPs"""
#     if input_size is None:
#         # 根据您的数据集设置默认尺寸
#         input_size = (1, 3, 256, 256)  # 这是CFD数据集的典型尺寸
    
#     device = next(model.parameters()).device
#     dummy_input = torch.randn(input_size).to(device)
    
#     flops, params = profile(model.netWD, inputs=(dummy_input,), verbose=False)
    
#     flops_g = flops / 1e9
#     params_m = params / 1e6
    
#     print("=" * 50)
#     print("FLOPs ANALYSIS REPORT")
#     print("=" * 50)
#     print(f"Input size: {input_size}")
#     print(f"FLOPs: {flops_g:.2f} G")
#     print(f"Parameters: {params_m:.2f} M")
#     print("=" * 50)
    
#     return flops_g, params_m
def calculate_flops(model, input_size=None, gpu_id=None):
    import torch
    from thop import profile
    import copy

    # choose device
    device = torch.device(f'cuda:{gpu_id}') if gpu_id is not None else next(model.parameters()).device

    # 复制 netWD（避免改变 DataParallel 原模型）
    if hasattr(model, 'netWD'):
        net = model.netWD
    else:
        raise ValueError("model has no netWD")

    # 如果是 DataParallel，取 module
    if isinstance(net, torch.nn.DataParallel):
        net = net.module

    # 深复制，不破坏原模型
    net_copy = copy.deepcopy(net)

    # 放到目标设备
    net_copy.to(device)
    net_copy.eval()

    if input_size is None:
        input_size = (1, 3, 256, 256)

    dummy_input = torch.randn(input_size).to(device)

    with torch.no_grad():
        flops, params = profile(net_copy, inputs=(dummy_input,), verbose=False)

    return flops / 1e9, params / 1e6








if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.best_model = 'latest'
    # opt.testdata = 'DeepCrack'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options

    

    model.setup(opt)               # regular setup: load and print networks; create schedulers


    # # ==================== 插入FLOPs计算代码 ====================
    # print("开始计算模型FLOPs...")
    
    # # 获取一个真实的数据样本来确定输入尺寸
    # sample_data = next(iter(dataset))
    # model.set_input(sample_data)
    
    # # 获取实际的输入张量尺寸
    # if hasattr(model, 'image'):
    #     input_tensor = model.image
    #     input_size = input_tensor.shape
    #     print(f"检测到输入尺寸: {input_size}")
    # else:
    #     # 如果无法自动获取，使用默认尺寸
    #     input_size = (1, 3, 256, 256)
    #     print(f"使用默认输入尺寸: {input_size}")
    
    # # 计算FLOPs
    # try:
    #     flops_g, params_m = calculate_flops(model, input_size)
        
    #     # 保存FLOPs信息到文件
    #     flops_report_path = os.path.join(opt.results_dir, opt.name, 'flops_report.txt')
    #     os.makedirs(os.path.dirname(flops_report_path), exist_ok=True)
    #     with open(flops_report_path, 'w') as f:
    #         f.write("FLOPs Analysis Report\n")
    #         f.write("=" * 30 + "\n")
    #         f.write(f"Model: {opt.name}\n")
    #         f.write(f"Dataset: {opt.dataset}\n")
    #         f.write(f"Input size: {input_size}\n")
    #         f.write(f"FLOPs: {flops_g:.2f} G\n")
    #         f.write(f"Parameters: {params_m:.2f} M\n")
    #         f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    #     print(f"FLOPs报告已保存至: {flops_report_path}")
        
    # except Exception as e:
    #     print(f"FLOPs计算失败: {e}")
    #     print("继续执行测试...")
    
    # print("=" * 50)
    # # ==================== FLOPs计算结束 ====================
    # 设定用于分析的 gpu（你用 opt.gpu_ids[0]）
    gpu_for_profile = None
    if hasattr(opt, 'gpu_ids') and isinstance(opt.gpu_ids, (list, tuple)) and len(opt.gpu_ids) > 0:
        try:
            gpu_for_profile = int(opt.gpu_ids[0])
        except:
            gpu_for_profile = None
    
    # 确认模型处于 eval
    model.eval()
    
    # 使用 dataset 的第一个样本推断输入尺寸（如果可用）
    sample_data = next(iter(dataset))
    model.set_input(sample_data)
    if hasattr(model, 'image'):
        input_tensor = model.image
        input_size = tuple(input_tensor.shape)  # e.g. (1,3,H,W)
    else:
        input_size = (1, 3, 256, 256)
    
    try:
        flops_g, params_m = calculate_flops(model, input_size=input_size, gpu_id=gpu_for_profile)
        print(f"FLOPs: {flops_g:.2f} G, Params: {params_m:.2f} M")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
    



    
    # create a website
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")
    web_dir = os.path.join(opt.results_dir, opt.name,opt.dataset,'1','%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    # web_dir = os.path.join(opt.results_dir, opt.name,opt.dataset,str(opt.gpu_ids[0]),'%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    inference_times = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        model.set_input(data)  # unpack data from data loader
        start_time = time.time()
        model.test()           # run inference
        end_time = time.time()
        inference_time = end_time- start_time
        inference_times.append(inference_time)
        # print(f"inference time for 1 image: {inference_time} seconds")
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    average_inference_time = sum(inference_times)/len(inference_times) if inference_times else 0
    print(f"Average inference time for {len(inference_times)} images: {average_inference_time:.4f} seconds")

