import time
import torch
import torch.nn as nn
# import layers
import torch.nn.functional as F
import torch.optim as optim

from math import sqrt
from tqdm.notebook import trange

from utils.reconstructed import loss_inverting_gt, loss_cosine_similarity
from utils.nb_utils import calculate_iqa

from .cross_skip import skip
import numpy as np
# import defense as defense
from tqdm import tqdm
torch.manual_seed(15560471273492565302)

class trainer():
    """
        The attack process only involves a training process
    """
    def __init__ (self,config,model,parameters,batch_size):
        self.config = config
        self.model = model
        self.pameraters = parameters
        self.b_size = batch_size #(50000*32*32*3)

    def attack_training(self,dy_dx,gt_label,img_size, means_vars, device, image_batch):
        #  Conduct the attacks many times with different images
        for num_exp in range(self.config['restarts']): #开始训练
            # if self.config['defense_strategy'] is None: #联邦学习对梯度攻击的防御策略
            #     print('No defense applied.')
            # else:
            #     defense_param = None
            #     if self.config['defense_strategy'] == 'noise':
            #         defense_param = self.config['noise_param']
            #         dy_dx = defense.additive_noise(dy_dx, std=defense_param)
            #     elif self.config['defense_strategy'] == 'clipping':
            #         defense_param = self.config['clipping_param']
            #         dy_dx = defense.gradient_clipping(dy_dx, bound=defense_param)
            #     elif self.config['defense_strategy'] == 'compression':
            #         defense_param = self.config['compression_param']
            #         dy_dx = defense.gradient_compression(dy_dx, percentage=defense_param)
            #     elif self.config['defense_strategy'] == 'representation':
            #         defense_param = self.config['representation_param']
            #         dy_dx = defense.perturb_representation(dy_dx, self.model, dataloader, pruning_rate=defense_param)
            #     else:
            #         raise NotImplementedError("Invalid defense method!")
            #     print('Defense applied: {} w/ {}.'.format(self.config['defense_strategy'], defense_param))
            self.model.zero_grad()
            original_dy_dx = list((_.detach().clone() for _ in dy_dx)) #分离并复制梯度值,转为列表格式

            print('-----------------EXP{} ATTACK BEGIN----------------'.format(num_exp+1))
            print(f'seed: {torch.initial_seed()}')
            start_time = time.time()
            attack_worker = CI_attacker(self.config,self.pameraters,self.b_size,img_size)
            reconstructed_image, G_losses = attack_worker.reconstructed_gt(original_dy_dx, gt_label,self.model, means_vars, device, image_batch) # Conduct the attack by minimizing the loss between dummy gradients and original gradients
            print('-----------------EXP{} ATTACK END----------------'.format(num_exp+1))

            #-- Compute Image Quality --#
            # print('-----------------EXP{} IMAGE ASSESSMENT BEGIN----------------'.format(num_exp+1))
            # avg_score[str(num_exp)][self.attack['method']] = calculate_iqa(dataloader,experimental_results[self.attack['method']],self.config)   #重构图像与原始图像dataloader作比较，计算250组图像的4个iqa  # Calculate the iqa score
            # reconstructed_image[str(num_exp)]['image'][self.attack['method']] ={}  
            # for iqa in ['ssim','fsim']: #遍历两个iqa：‘ssim’和‘fsim’
            #     reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa] = {
            #     'timeline' :[experimental_results[self.attack['method']]]                                                                                    # Take a snap shot of reconstruction attack 
            #         }
            #     # In some case  that the higest score is not the last attack iteration, we will save the highest iqa index and image
            #     highest_iqa_at_idx = avg_score[str(num_exp)][self.attack['method']][iqa]['score'].index(max(avg_score[str(num_exp)][self.attack['method']][iqa]['score'])) #找到250组图像中的iqa最大的那一组次序
            #     if highest_iqa_at_idx == self.config['num_epochs']-1:  
            #         reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_idx'] = -1                                             # The last reconstructed images are the highest IQA
            #     else:
            #         print('Hightest IQA at other index',highest_iqa_at_idx)
            #         reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_idx'] = highest_iqa_at_idx     ## The last reconstructed images are not the highest IQA
            #         reconstructed_image[str(num_exp)]['image'][self.attack['method']][iqa]['highest_score_img'] =  experimental_results[self.attack['method']][highest_iqa_at_idx] #保存8次训练中两个指标下最优的重构图像结果及其次序
            #     # experimental_results[attack['method']] = []
            # print('-----------------EXP{} IMAGE ASSESSMENT END----------------'.format(num_exp+1))

            print('Choosing optimal result ...')
            G_losses = torch.tensor(G_losses)
            scores = G_losses[torch.isfinite(G_losses)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            x_optimal = reconstructed_image[optimal_index]
            # x_optimal = reconstructed_image

        print(f'Total time: {time.time() - start_time}.')
        return x_optimal


class CI_attacker():
    """
        Return the attacker object for reconstruction attack based on the hyper parameter setting
        .reconstructed_gt(): It will reconstruct the input images based on the updated gradients.
        
    """
    def __init__ (self,config,parameters,b_size,img_size):
        self.config = config
        self.parameters = parameters
        # if self.config["dst"]=="cifar10":
        #     self.img_res = 32 #cifar10数据集图片3*32*32
        # elif self.config["dst"]=="imagenet": #imagenet数据集图片3*256*256
        #     self.img_res = 256
        # elif self.config["dst"]=="mydata": #eurosat数据集图片3*64*64
        #     self.img_res = 256
        # else:
        #     print("Please set the image resolution manually")
        self.b_size = b_size
        self.img_size = img_size

    def over_parameterization(self):
        """ This function sets the channel number of GI_Net,
          so that the generator is over-parameterized
        """
        batch_size = self.config["b_size"]
        
        image_params = batch_size * (self.img_res**2)
        channel = 32
        model_params = sum(p.numel() for p in Generator(in_channel=channel).parameters())
        while model_params <= 4*image_params:
            channel = channel*2
            model_params = sum(p.numel() for p in Generator(in_channel=channel).parameters())
        return channel


    def reconstructed_gt(self,original_gt,original_label,model,means_vars,device,image_batch):
        # Set the configuration based on variable "config()"
        # nz              = self.config['nz']
        tv_value        = self.config['total_variation']
        num_epochs      = self.config['max_iterations']
        lr_decay        = self.config['lr_decay']
        model           = model.to(device)
    
        # print('Configuration Parameters :',self.config)

        # if not self.config['architecture_search']:
        #     channel = self.over_parameterization() #过参数化网络
        #     print("The channel number is {}".format(channel))
        # Define Generator and noise        
        # if self.config['architecture_search']:
        #     self.noise = torch.randn(b_size, 3, self.img_res, self.img_res, device=device) #标准正态分布随机生成固定噪声tensor(4*3*32*32) (对应论文中的z0)
        #     with open(self.config['model_search_space_path']) as model_search_space_file:
        #         best_skip_index = None
        #         best_model_index = None
        #         best_search_metric = 999
        #         best_model = None
        #         skip_indexs = model_search_space_file.readlines() #加载网络结构搜索空间 skip_indexs = list(5000) 对应论文：M={G1,G2,...G5000}
        #         for model_index, skip_index in enumerate(skip_indexs[:1]): #model_index = 0~4999
        #             torch.cuda.empty_cache() #释放缓存空间
        #             skip_index = np.array(list(map(int, skip_index.strip()))) #skip_indexs的一组‘01’字符串转为整型列表list(25)
        #             skip_index = np.reshape(skip_index, (5, 5)) #skip_index列表再转为5*5矩阵 对应论文中U-Net网络encoder层和decoder层之间的跳跃连接矩阵A
        #             try:
        #                 trial_netG = skip( #根据搜索空间调整U-Net卷积神经网络中的skip connect架构
        #                     model_index=(model_index%300),
        #                     skip_index=skip_index,
        #                     num_input_channels=3,
        #                     num_output_channels=3,
        #                     num_channels_down=[128] * 5,
        #                     num_channels_up=[128] * 5,
        #                     num_channels_skip=[4] * 5,
        #                     upsample_mode='bilinear',
        #                     downsample_mode='stride',
        #                     need_sigmoid=True,
        #                     need_bias=True,
        #                     pad='constant',
        #                     act_fun='LeakyReLU'
        #                 ).to(device)
        #             except Exception as e:
        #                 print(f'model_index: {model_index}, skip_index: {skip_index.reshape((25))}, cannot create')
        #                 print(e)
        #                 continue
        #             try:
        #                 model.zero_grad()
        #                 trial_fake  = trial_netG(self.noise).to(device) #对应论文：G_r(z0;φ)
        #                 trial_fake_output = model(trial_fake) #对应论文：F(G_r(z0;φ))
        #                 trial_criterion = nn.CrossEntropyLoss().to(device)
        #                 trial_dummy_loss = trial_criterion(trial_fake_output, original_label)
        #                 trial_fake_dy_dx = torch.autograd.grad(trial_dummy_loss, model.parameters(), create_graph=True) #计算单个模型架构G_r(G1~G5000)下的初始梯度

        #                 trial_fake_dy_dx = [grad for grad in trial_fake_dy_dx]
        #                 if self.config['defense_strategy'] is not None:
        #                     if self.config['defense_strategy'] == 'noise':
        #                         pass
        #                     elif self.config['defense_strategy'] == 'clipping':
        #                         trial_fake_dy_dx = defense.gradient_clipping(trial_fake_dy_dx, bound=self.config['clipping_param'])
        #                     elif self.config['defense_strategy'] == 'compression':
        #                         trial_fake_dy_dx = defense.gradient_compression(trial_fake_dy_dx, percentage=self.config['compression_param'])
        #                     elif self.config['defense_strategy'] == 'representation':
        #                         mask = (original_gt[-2][0] != 0)
        #                         trial_fake_dy_dx[-2] = trial_fake_dy_dx[-2] * mask

        #                 search_metric = loss_cosine_similarity(original_gt, trial_fake_dy_dx).cpu().item() #计算单个模型架构G_r(G1~G5000)下的初始梯度匹配损失(利用余弦相似度函数)  对应论文：L_grad(G_r)
        #                 if search_metric < best_search_metric:
        #                     best_model_index = model_index
        #                     best_skip_index = skip_index
        #                     best_search_metric = search_metric
        #                     best_model = copy.deepcopy(trial_netG) #选取梯度损失最小的模型架构best_model
        #                 print(f'model_index: {model_index}, skip_index: {skip_index.reshape((25))}, search_metric: {search_metric}, best_model_index: {best_model_index}, best_skip_index: {best_skip_index.reshape((25))}, best_search_metric: {best_search_metric}')
        #                 del trial_netG
        #                 del trial_fake
        #                 del trial_fake_output
        #                 del trial_criterion
        #                 del trial_dummy_loss
        #                 del trial_fake_dy_dx
        #                 del search_metric
        #                 torch.cuda.empty_cache()
        #             except Exception as e:
        #                 print(f'model_index: {model_index}, skip_index: {skip_index.reshape((25))}, other error')
        #                 print(e)
        #                 continue
        self.noise = torch.randn(self.b_size, *self.img_size, device=device)
        self.netG = skip( 
                    num_input_channels=3,
                    num_output_channels=3,
                    num_channels_down=[128] * 5,
                    num_channels_up=[128] * 5,
                    num_channels_skip=[4] * 5,
                    upsample_mode='bilinear',
                    downsample_mode='stride',
                    need_sigmoid=True,
                    need_bias=True,
                    pad='constant',
                    act_fun='LeakyReLU'
                    ).to(device)
        # self.netG.load_state_dict(torch.load('netG_64_vb18_rsi.pth'))
        # else:
        #     self.netG = Generator(image_res=self.img_res,in_channel=channel).to(device)
        #     self.noise = torch.randn(b_size,nz, device=device)
        optim_parameters = self.netG.parameters()
        optim_eps = self.parameters['eps']
        # optim_eps = self.parameters['eps'] + self.parameters['multiplier']

        optimizerG = optim.Adam(optim_parameters,lr=self.config['lr'])
        optimizerE = optim.Adam(optim_eps,lr=self.config['lr'] * 100)
        # self.noise.requires_grad_()
        # optimizerE = optim.Adam(optim_eps + [self.noise],lr=self.config['lr'] * 100)
        
        if lr_decay:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizerE,milestones=[num_epochs // 2.667, num_epochs// 1.6,num_epochs // 1.142], gamma=0.1)   # 3/8 5/8 7/8

        # Lists to keep track of progress
        G_losses = []
        image_recon = []
        loss_mse = {'iter':[],'grad':[],'img':[]}
        # Start the reconstrcution attack
        try:
            for iters in tqdm(range(num_epochs)): 
                optimizerG.zero_grad()
                optimizerE.zero_grad()
                model.zero_grad()
                fake = self.netG(self.noise).to(device)
                # fake = self.noise
                # if self.config['boxed']:
                #     fake.data = torch.max(torch.min(fake, (1 - dm) / ds), -dm / ds) 
                fake_output = model(fake)
                criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
                #Calculating the dummy gradient
                # dummy_loss = criterion(fake_output,original_label)
                dummy_loss = criterion(fake_output,original_label) + model.loss()

                fake_dy_dx = torch.autograd.grad(dummy_loss, self.parameters['models'], create_graph=True) 

                fake_dy_dx = [grad for grad in fake_dy_dx]
                # if self.config['defense_strategy'] is not None:
                #     if self.config['defense_strategy'] == 'noise':
                #         pass
                #     elif self.config['defense_strategy'] == 'clipping':
                #         fake_dy_dx = defense.gradient_clipping(fake_dy_dx, bound=self.config['clipping_param'])
                #     elif self.config['defense_strategy'] == 'compression':
                #         fake_dy_dx = defense.gradient_compression(fake_dy_dx, percentage=self.config['compression_param'])
                #     elif self.config['defense_strategy'] == 'representation':
                #         mask = (original_gt[-2][0] != 0)
                #         fake_dy_dx[-2] = fake_dy_dx[-2] * mask

                #Calculating the loss between the original gradients and dummy gradients

                errG = loss_inverting_gt(original_gt,fake_dy_dx,fake,tv_value)

                if (iters+1)%10 == 0:
                    loss_mse['iter'].append(iters+1)
                    # loss_mse['grad'].append(errG.item())
                    loss_mse['grad'].append(MSE(fake_dy_dx,original_gt).item())
                    loss_mse['img'].append(F.mse_loss(fake,image_batch).item())

                if self.config['kl_value'] > 0:
                    errG += self.config['kl_value'] * KL(self.parameters['eps']).item()
                    
                dummy_dict = model.state_dict()
                dummy_means_vars = BNlayers_mean_var(dummy_dict)
                errG += 0.01 * rBN(dummy_means_vars, means_vars)

                errG.backward()
                
                # Changing the value of gradient to sign only. 
                if self.config['signed']:
                    for layer in self.netG.parameters():
                        if layer.grad is not None:
                            layer.grad.sign_()
                    # self.noise.grad.sign_()
                # Update G
                optimizerG.step()
                optimizerE.step()
                if self.config['lr_decay']:
                    scheduler.step()
                
                if (iters+1)%500==0 or iters == 0:
                    print(f"loss:{errG}")
                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    # Save the reconstructed image on each to variable "image_recon"
                    image_recon.append(fake.detach())

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iters}!')
            pass
        
        # np.save('vb34_xbn.npy',loss_mse)
        # torch.save(self.netG.state_dict(), 'netG_4_vb18_nwpu.pth')
        return image_recon,G_losses 
        # return fake.detach(),G_losses

def MSE(x,y):
    loss = 0
    for m,n in zip(x,y):
        loss += F.mse_loss(m,n)
    return loss/len(x)

def KL(eps):
        kl = 0
        for ep in eps:
            std = torch.std(ep)
            mean = torch.mean(ep)
            kl += 0.5 * (-2*torch.log(std) + std**2 + mean**2 - 1)
        return kl

def rBN(mv,mv0):
    # rbn = 0
    # for l in range(len(x)):
    #     dm = []
    #     dv = []
    #     for k in range(len(x[l])):
    #         dm.append(torch.mean(x[l][k])-torch.mean(x0[l][k]))
    #         dv.append(torch.var(x[l][k])-torch.var(x0[l][k]))
    #     rbn += torch.norm(torch.stack(dm)) + torch.norm(torch.stack(dv))
    dm = dv = 0
    m,v = mv
    m0,v0 = mv0
    for i in range(len(m)):
        dm += torch.norm(m[i]-m0[i]).item()
        dv += torch.norm(v[i]-v0[i]).item()
    return dm + dv

def BNlayers_mean_var(dict):
    means = []
    vars = []
    for key in dict.keys():
        if '.running_mean' in key:
            means.append(dict[key])
        elif '.running_var' in key:
            vars.append(dict[key])
    return (means,vars)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super().__init__()
        convs = [layers.SNConv2d(in_channel, out_channel, kernel_size, padding=padding)]
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        convs.append(layers.SNConv2d(out_channel, out_channel, kernel_size, padding=padding))
        convs.append(nn.BatchNorm2d(out_channel))
        convs.append(nn.LeakyReLU(0.1))
        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out    

def upscale(feat):
    return F.interpolate(feat, scale_factor=2) #, mode="bilinear") 


class Generator(nn.Module):
    def __init__(self, image_res=32, input_code_dim=128, in_channel=256, tanh=True):
        super().__init__()
        self.image_res = image_res
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.1))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm) 
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1)#, pixel_norm=pixel_norm)
        if self.image_res >= 64:
            self.progression_64 = ConvBlock(in_channel, in_channel//2, 3, 1) # pixel_norm=pixel_norm)
        if self.image_res >= 128:
            self.progression_128 = ConvBlock(in_channel//2, in_channel//4, 3, 1) #, pixel_norm=pixel_norm)
        if self.image_res >= 256:
            self.progression_256 = ConvBlock(in_channel//4, in_channel//4, 3, 1) #, pixel_norm=pixel_norm)

        if self.image_res == 32:
            self.to_rgb_32 = nn.Conv2d(in_channel, 3, 1)
        if self.image_res == 64:
            self.to_rgb_64 = nn.Conv2d(in_channel, 3, 1)
        if self.image_res == 128:
            self.to_rgb_128 = nn.Conv2d(in_channel//4, 3, 1)
        if self.image_res == 256:
            self.to_rgb_256 = nn.Conv2d(in_channel//4, 3, 1)
        
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2) #, mode="bilinear")
        out = module(out)
        return out

    def output_simple(self, feat1,  module1, alpha):
        out = module1(feat1)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=6, alpha=0):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)        
        out = self.progress(out_8, self.progression_16)

        resolutions = [32, 64, 128, 256]

        for res in resolutions:
            if self.image_res >= res:
                out = self.progress(out, getattr(self, f'progression_{res}'))
                if self.image_res == res:
                    return self.output_simple(out, getattr(self, f'to_rgb_{res}'), alpha)


        # out_32 = self.progress(out_16, self.progression_32)
        # if self.image_res == 32:
        #     return self.output_simple(out_32,self.to_rgb_32,alpha)
        # if self.image_res >= 64:
        #     out_64 = self.progress(out_32, self.progression_64)
        #     if self.image_res == 64:
        #         return self.output_simple(out_64,self.to_rgb_64,alpha)
        # if self.image_res >= 128:
        #     out_128 = self.progress(out_64, self.progression_128)
        #     if self.image_res == 128:
        #         return self.output_simple(out_128,self.to_rgb_128,alpha)
        # if self.image_res >= 256:
        #     out_256 = self.progress(out_128, self.progression_256)
        #     if self.image_res == 256:
        #         return self.output_simple(out_256,self.to_rgb_256,alpha)
        
        
        