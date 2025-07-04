import os
import numpy as np
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .mutual_information import MIScore
try:
    from apex import amp
except ImportError as error:
    print(error)
# # Arthur: import PyTorch's AMP as an alternative to apex (AMP benefits come with larger batch sizes)
# from torch.amp import GradScaler, autocast
# Arthur: import RAFT
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
# Arthur: import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


class CycleGANStructModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True)  # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            # shuxian: additional args for structural loss & depth estimation
            parser.add_argument('--lambda_struct', type=float, default=1.0, help='weight for structural loss (depth consistency)')
            # parser.add_argument('--tgt_depth', type=str, default='', help='model for depth esimtation in domain B')
            # Arthur: additional args for temporal loss & optical flow loss
            parser.add_argument('--lambda_temporal', type=float, default=0.0, help='weight for temporal loss (temporal consistency)')
            parser.add_argument('--lambda_flow', type=float, default=0.0, help='weight for optical flow consistency loss')
            # # Arthur: add mixed precision training option
            # parser.add_argument('--amp', action='store_true', help='use automatic mixed precision')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # shuxian: add opt.amp
        self.opt.amp = False

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'struct']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'struct', 'temporal', 'flow']

        # initialize previous frames
        self.prev_real_A = None
        self.prev_fake_B = None

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']  # AtoB translation
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']  # BtoA translation
        visual_names_A = ['fake_B']
        visual_names_B = [] 

        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) & idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        # # shuxian: add estimated depth visualization
        # visual_names_A.append('fake_B_depth')

        # Arthur: add optical flow images
        if self.isTrain and self.opt.lambda_flow > 0:
            visual_names_A.extend(['flow_real_A_img', 'flow_fake_B_img'])

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A & B
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt=opt)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt=opt)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.struct_loss = MIScore(bins=256,min=0,max=1)

            # # Arthur: mixed precision training
            # if self.opt.amp:
            #     self.scaler = GradScaler()
        
        # Arthur: RAFT preprocessing
        weights = Raft_Large_Weights.DEFAULT
        self.raft_transforms = weights.transforms()
        
        # Arthur: create RAFT model (for optical flow)
        self.raft_model = raft_large(weights=None).to(self.device)

        # Load weights (download weights with "wget https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth -O raft_weights.pth")
        state_dict = torch.load('RAFT/raft_weights.pth', map_location=self.device)
        self.raft_model.load_state_dict(state_dict)
        self.raft_model.eval()

        for param in self.raft_model.parameters():
            param.requires_grad = False  # freeze RAFT
        
        # Arthur: add tensorboard logging
        if self.isTrain:
            self.writer = SummaryWriter(f'checkpoints/{opt.name}')
            self.step = 0

    def compute_raft_flow(self, frame1, frame2):
        """Arthur: compute optical flow using the RAFT model.

        Parameters:
            frame1, frame2: tensors with shape [B, 3, H, W] where B = batch size & 3 = RGB channels.
        """
        frame1_01 = (frame1 + 1) / 2
        frame2_01 = (frame2 + 1) / 2
        frame1_raft, frame2_raft = self.raft_transforms(frame1_01, frame2_01)
        with torch.no_grad():
            flow = self.raft_model(frame1_raft, frame2_raft)[-1]
        return flow

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # # shuxian: load depths as well
        # if AtoB:
        #     self.real_A_depth = input['A_depth'].to(self.device)
        # else:
        #     self.real_A_depth = None

        # Arthur: only load depths during training (saves memory & faster)
        if self.isTrain and AtoB:
            self.real_A_depth = input['A_depth'].to(self.device)
        else:
            self.real_A_depth = None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""  # generates outputs
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # # Arthur: mixed precision wrapper
        # if self.opt.amp:
        #     with autocast('cuda'):
        #         self.fake_B = self.netG_A(self.real_A)
        #         self.rec_A = self.netG_B(self.fake_B)
        #         self.fake_A = self.netG_B(self.real_B)
        #         self.rec_B = self.netG_A(self.fake_A)
        # else:
        #     self.fake_B = self.netG_A(self.real_A)
        #     self.rec_A = self.netG_B(self.fake_B)
        #     self.fake_A = self.netG_B(self.real_B)
        #     self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss & calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.opt.amp:
            with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()
        # # Arthur: mixed precision wrapper (move loss calculations inside autocast to maintain consistent dtypes)
        # if self.opt.amp:
        #     with autocast('cuda'):
        #         pred_real = netD(real)
        #         pred_fake = netD(fake.detach())
        #         loss_D_real = self.criterionGAN(pred_real, True)
        #         loss_D_fake = self.criterionGAN(pred_fake, False)
        #         loss_D = (loss_D_real + loss_D_fake) * 0.5
        #     self.scaler.scale(loss_D).backward()
        # else:
        #     pred_real = netD(real)
        #     pred_fake = netD(fake.detach())
        #     loss_D_real = self.criterionGAN(pred_real, True)
        #     loss_D_fake = self.criterionGAN(pred_fake, False)
        #     loss_D = (loss_D_real + loss_D_fake) * 0.5
        #     loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # shuxian: add structural loss
        if self.opt.lambda_struct > 0:
            self.loss_struct = self.opt.lambda_struct * self.calculate_struct_loss(self.real_A_depth, self.fake_B)
        else:
            self.loss_struct = 0
        
        # Arthur: add temporal loss
        if self.opt.lambda_temporal > 0 and self.prev_fake_B is not None:
            self.loss_temporal = self.opt.lambda_temporal * torch.nn.functional.mse_loss(self.fake_B, self.prev_fake_B)  # compute loss between batches
        else:
            self.loss_temporal = 0
        
        # Arthur: add optical flow loss
        if self.opt.lambda_flow > 0 and self.prev_real_A is not None:  # GAN tensor range [-1, 1]
            flow_real_A = self.compute_raft_flow(self.prev_real_A, self.real_A)
            flow_fake_B = self.compute_raft_flow(self.prev_fake_B, self.fake_B)
            # # Swap argument order to get correct optical flow maps (don't know why)
            # flow_real_A = self.compute_raft_flow(self.real_A, self.prev_real_A) 
            # flow_fake_B = self.compute_raft_flow(self.fake_B, self.prev_fake_B)
            # normalize
            flow_real_A_norm = flow_real_A / (flow_real_A.abs().max() + 1e-8)
            flow_fake_B_norm = flow_fake_B / (flow_fake_B.abs().max() + 1e-8)
            # compute MSE
            self.loss_flow = self.opt.lambda_flow * torch.nn.functional.mse_loss(flow_real_A_norm, flow_fake_B_norm)
            # convert flow to RGB for visualization
            self.flow_real_A_img = flow_to_image(flow_real_A).float() / 255.0
            self.flow_fake_B_img = flow_to_image(flow_fake_B).float() / 255.0
        else:
            self.loss_flow = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # # Arthur: mixed precision wrapper
        # if self.opt.amp:
        #     with autocast('cuda'):
        #         self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        #         self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # else:
        #     self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        #     self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        # combine losses & calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_struct + self.loss_temporal + self.loss_flow

        if self.opt.amp:
            with amp.scale_loss(self.loss_G, self.optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            self.loss_G.backward()
        # # Arthur: mixed precision wrapper
        # if self.opt.amp:
        #     self.scaler.scale(self.loss_G).backward()
        # else:
        #     self.loss_G.backward()

    def data_dependent_initialize(self, data):
        return

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images & reconstruction images.
        
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # # Arthur: mixed precision wrapper
        # if self.opt.amp:
        #     self.scaler.step(self.optimizer_G)
        # else:
        #     self.optimizer_G.step()

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # # Arthur: mixed precision wrapper
        # if self.opt.amp:
        #     self.scaler.step(self.optimizer_D)
        #     self.scaler.update()  
        # else:
        #     self.optimizer_D.step()

        # if self.prev_real_A is not None:
            # print(f"Previous path: {self.prev_image_path}")
            # print(f"Current path:  {self.image_paths[0]}")

        # Arthur: store frames used in temporal & flow losses
        self.prev_real_A = self.real_A.detach().clone()
        self.prev_fake_B = self.fake_B.detach().clone() 
        self.prev_path = self.image_paths[0]  # Store path for debugging

        self.prev_path = self.image_paths[0]
         
        # Arthur: log losses in tensorboard every 100 steps
        if self.step % 100 == 0:
            self.writer.add_scalar('Loss/D_A', self.loss_D_A, self.step)
            self.writer.add_scalar('Loss/G_A', self.loss_G_A, self.step)
            self.writer.add_scalar('Loss/D_B', self.loss_D_B, self.step)
            self.writer.add_scalar('Loss/G_B', self.loss_G_B, self.step)
            self.writer.add_scalar('Loss/cycle_A', self.loss_cycle_A, self.step)
            self.writer.add_scalar('Loss/cycle_B', self.loss_cycle_B, self.step)
            self.writer.add_scalar('Loss/idt_A', self.loss_idt_A, self.step)
            self.writer.add_scalar('Loss/idt_B', self.loss_idt_B, self.step)
            self.writer.add_scalar('Loss/struct', self.loss_struct, self.step)
            if self.opt.lambda_temporal > 0:
                self.writer.add_scalar('Loss/temporal', self.loss_temporal, self.step)
            if self.opt.lambda_flow > 0:
                self.writer.add_scalar('Loss/flow', self.loss_flow, self.step)
        
        self.step += 1

    def calculate_struct_loss(self, src_depth, tgt):
        consolidated_tgt = torch.mean(tgt, axis=-3)
        return self.struct_loss(src_depth, consolidated_tgt)
