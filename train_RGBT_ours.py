import torch.nn as nn
import torch.optim as optim
import torchvision.transforms


from ltr.dataset.UAV_RGBT import UAV_RGBT
from ltr.dataset.gtot import GTOT
from ltr.dataset.rgbt234 import RGBT234
from ltr.dataset.Lasher import LasHeR


from ltr.data import processing, sampler, LTRLoader
import ltr.models.tracking.optim_tracker_trans as optim_tracker_models
import ltr.models.loss as ltr_losses
import ltr.models.loss.kl_regression as klreg_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as dltransforms
import pdb

def run(settings):
    settings.description = 'First training with gradient descent.'
    settings.batch_size = 4
    settings.num_workers = 8   

    settings.print_interval = 10
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]

    settings.search_area_factor = 5.0  #6.0
    settings.output_sigma_factor = 1/4
    settings.target_filter_sz = 4
    settings.feature_sz = 22
    settings.output_sz = settings.feature_sz * 16

    settings.center_jitter_factor = {'train': 3, 'test': 4.5} # 3, 5.5
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    settings.hinge_threshold = 0.05
    
    settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/init_loss', 'ClfTrain/train_loss', 'ClfTrain/iter_loss',
                            'ClfTrain/test_loss', 'ClfTrain/test_init_loss', 'ClfTrain/test_iter_loss','Loss/RMI_loss']

    #####################  Train datasets ##########################
    vtuav_train = UAV_RGBT(settings.env.UAV_RGBT_dir, split='train', modality = 'RGBT')
    gtot_train = GTOT(settings.env.GTOT_dir)
    rgbt234_train = RGBT234(settings.env.RGBT234_dir)
    lasher_train = LasHeR(settings.env.LasHeR_dir)
    ### Validation datasets
    vtuav_val = UAV_RGBT(settings.env.UAV_RGBT_dir, split='val', modality = 'RGBT')

    ###############################################################

    ####################### Data transform  #######################
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    ############# The tracking pairs processing module ###########
    output_sigma = settings.output_sigma_factor / settings.search_area_factor
    proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 8, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}
    label_params = {'feature_sz': settings.feature_sz, 'sigma_factor': output_sigma, 'kernel_sz': settings.target_filter_sz}
    ###############################################################
    data_processing_train = processing.TrackingProcessing(search_area_factor=settings.search_area_factor,
                                                        output_sz=settings.output_sz,
                                                        center_jitter_factor=settings.center_jitter_factor,
                                                        scale_jitter_factor=settings.scale_jitter_factor,
                                                        mode='sequence',
                                                        proposal_params=proposal_params,
                                                        label_function_params=label_params,
                                                        transform=transform_train,
                                                        joint_transform=transform_joint)

    data_processing_val = processing.TrackingProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=proposal_params,
                                                      label_function_params=label_params,
                                                      transform=transform_val,
                                                      joint_transform=transform_joint)

    #### Train sampler and loader
    iters_per_epoch = 16000
    dataset_train = sampler.RandomSequenceWithDistractors([vtuav_train,gtot_train,lasher_train], [1,0.2,1],
                                                  samples_per_epoch=iters_per_epoch, max_gap=5, frame_sample_mode='causal',
                                                  num_seq_test_frames=3, num_class_distractor_frames=0,
                                                  num_seq_train_frames=3, num_class_distractor_train_frames=0,
                                                  processing=data_processing_train)

    loader_train = LTRLoader('train', dataset_train, training=True, 
                             batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    #### Validation samplers and loaders
    dataset_val = sampler.RandomSequenceWithDistractors([vtuav_val], [1], samples_per_epoch=1000, max_gap=5, 
                                                frame_sample_mode='causal',
                                                num_seq_test_frames=3, num_class_distractor_frames=0,
                                                num_seq_train_frames=3, num_class_distractor_train_frames=0,
                                                processing=data_processing_val)

    loader_val = LTRLoader('val', dataset_val, training=False, 
                           batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=10, stack_dim=1)
    
    ############### Create network and actor  #############################
    net = optim_tracker_models.steepest_descent_learn_filter_resnet50_newiou(
        filter_size=settings.target_filter_sz, backbone_pretrained=True, 
        optim_iter=5,clf_feat_norm=True, clf_feat_blocks=0, final_conv=True, out_feature_dim=512, 
        optim_init_step=0.9, optim_init_reg=0.1,
        init_gauss_sigma=output_sigma*settings.feature_sz, 
        num_dist_bins=10, bin_displacement=0.5,
        mask_init_factor=3.0,frozen_backbone_layers=['conv1','bn1','layer1','layer2'])#'layer2'
        
    objective = {'iou': nn.MSELoss(size_average=True), 
                'test_clf': ltr_losses.LBHinge(threshold=settings.hinge_threshold),
                'RMI_loss': nn.KLDivLoss()}
    
    loss_weight = {'iou': 1, 'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                   'train_clf': 0, 'init_clf': 0,'RMI_loss':100}

    actor = actors.OptimTrackerActor(net=net, objective=objective, loss_weight=loss_weight)

    ########################### Optimizer ###################################
    lrgain = 0.1
    optimizer = optim.Adam([
                            ###############  Backbone #######################
                            {'params': actor.net.feature_extractor.parameters(), 'lr':lrgain* 2e-5},
                            {'params': actor.net.feature_extractor_i.parameters(), 'lr':lrgain* 2e-5},
                            {'params': actor.net.feature_fusion.parameters(), 'lr':lrgain * 2e-5},
                            ###############  Classifier #####################
                            {'params': actor.net.classifier.filter_initializer.parameters(), 'lr': lrgain*5e-5},
                            {'params': actor.net.classifier.filter_optimizer.parameters(), 'lr': lrgain*5e-4},
                            {'params': actor.net.classifier.feature_extractor.parameters(), 'lr': lrgain*5e-5},
                            {'params': actor.net.classifier.transformer.parameters(), 'lr': lrgain*1e-3},

                            {'params': actor.net.late_fusion.parameters(),'lr': 2e-4},
                            ##############3  BBregressor ####################
                            {'params': actor.net.bb_regressor.parameters(),'lr': 1e-3},
                            ],
                           lr=lrgain*2e-4)
                           
    ############################################################################
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    
    trainer.train(70, load_latest=True, fail_safe=True)
