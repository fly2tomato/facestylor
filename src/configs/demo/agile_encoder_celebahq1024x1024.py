encoder_ckpt_path = 'work_dirs/pre-trained/agile_encoder_celebahq1024x1024_lr_1e-4_150k_20211104_133124-a7e2fd7f.pth'

model = dict(type='PSPEncoderDecoder',
             encoder=dict(type='VAEStyleEncoder',
                          num_layers=50,
                          pretrained=dict(ckpt_path=encoder_ckpt_path,
                                          prefix='encoder',
                                          strict=False)),
             decoder=dict(type='SwapStyleGANv2Generator',
                          out_size=1024,
                          style_channels=512,
                          num_mlps=8,
                            pretrained=dict(ckpt_path=encoder_ckpt_path,
                                          prefix='decoder')),
             pool_size=(1024, 1024),
             id_lambda=0.0,
             lpips_lambda=0.0,
             id_ckpt=None,
             kl_loss=None,
             train_cfg=None,
             test_cfg=None)

train_cfg = None
test_cfg = None
