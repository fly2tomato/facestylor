model = dict(type='PSPEncoderDecoder',
             encoder=dict(type='VAEStyleEncoder',
                          num_layers=50),
             decoder=dict(type='SwapStyleGANv2Generator',
                          out_size=1024,
                          style_channels=512,
                          num_mlps=8),
             pool_size=(1024, 1024),
             id_lambda=0.0, # avoid `The model and loaded state dict do not match exactly`
             lpips_lambda=0.0, # avoid `The model and loaded state dict do not match exactly`
             kl_loss=None,
             train_cfg=None,
             test_cfg=None)
train_cfg = None
test_cfg = None
