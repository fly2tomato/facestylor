model = dict(
    type='PSPTransfer',
    src_generator=None,
    generator=dict(type='SwapStyleGANv2Generator',
                   out_size=1024,
                   style_channels=512,
                   num_mlps=8),
    discriminator=None,
    gan_loss=None,
    lpips_lambda=0.0)

train_cfg = dict(use_ema=True)
test_cfg = None

