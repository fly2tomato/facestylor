import gradio as gr
import os,sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))  # isort:skip  # noqa
from demo.quick_try import AgileGANTest, _SUPPORTED_STYLE, download_ckpt, _CKPT_URL, _RES256
import mmcv
import cv2
import torch
import numpy as np
import agilegan


def ad_qr_code(img, qr_path=None):
    h, w, _ = img.shape
    qr_size = min(h, w)//4
    qr_patch = cv2.imread(qr_path)
    qr_patch = cv2.resize(qr_patch, (qr_size, qr_size))
    img[-qr_size:, -qr_size:,] = qr_patch
    return img

def add_src_img(img, aligned_img):
    h, w, _ = img.shape
    radius = int(min(h, w) // 8)
    patch_size = 2 * radius
    # make mask
    left_btm = img[(h - patch_size):, :patch_size, ]
    mask = np.ones_like(left_btm)
    mask = cv2.circle(mask, (radius, radius), radius, (0, 0, 0), -1,
                      cv2.LINE_AA)
    aligned_img = cv2.resize(aligned_img, (patch_size, patch_size))
    left_btm = aligned_img * (1 - mask) + mask * left_btm
    img[(h - patch_size):, :patch_size, ] = left_btm
    # make circle frame
    img = cv2.circle(img, (radius, h - radius), radius, (147, 107, 9), 1,
                     cv2.LINE_AA)
    return img

def make_testors():
    print(f"start make_testors")
    testors = dict()
    for style in _SUPPORTED_STYLE:
        if style in _RES256:
            encoder_config = base_root + '/configs/demo/agile_encoder_256x256.py'
            encoder_ckpt = base_root + '/work_dirs/lite-weights/agile_encoder_celebahq256x256_lr_1e-4_150k_20211104_134520-9cce67da.pth'
            download_ckpt(encoder_ckpt, _CKPT_URL['encoder256'])
            transfer_config = base_root + '/configs/demo/agile_transfer_256x256.py'
        else:
            encoder_config = base_root+'/configs/demo/agile_encoder_1024x1024.py'
            encoder_ckpt = base_root + '/work_dirs/lite-weights/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth'
            download_ckpt(encoder_ckpt, _CKPT_URL['encoder1024'])
            transfer_config = base_root+'/configs/demo/agile_transfer_1024x1024.py'
            
        transfer_ckpt = _SUPPORTED_STYLE[style]
        download_ckpt(transfer_ckpt, _CKPT_URL[style])

        testor = AgileGANTest(encoder_config=encoder_config, encoder_ckpt=
                            encoder_ckpt, transfer_config=transfer_config, transfer_ckpt=transfer_ckpt)
        testors[style] = testor
    print(f"end make_testors")
    return testors

def make_testor(style):
    print(f"start a make_testor")
    if style in _RES256:
        encoder_config = base_root + '/configs/demo/agile_encoder_256x256.py'
        encoder_ckpt = base_root + '/work_dirs/lite-weights/agile_encoder_celebahq256x256_lr_1e-4_150k_20211104_134520-9cce67da.pth'
        download_ckpt(encoder_ckpt, _CKPT_URL['encoder256'])
        transfer_config = base_root + '/configs/demo/agile_transfer_256x256.py'
    else:
        encoder_config = base_root+'/configs/demo/agile_encoder_1024x1024.py'
        encoder_ckpt = base_root + '/work_dirs/lite-weights/agile_encoder_ffhq1024x1024_lr_1e-4_500kiter_20211201_112111-fb1312dc.pth'
        download_ckpt(encoder_ckpt, _CKPT_URL['encoder1024'])
        transfer_config = base_root+'/configs/demo/agile_transfer_1024x1024.py'
            
    transfer_ckpt = _SUPPORTED_STYLE[style]
    download_ckpt(transfer_ckpt, _CKPT_URL[style])

    testor = AgileGANTest(encoder_config=encoder_config, encoder_ckpt=
                        encoder_ckpt, transfer_config=transfer_config, transfer_ckpt=transfer_ckpt)
    print(f"end make a testor")
    return testor

class Img_savor:
    def __init__(self):
        self.index = 3737
        self.input_path = base_root + "/user/input"
    def save_input(self, img):
        self.index += 1
        filepath = os.path.join(self.input_path, str(self.index).zfill(10) + ".png")
        cv2.imwrite(filepath, img)

def faceStylor(img, style_label, with_qrcode=True):
    style = label2args[style_label]
    testor = make_testor(style)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_savor.save_input(image)
    try:
      style_image, aligned_img = testor.run(image, os.path.join(base_root + '/user/output', str(img_savor.index).zfill(10) + ".png"), style, resize=True)
      style_image = torch.clamp(style_image, -1., 1.)
      style_image = mmcv.tensor2imgs(style_image, mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5), to_rgb=False)[0]
      if with_qrcode:
          style_image = ad_qr_code(style_image, _QR_PATH)
          style_image = add_src_img(style_image, aligned_img)
      return style_image, "手机长按保存图片，电脑右键点击保存图片。 \n Long press or right click to save the picture.", "<center><a href=\"https://github.com/open-mmlab/MMGEN-FaceStylor/issues\"><button style=\"border: none;color: white;padding: 15px 32px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer;background-color: #4CAF50;\">反馈 Feedback</button></a></center>"
    except:
      print("error", "facestylor")
      testor.move_out_cuda()
      return "failed"


def exe_shell(command: str, shell: str = '/bin/bash'):
    return Popen(command, stdout=PIPE, stderr=STDOUT, shell=True, executable=shell)


# install mmcv mmgen and mmcls
process=exe_shell(command="echo 'start to run shell script' && bash install.sh")
try:
    with (process.stdout):
        y = 0
        for line in iter(process.stdout.readline, b''):
            # s = str(line).replace("b'", "").replace("'", "").replace("\\n", "")
            s = line.decode('utf-8')
            print(s)
except Exception as e:
    print(e)



base_root = os.path.abspath(os.path.join(__file__, "../"))
_QR_PATH = base_root + "/qrcode.png"

# testors = make_testors()

label2args = dict()
label2args["卡通 toonify"] = "toonify"
label2args["油画 oil"] = "oil"
label2args["卡通(小姐姐专用) cartoon"] = "cartoon"
label2args["美漫 comic"] = "comic"
label2args["人像emoji bitmoji"] = "bitmoji"
label2args["素描 sketch"] = "sketch"
img_savor = Img_savor()

share = False
title="MMGEN-FaceStylor"
description="This is a demo for MMGEN-FaceStylor.\n\n\t\
    To use it, simply upload an image, and select a style. Better use an sharp front face."
description_zh = "这是MMGEN-FaceStylor的一个演示，只需要上传一张图片并选择一个风格，它就可以生成风格化的人脸图片。最好使用清晰的人正脸照片。网页下方有关于风格的解释。"
description_all = description_zh+description

btn = gr.outputs.HTML(label="ISSUE")
hint_box = gr.outputs.Textbox(type="auto", label="提示 Hint")
article = open(base_root + "/article.txt", "r", encoding='UTF-8').read()
checkbox = gr.inputs.Checkbox(default=True, label="附二维码 With QR code")
iface = gr.Interface(fn=faceStylor, inputs=["image", gr.inputs.Radio([
  '卡通 toonify', '油画 oil', '卡通(小姐姐专用) cartoon', '美漫 comic', '人像emoji bitmoji', '素描 sketch']), checkbox],
    outputs=["image", hint_box, btn], title=title, description=description_all, article=article, allow_flagging=True).launch(share=share, server_name="0.0.0.0", server_port=7600)

