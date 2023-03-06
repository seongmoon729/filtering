import os
import math
import tempfile

from subprocess import Popen, PIPE
from pathlib import Path
from PIL import Image

import numpy as np


# Define base commands.
FFMPEG_BASE_CMD = f"ffmpeg -y -loglevel error"
VTM_BASE_CMD    = f"vtm -c {os.environ['VTM_CFG_PATH']}"
# VVENC_BASE_CMD  = f"vvencFFapp -c {os.environ['VVENC_CFG_PATH']}"
VVENC_BASE_CMD  = f"vvencFFapp -c {os.environ['VVENC_ORG_CFG_PATH']}"

DS_LEVELS = [0, 1, 2, 3]
CODEC_LIST = ['jpeg', 'webp', 'vtm', 'vvenc']
JPEG_QUALITIES  = [31, 11, 5, 2]
WEBP_QUALITIES  = [1, 18, 53, 83, 95]
VTM_QUALITIES   = [47, 42, 37, 32, 27, 22]
VVENC_QUALITIES = [50, 45, 40, 35, 30, 25]
    

def codec_fn(x, codec, quality, downscale=0, uv_zero=False):
    """ Encode & decode input with codec. 
        Args:
            x: 'np.ndarray' with range of [0, 1] and order of (C, H, W).
            codec: 'jpeg', 'webp', or 'vtm'
            quality: a parameter to control quantization level
            downscale: a parameter to control downscaling level
    """
    assert codec in CODEC_LIST

    x = x.copy()
    x *= 255.                   # Denormalize
    x = x.transpose(1, 2, 0)    # (C, H, W) -> (H, W ,C)
    x = x.round().astype('uint8')
    pil_img = Image.fromarray(x)
    
    pil_img_recon, bpp = run_codec(pil_img, codec, quality, downscale, uv_zero)
    pil_img.close()

    x = np.array(pil_img_recon)
    pil_img_recon.close()

    x = x.astype('float32')
    x = x.transpose(2, 0, 1)    # (H, W, C) -> (C, H, W)
    x /= 255.                   # Normalize to [0, 1]
    return x, bpp


def run_codec(input, codec, q, ds=0, uv_zero=False):
    assert ds in DS_LEVELS, f"Choose one of {DS_LEVELS}."

    # Make temp directory for processing.
    dst_dir_obj = tempfile.TemporaryDirectory()
    dst_dir = Path(dst_dir_obj.name)

    # Save input image in temp directory for cmd processing.
    src_img_path = dst_dir / 'raw.png'
    if isinstance(input, (str, Path)):
        pil_img = Image.open(input)
    else:
        pil_img = input
    pil_img.save(src_img_path)

    # Define intermediate file names.
    file_name       = 'img'
    tmp_png_path    = dst_dir / (file_name + '.png')
    tmp_yuv_path    = dst_dir / (file_name + '.yuv')
    recon_yuv_path  = dst_dir / (file_name + '_recon.yuv')
    bin_path        = dst_dir / (file_name + '_comp.bin')
    log_path        = dst_dir / (file_name + '.log')
    recon_png_path  = dst_dir / (file_name + '_recon.png')

    # Compute down-scaled or padded size.
    w, h = pil_img.size
    if ds == 0:
        dw, dh = map(lambda x: math.ceil(x / 2) * 2, [w, h])
    else:
        dw, dh = map(lambda x: math.ceil(x * (4 - ds) / 8) * 2, [w, h])

    # 1. Downscaling/Padding.
    _run_ffmpeg_down_scaling(src_img_path, tmp_png_path, dw, dh, ds)

    # 2. Image to YUV.
    _run_ffmpeg_img2yuv(tmp_png_path, tmp_yuv_path)

    # optional
    if uv_zero:
        _make_uv_zero(tmp_yuv_path, dw, dh)

    # 3. Codec.
    if codec == 'jpeg':
        _run_ffmpeg_jpeg(tmp_yuv_path, bin_path, recon_yuv_path, dw, dh, q)
    elif codec == 'webp':
        _run_ffmpeg_webp(tmp_yuv_path, bin_path, recon_yuv_path, dw, dh, q)
    elif codec == 'vtm':
        _run_vtm(tmp_yuv_path, bin_path, recon_yuv_path, dw, dh, q, log_path)
    elif codec == 'vvenc':
        _run_vvenc(tmp_yuv_path, bin_path, recon_yuv_path, dw, dh, q, log_path)
    else:
        raise ValueError(f"'{codec}' is wrong codec, available: {CODEC_LIST}.")
    # 4. YUV to Image
    _run_ffmpeg_yuv2img(recon_yuv_path, tmp_png_path, dw, dh)
    # 5. Upscaling/Cropping
    _run_ffmpeg_up_scaling(tmp_png_path, recon_png_path, w, h, ds)

    # Generate processing results.
    recon_img = Image.open(recon_png_path)
    bytes = bin_path.stat().st_size
    bpp = bytes * 8 / (h * w)
    return recon_img, bpp


def _run_ffmpeg_down_scaling(src_path, dst_path, width, height, ds):
    if ds == 0:
        out_opts = f"-vf 'pad={width}:{height}'"
    else:
        out_opts = f"-vf 'scale={width}:{height}'"

    cmd = f"{FFMPEG_BASE_CMD} -i {src_path} {out_opts} {dst_path}"
    _run_cmd(cmd)


def _run_ffmpeg_up_scaling(src_path, dst_path, width, height, ds):
    if ds == 0:
        out_opts = f"-vf 'crop={width}:{height}:0:0'"
    else:
        out_opts = f"-vf 'scale={width}:{height}'"
    cmd = f"{FFMPEG_BASE_CMD} -i {src_path} {out_opts} {dst_path}"
    _run_cmd(cmd)


def _run_ffmpeg_img2yuv(src_path, dst_path):
    out_opts = "-f rawvideo -pix_fmt yuv420p -dst_range 1"
    cmd = f"{FFMPEG_BASE_CMD} -i {src_path} {out_opts} {dst_path}"
    _run_cmd(cmd)


def _make_uv_zero(src_path, width, height):
    yuv = np.fromfile(src_path, dtype=np.uint8)
    yuv = yuv.reshape((int(height*1.5), width))  # YUV420 format

    # Set U and V components to zero
    yuv[int(height):int(height*1.25), :] = 128  # Set U component to 128
    yuv[int(height*1.25):, :] = 128  # Set V component to 128

    # Save modified YUV file
    yuv.tofile(src_path)


def _run_ffmpeg_yuv2img(src_path, dst_path, width, height):
    in_opts = f"-f rawvideo -pix_fmt yuv420p10le -s {width}x{height} -src_range 1"
    out_opts = "-frames 1 -pix_fmt rgb24"
    cmd = f"{FFMPEG_BASE_CMD} {in_opts} -i {src_path} {out_opts} {dst_path}"
    _run_cmd(cmd)


def _run_ffmpeg_jpeg(src_path, bin_path, recon_path, width, height, quality):
    in_opts  = f"-f rawvideo -s {width}x{height} -pix_fmt yuv420p"
    out_opts = f"-q:v {quality} -f mjpeg"
    cmd = (f"{FFMPEG_BASE_CMD} {in_opts} -i {src_path} {out_opts} {bin_path} && "
           f"{FFMPEG_BASE_CMD} -i {bin_path} -pix_fmt yuv420p10le {recon_path}")
    _run_cmd(cmd)


def _run_ffmpeg_webp(src_path, bin_path, recon_path, width, height, quality):
    in_opts  = f"-f rawvideo -s {width}x{height} -pix_fmt yuv420p"
    out_opts = f"-q:v {quality} -f webp"
    cmd = (f"{FFMPEG_BASE_CMD} {in_opts} -i {src_path} {out_opts} {bin_path} && "
           f"{FFMPEG_BASE_CMD} -i {bin_path} -pix_fmt yuv420p10le {recon_path}")
    _run_cmd(cmd)


def _run_vtm(src_path, bin_path, recon_path, width, height, quality, log_path):
    cmd = (f"{VTM_BASE_CMD} -i {src_path} -b {bin_path} -o {recon_path}"
           f" --ConformanceWindowMode=1 -q {quality} -wdt {width} -hgt {height}"
           f" -f 1 -fr 1 --InternalBitDepth=10 > {log_path}")
    _run_cmd(cmd)


def _run_vvenc(src_path, bin_path, recon_path, width, height, quality, log_path):
    cmd = (f"{VVENC_BASE_CMD} -i {src_path} -b {bin_path} -o {recon_path}"
           f" --ConformanceWindowMode=1 -q {quality} -s {width}x{height}"
           f" -f 1 -fr 1 --InternalBitDepth=10 --threads=4")
    _run_cmd(cmd)


def _run_cmd(cmd):
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    _, err = proc.communicate()
    if err:
        print(err)
