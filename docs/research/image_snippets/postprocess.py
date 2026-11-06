"""Game-asset post-processing that is already available offline (run with envs/hy3d/bin/python).

  bg removal : rembg 2.0.65 + ~/.u2net/u2net.onnx (176 MB, CPU onnxruntime, ~0.4 s @1024px). Fine for objects on
               plain backgrounds. Better mattes (hair, fur, thin parts): rembg "birefnet-general" (~973 MB onnx,
               MIT) or "birefnet-general-lite" (~224 MB) -- auto-download to ~/.u2net on first use (not local yet).
               BRIA RMBG-1.4/2.0 are non-commercial (and 2.0 gated) -> avoid for a commercial studio.
               Prefer Qwen-Image-2.1 native RGBA output when generating; use rembg for other models / user uploads.
  upscale    : Real-ESRGAN x4plus (engines/hy3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth, 64 MB, BSD-3)
               via realesrgan 0.3.0 + basicsr 1.4.2 (installed in envs/hy3d). Upscale RGB and alpha separately.
"""
import sys
import numpy as np
from PIL import Image

U2NET_HOME = "~/.u2net"
ESRGAN = "engines/hy3d/vendor/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"


def remove_bg(img: Image.Image, model: str = "u2net") -> Image.Image:
    from rembg import new_session, remove
    return remove(img.convert("RGB"), session=new_session(model))  # RGBA


def upscale_x4(img: Image.Image, tile: int = 512, out_scale: float = 4) -> Image.Image:
    import cv2
    import torch
    import torchvision.transforms.functional as _F  # basicsr 1.4.2 still imports the removed functional_tensor module
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _F)
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    up = RealESRGANer(scale=4, model_path=ESRGAN, model=net, tile=tile, tile_pad=10, pre_pad=0, half=True,
                      device=torch.device("cuda"))
    arr = np.asarray(img.convert("RGBA" if img.mode == "RGBA" else "RGB"))
    bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
    if img.mode == "RGBA":  # RealESRGANer upscales the alpha channel too when given BGRA
        bgr = np.dstack([bgr, arr[..., 3]])
    out, _ = up.enhance(bgr, outscale=out_scale)
    if out.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    im = Image.open(src)
    if "--rembg" in sys.argv:
        im = remove_bg(im)
    if "--x4" in sys.argv:
        im = upscale_x4(im)
    im.save(dst)
