import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def forward_transform_save(image_path, output_png="ft_combined.png"):
    im = Image.open(image_path)
    if im.mode == 'RGB':
        C = 3
        img = np.array(im).astype(np.float64)
    elif im.mode == 'L':
        C = 1
        img = np.array(im).astype(np.float64)[..., None]
    else:
        raise ValueError("Image must be RGB or grayscale.")
    H, W = img.shape[:2]
    ft_bytes = []
    for c in range(C):
        channel = img[..., c]
        F = np.fft.fft2(channel, axes=(0, 1), norm=None).astype(np.complex64)
        F_shifted = np.fft.fftshift(F)
        ft_bytes.append(F_shifted.tobytes())
    header = np.zeros(8*W, dtype=np.uint8)
    header[0] = C
    header[1] = H % 256
    header[2] = H // 256
    header[3] = W % 256
    header[4] = W // 256
    all_bytes = b''.join([header.tobytes()] + ft_bytes)
    expected_size = (1 + C*H) * (8*W)
    if len(all_bytes) != expected_size:
        raise ValueError(f"Byte size mismatch: expected {expected_size}, got {len(all_bytes)}.")
    png_data = np.frombuffer(all_bytes, dtype=np.uint8).reshape((1 + C*H, 8*W))
    Image.fromarray(png_data, mode='L').save(output_png)

def reconstruct_from_png(png_path):
    im = Image.open(png_path)
    if im.mode != 'L':
        raise ValueError("PNG must be grayscale.")
    png_data = np.array(im).astype(np.uint8)
    height, width = png_data.shape
    header = png_data[0, :5]
    C = int(header[0])
    H = int(header[1]) + 256 * int(header[2])
    W = int(header[3]) + 256 * int(header[4])
    if width != 8*W or height != 1 + C*H:
        raise ValueError(f"PNG shape {height}x{width} inconsistent with header C={C}, H={H}, W={W}.")
    recon_channels = []
    for c in range(C):
        start_row = 1 + c*H
        end_row = start_row + H
        ft_bytes = png_data[start_row:end_row, :].tobytes()
        expected_bytes = H * W * 8
        if len(ft_bytes) != expected_bytes:
            raise ValueError(f"Channel {c} byte size mismatch: expected {expected_bytes}, got {len(ft_bytes)}.")
        F_shifted = np.frombuffer(ft_bytes, dtype=np.complex64).reshape((H, W))
        F = np.fft.ifftshift(F_shifted)
        channel = np.real(np.fft.ifft2(F))
        recon_channels.append(channel)
    recon_img = recon_channels[0] if C == 1 else np.stack(recon_channels, axis=-1)
    recon_img = np.clip(recon_img, 0, 255).astype(np.uint8)
    return recon_img

if __name__ == "__main__":
    mode = 2
    if mode == 1:
        input_image = "skks.jpg"
        if not os.path.exists(input_image):
            pass
        else:
            forward_transform_save(input_image, "skks.png")
    elif mode == 2:
        ft_png = "skks.png"
        if not os.path.exists(ft_png):
            pass
        else:
            recon_img = reconstruct_from_png(ft_png)
            plt.figure(figsize=(6, 6))
            if recon_img.ndim == 2:
                plt.imshow(recon_img, cmap='gray')
            else:
                plt.imshow(recon_img)
            
            plt.title("here's the picture FT decoded <3 (anatomical heart emoji)") 
            plt.axis("off")
            plt.show()
            Image.fromarray(recon_img).save("ily.jpg")
    else:
        pass