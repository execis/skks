import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_16bit_png(array16, out_path):
    if array16.ndim == 2:
        im = Image.fromarray(array16, mode="I;16")
    elif array16.ndim == 3:
        channels = []
        for c in range(array16.shape[-1]):
            ch = array16[..., c]
            channels.append(Image.fromarray(ch, mode="I;16"))
        raise ValueError(":(")
    else:
        raise ValueError(":(")
    im.save(out_path)

def load_16bit_png(file_path):
    im = Image.open(file_path)
    return np.array(im).astype(np.float64)

def forward_transform_save(image_path, mag_png="m.png", phase_png="p.png"):
    im = Image.open(image_path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    img = np.array(im).astype(np.float64)
    
    if img.ndim == 2:
        img = img[..., None]
    
    H, W, C = img.shape
    mag_channels = []
    phase_channels = []
    
    for c in range(C):
        channel = img[..., c]
        F = np.fft.fft2(channel)
        F_shifted = np.fft.fftshift(F)
        mag = np.abs(F_shifted)
        phase = np.angle(F_shifted)
        
        mag_max = mag.max() if mag.max() != 0 else 1.0
        mag_norm = mag / mag_max  
        mag_16 = (mag_norm * 65535).clip(0, 65535).astype(np.uint16)
        
        phase_norm = (phase + np.pi) / (2 * np.pi)
        phase_16 = (phase_norm * 65535).clip(0, 65535).astype(np.uint16)
        
        mag_channels.append(mag_16)
        phase_channels.append(phase_16)
    
    
    mag_to_save = mag_channels[0]
    phase_to_save = phase_channels[0]
    
    save_16bit_png(mag_to_save, mag_png)
    save_16bit_png(phase_to_save, phase_png)

def reconstruct_from_png(mag_png, phase_png):
    mag_16 = load_16bit_png(mag_png)
    phase_16 = load_16bit_png(phase_png)
        
    mag_norm = mag_16.astype(np.float64) / 65535.0   
    phase_norm = phase_16.astype(np.float64) / 65535.0
    phase = phase_norm * (2 * np.pi) - np.pi
    
    F_shifted = mag_norm * np.exp(1j * phase)
    F_ishift = np.fft.ifftshift(F_shifted)
    recon = np.fft.ifft2(F_ishift)
    recon = np.abs(recon)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min()) * 255
    recon_uint8 = recon_norm.clip(0, 255).astype(np.uint8)
    
    return recon_uint8

if __name__ == "__main__":
    mode = 2
    
    if mode == 1:
        input_image = "skks.jpg"
        forward_transform_save(input_image, mag_png="m.png", phase_png="p.png")
    elif mode == 2:
        recon_img = reconstruct_from_png(mag_png="m.png", phase_png="p.png")
        recon_pil = Image.fromarray(recon_img)
        recon_pil.save("ily.png")
        plt.figure(figsize=(6,6))
        plt.imshow(recon_img, cmap="gray")
        plt.title("<3 (anatomical heart emoji)")
        plt.axis("off")
        plt.show()
