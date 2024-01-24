import mrcfile
import argparse
import numpy as np
import pyvista as pv 
import matplotlib.pyplot as plt

def render_volume(volume: np.ndarray, fn: str):
    pv.rcParams['transparent_background'] = True
    pl = pv.Plotter(off_screen=True)
    clim = [0.0, volume.max()]
    vol = pl.add_volume(volume, colormap='plasma', opacity='sigmoid', clim=clim, show_scalar_bar=False)
    vol.prop.interpolation_type = 'linear'
    
    pl.camera.enable_parallel_projection()
    pl.camera.azimuth = -30
    pl.camera.elevation = -30
    pl.camera.zoom(1.5)
    pl.set_background([1.0, 1.0, 1.0, 0.0])
    pl.save_graphic(fn)

def render_image(image: np.ndarray, fn: str):
    plt.imshow(image, cmap='plasma', vmin=0, vmax=image.max())
    plt.axis('off')
    plt.savefig(fn)
    

def main():
    parser = argparse.ArgumentParser(prog='3D Fourier visualization')
    parser.add_argument('-i', required=True)
    args = parser.parse_args()

    volume = mrcfile.read(args.i)
    volumeFourier = np.fft.fftshift(np.fft.fftn(volume, axes=(0, 1, 2)), axes=(0, 1, 2))
    volumeFourier = np.log(np.abs(volumeFourier))
    
    render_image(volume.sum(axis=2), 'projection.svg')
    render_image(volumeFourier[:,:,volumeFourier.shape[2]//2], 'fourier2d.svg')
    render_volume(volume, 'volume.svg')
    
    volumeFourier = np.where(volumeFourier > 7, volumeFourier, 0.0)
    render_volume(volumeFourier, 'fourier3d.svg')
    
if __name__ == '__main__':
    main()
