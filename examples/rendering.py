import pydrr
import pydrr.autoinit
import mhd
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
from pydrr import utils

def main():
    # Load materials
    mhd_filename = 'image.mhd'
    volume, header = mhd.read(mhd_filename)
    spacing = header['ElementSpacing']
    spacing = spacing[::-1]

    #volume, spacing, _ = utils.load_volume(mhd_filename)
    volume = pydrr.utils.HU2Myu(volume - 700, 0.2683)

    pm_Nx3x4, image_size, image_spacing = load_test_projection_matrix()
    T_Nx4x4 = load_test_transform_matrix()

    # Construct objects
    volume_context = pydrr.VolumeContext(volume.astype(np.float32), spacing)
    geometry_context = pydrr.GeometryContext()
    geometry_context.projection_matrix = pm_Nx3x4

    n_channels = T_Nx4x4.shape[0] * pm_Nx3x4.shape[0]
    detector = pydrr.Detector(pydrr.Detector.make_detector_size(image_size, n_channels), image_spacing)
    # detector = pydrr.Detector.from_geometry(geometry_context, T_Nx4x4) # You can use from_geometry if you set pixel_size and image_size.
    projector = pydrr.Projector(detector, 1.0).to_gpu()

    # Host memory -> (Device memory) -> Texture memory
    t_volume_context = volume_context.to_texture()
    
    d_image = projector.project(t_volume_context, geometry_context, T_Nx4x4)

    # Device memory -> Host memory
    image = d_image.get()
    print('Result image shape:', image.shape)
    plt.figure(figsize=(16,9))
    n_show_channels = 3
    for i in range(min(image.shape[2], n_show_channels)):
        ax = plt.subplot(1, min(image.shape[2], n_show_channels), i+1)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        im = ax.imshow(image[:, :, i], interpolation='none', cmap='gray')
        fig.colorbar(im, cax=cax)
    plt.show()

def load_test_projection_matrix(SDD=2000, SOD=1800, image_size=[1280, 1280], spacing=[0.287, 0.287] ):

    if isinstance(image_size, list):
        image_size = np.array(image_size)

    if isinstance(spacing, list):
        spacing = np.array(spacing)

    extrinsic_R = utils.convertTransRotTo4x4([[0,0,0,90,0,0],
                                              [0,0,0,0,90,0],
                                              [0,0,0,0,0,90]])

    print('extrinsic_R:', extrinsic_R)
    print('extrinsic_R.shape:', extrinsic_R.shape)

    extrinsic_T = utils.convertTransRotTo4x4([0,0,-SOD,0,0,0])

    print('extrinsic_T:', extrinsic_T)
    print('extrinsic_T.shape:', extrinsic_T.shape)


    extrinsic = utils.concatenate4x4(extrinsic_T, extrinsic_R)

    print('extrinsic:', extrinsic)
    print('extrinsic.shape:', extrinsic.shape)


    intrinsic = np.array([[-SDD/spacing[0], 0, image_size[0]/2.0], # unit: [pixel]
                          [0, -SDD/spacing[1], image_size[1]/2.0],
                          [0,                0,               1]])

    print('intrinsic:', intrinsic)
    print('intrinsic.shape:', intrinsic.shape)


    pm_Nx3x4 = utils.constructProjectionMatrix(intrinsic, extrinsic)
    #pm_Nx3x4 = np.repeat(pm_Nx3x4, 400, axis=0)
    
    print('pm_Nx3x4:', pm_Nx3x4)
    print('pm_Nx3x4.shape:', pm_Nx3x4.shape)

    return pm_Nx3x4, image_size, spacing

def load_test_transform_matrix(n_channels=1):
    T_Nx6 = np.array([0,0,0,90,0,0])
    T_Nx6 = np.expand_dims(T_Nx6, axis=0)
    T_Nx6 = np.repeat(T_Nx6, n_channels, axis=0)
    T_Nx4x4 = utils.convertTransRotTo4x4(T_Nx6)

    print('T_Nx4x4:', T_Nx4x4)
    print('T_Nx4x4.shape:', T_Nx4x4.shape)

    return T_Nx4x4

if __name__ == '__main__':
    main()
