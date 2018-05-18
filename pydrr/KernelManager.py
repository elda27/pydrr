from pycuda import driver, compiler, gpuarray, tools 

from pydrr import kernels
from .KernelModule import KernelModule

class KernelManager:
    Kernel = None
    Module = None
    Modules = []

    def __init__(self, default_kernel, *kernel_codes):
        for kernel_code, kernel_info in kernel_codes:
            KernelManager.Modules.append(KernelModule(kernel_code, kernel_info)) 

        KernelManager.Module = KernelManager.Modules[0]
        KernelManager.Kernel = KernelManager.Module.get_kernel(default_kernel)

_manager = KernelManager('render_with_linear_interp', 
    (
        kernels.render_kernel, 
        {
            'render_with_linear_interp': 
            {
                'global': [
                    'd_step_size_mm',
                    'd_image_size',
                    'd_volume_spacing',
                    'd_volume_corner_mm',
                ],
                'texture' : [
                    't_volume',
                    't_proj_param_Nx12',
                ],
            },
            'print_device_params': { 'global':[], 'texture':['t_proj_param_Nx12'] }
        }
    ),
)
