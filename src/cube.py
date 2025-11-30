import numpy as np

class SimCube:
    def __init__(self, delta, L=1000.):      
        N = delta.shape[0]
        N_Z = (N//2 +1)
            
        self.L = L
        self.l = L / N
        
        kx = 2*np.pi * np.fft.fftfreq(N, d=L/N)
        ky = 2*np.pi * np.fft.fftfreq(N, d=L/N)
        kz = 2*np.pi * np.fft.rfftfreq(N, d=L/N)

        # mesh of the 2D frequencies
        self.kx_grid = np.tile(kx[:, None, None], (1, N, N_Z))       
        self.ky_grid = np.tile(ky[None, :, None], (N, 1, N_Z))
        self.kz_grid = np.tile(kz[None, None, :], (N, N, 1))

        self.k_grid = np.sqrt(self.kx_grid**2 + self.ky_grid**2 + self.kz_grid**2 + 1e-40)
        
        self.delta = delta
        self.delta_fft = np.fft.rfftn(delta)
        
        self.smoothing_filter = 1. 
        
        self.compute_tidal_fields()

    def compute_tidal_fields(self):
        print("Computing tidal fields...")
        self.s_xx = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.kx_grid * self.kx_grid / self.k_grid**2 - 1./3))
        self.s_yy = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.ky_grid * self.ky_grid / self.k_grid**2 - 1./3))
        self.s_zz = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.kz_grid * self.kz_grid / self.k_grid**2 - 1./3))
        self.s_xy = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.kx_grid * self.ky_grid / self.k_grid**2))
        self.s_yz = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.ky_grid * self.kz_grid / self.k_grid**2))
        self.s_zx = np.fft.irfftn(self.smoothing_filter * self.delta_fft * (self.kz_grid * self.kx_grid / self.k_grid**2))
    
    def smooth_fields(self, smoothing_scale=None):
        if smoothing_scale is not None:
            self.smoothing_filter = np.exp(-0.5 * self.k_grid**2 * smoothing_scale**2)
        else:
            self.smoothing_filter = 1.
        self.compute_tidal_fields()
    
    def get_projected_delta_field(self, N_START, N_END, axis='z'):
        """
        Returns the projected 2d field from the 3d density field.
        """
        assert axis in ['x', 'y', 'z'], "axis must be one of x, y or z."
        if(axis=='x'):
            delta_2d = self.delta[N_START:N_END,:,:].mean(0)
        elif(axis=='y'):
            delta_2d = self.delta[:,N_START:N_END,:].mean(1)
        elif(axis=='z'):
            delta_2d = self.delta[:,:,N_START:N_END].mean(2)
        return delta_2d
    
    def get_all_delta_slabs(self, slab_width, axis='z'):
        """
        Returns all the projected 2d delta slabs along a certain axis
        """
        slab_index_width = int(slab_width / self.l)
        N_slabs          = int(self.L / slab_width)
    
        delta_2d_list = []
    
        for i in range(N_slabs):
            N_START = slab_index_width * i
            N_END   = slab_index_width * (i+1)
    
            delta_2d = self.get_projected_delta_field(N_START, N_END, axis)
            delta_2d_list.append(delta_2d)
    
        return delta_2d_list