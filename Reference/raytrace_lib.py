import numpy as np
import math

class raytrace:
    def __init__(self):

        self.input_ray_dir = np.zeros(3)    # Direction vector of the input ray (unit vector)
        self.input_ray_pos = np.zeros(3)    # XYZ coordinates at the starting point of the input ray [mm]
        self.surface_pos = np.zeros(3)      # XYZ coordinates of the arbitrary position on the surface [mm]
        self.surface_txy = np.zeros(2)      # X/Y-tilt angle of the surface (rotation about Z-axis/X-axis), txy = [tx, ty][radian]
        self.output_ray_dir = np.zeros(3)   # Direction vector of the output ray (unit vector)
        self.output_ray_pos = np.zeros(3)   # XYZ coordinates at the starting point of the output ray [mm]

        # Reflective index before and after entering the surface (transmission only)
        self.n1 = 1.0  # before
        self.n2 = 1.0  # after 
    
    def sdot(self, a, b): # Calculate inner product
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def calc_norm(self, txy): # Compute a unit vector with a tilt txy = [tx, ty] 
        return np.array([math.sin(txy[0]) * math.cos(txy[1]), math.cos(txy[0]) * math.cos(txy[1]), math.sin(txy[1])])
        
    def reflect(self):  # Compute reflection vector and origin of the reflected ray

        norm = self.calc_norm(self.surface_txy)
        c = self.sdot(self.input_ray_dir, norm)
        t = (self.sdot(self.surface_pos, norm) - self.sdot(self.input_ray_pos, norm))/c

        self.output_ray_dir = self.input_ray_dir - 2*c*norm
        self.output_ray_pos = self.input_ray_pos + t * self.input_ray_dir  
        
    def refract(self): # Compute refraction vector and origin of the refracted ray

        norm = self.calc_norm(self.surface_txy)
        c = self.sdot(self.input_ray_dir, norm)
        t = (self.sdot(self.surface_pos, norm) - self.sdot(self.input_ray_pos, norm))/c
        
        g = math.sqrt((self.n2/self.n1)**2 + c**2 - 1.0)
        self.output_ray_dir = (self.n1/self.n2)*(self.input_ray_dir - (c+g)*norm)
        self.output_ray_pos = self.input_ray_pos + t * self.input_ray_dir