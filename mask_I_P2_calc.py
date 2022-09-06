import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as opt
from scipy import interpolate
import aotools
import poppy
import astropy.units as u


N_pix = 128 # number of pixels
nx_size = 1.0 # meter 
pixel_scale = nx_size / float(N_pix) # meter/pix
wl_laser = 589.0 # laser wavelength in nm
t_alt = np.array([0., 500., 1000., 2000., 4000., 8000., 16000.]) # Turbulence height in m
EL = np.array([30.0, 60.0, 90.0]) # telescope elevation in deg
EL_index = 2 # EL
X = (1.0/np.cos(np.radians(90.0-EL))) # Airmass

x,y = np.meshgrid(np.arange(0,N_pix,1), np.arange(0,N_pix,1))
R = np.sqrt((x-N_pix/2.)**2 + (y-N_pix/2.)**2)

w0 = 0.112 # m
w0_pix = w0 * math.sqrt(2.0) / pixel_scale # multiplyt by sqrt(2) to adjust the waist size in the fresnel propagator
gauss =np.exp(-2*R**2/w0_pix**2)


def make_beam_image_10arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2+0.0685*N_pix)**2 + (mask_Y-N_pix/2+0.0666*N_pix)**2)
    mask2_radius = 0.247  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = np.sqrt((mask_X-N_pix/2+0.0685*N_pix)**2 + (mask_Y-N_pix/2+0.0666*N_pix)**2)
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.138/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.143/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_20arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2+0.0628*N_pix)**2 + (mask_Y-N_pix/2+0.0642*N_pix)**2)
    mask2_radius = 0.247  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = np.sqrt((mask_X-N_pix/2+0.0628*N_pix)**2 + (mask_Y-N_pix/2+0.0642*N_pix)**2)
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.2043/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.1986/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_30arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2-0.0874*N_pix)**2 + (mask_Y-N_pix/2-0.0882*N_pix)**2)
    mask2_radius = 0.253  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = r_2
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    mask4_line = N_pix/2 - 0.1386/(1./128.)
    mask4 = np.copy(mask_X)*0.0
    idx4_1 = np.where(mask_X>=mask4_line)
    idx4_2 = np.where(mask_X<mask4_line)
    mask4[idx4_1] = 1.0
    mask4[idx4_2] = 0.0


    mask5_line = N_pix/2 - 0.1378/(1./128.)
    mask5 = np.copy(mask_Y)*0.0
    idx5_1 = np.where(mask_Y>=mask5_line)
    idx5_2 = np.where(mask_Y<mask5_line)
    mask5[idx5_1] = 1.0
    mask5[idx5_2] = 0.0

    image = gauss*mask1*mask2*mask3*mask4*mask5
    return image

def make_beam_image_40arcsec():
    # mask
    mask_x = np.arange(0, 128, 1)
    mask_y = np.arange(0, 128, 1)
    mask_X, mask_Y = np.meshgrid(mask_x, mask_y)


    r_1 = np.sqrt((mask_X-N_pix/2)**2 + (mask_Y-N_pix/2)**2)
    mask1_radius = 0.247  # m
    mask1 = np.copy(r_1)*0.0
    idx1_1 = np.where(r_1>=mask1_radius/ (1./128.))
    idx1_2 = np.where(r_1<mask1_radius/ (1./128.))
    mask1[idx1_1] = 0.0
    mask1[idx1_2] = 1.0


    r_2 = np.sqrt((mask_X-N_pix/2-0.1025*N_pix)**2 + (mask_Y-N_pix/2-0.1034*N_pix)**2)
    mask2_radius = 0.253  # m
    mask2 = np.copy(r_2)*0.0
    idx2_1 = np.where(r_2>=mask2_radius/ (1./128.))
    idx2_2 = np.where(r_2<mask2_radius/ (1./128.))
    mask2[idx2_1] = 0.0
    mask2[idx2_2] = 1.0


    r_3 = r_2
    mask3_radius = 0.026  # m
    mask3 = np.copy(r_3)*0.0
    idx3_1 = np.where(r_3>=mask3_radius/ (1./128.))
    idx3_2 = np.where(r_3<mask3_radius/ (1./128.))
    mask3[idx3_1] = 1.0
    mask3[idx3_2] = 0.0


    image = gauss*mask1*mask2*mask3
    return image

def make_phase_screen():
    # total Cn2 [m^1/3] 
    total_cn2 = 5.2770e-13 # m^(1/3), seeing condition: median

    # Wavelength in m
    lam = wl_laser * 1.0e-9 

    # Fried length in m 
    r0_total = (0.423 * (2.0*math.pi/lam)**2 * X * total_cn2)**(-3./5.)
    #print('r0_total = ', r0_total)

    # seeing FWHM in arcsec
    fwhm = np.rad2deg(1.028 * (lam / r0_total))*3600.0 

    # Cn2 fraction in %
    t_frac = np.array([73.16, 6.50, 1.93, 2.52, 5.74, 5.00, 5.15])

    # Cn2 profile in m^(1/3)
    cn2 = total_cn2 * t_frac / 100.0

    # r0 profile in m
    r0 = []
    for  a in X:
        r0.append((0.423 * (2.0*math.pi/lam)**2 * a * cn2)**(-3./5.))
    r0 = np.array(r0)

    # Outer scale in m
    L0 = 30.0 

    phscrn_el = []
    for i in range(len(EL)):
        
        phscrn_arr = []
        for j in range(len(t_alt)):
            phi = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(N_pix, pixel_scale, r0[i][j], L0)
            phscrn_arr.append(phi.scrn)
        
        phscrn_el.append(np.array(phscrn_arr))
        
    phscrn_el = np.array(phscrn_el)
    return phscrn_el

def gauss2d(pos, A, x0, y0, w, q, pa):  
    x = pos[0]
    y = pos[1]
    
    xc = (x - x0) * np.cos(np.deg2rad(-pa)) - (y - y0) * np.sin(np.deg2rad(-pa))
    yc = (x - x0) * np.sin(np.deg2rad(-pa)) + (y - y0) * np.cos(np.deg2rad(-pa))
    r = np.sqrt(xc**2 + (yc/q)**2)
    
    g = A * np.exp(-(2.0*r**2/(w**2)))
    
    return g.ravel()


start_image = make_beam_image_20arcsec()
phscrn_el_arr = []
for i in range(10):
    print('i = ', i)
    phscrn_el = make_phase_screen()
    phscrn_el_arr.append(phscrn_el)


# i: EL, j: t_alt
# (e.g. set the 0m phase screen for EL=60m => i = 1, j = 0)
opd = phscrn_el[EL_index][0] * (wl_laser*1.0e-9/(2.0*math.pi)) # Optical path difference in m
input_wave = poppy.ArrayOpticalElement(transmission=start_image, opd=opd*u.m, name='input wavefront', pixelscale=pixel_scale*u.m/u.pixel)
#input_wave.display(what='both')


plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 1.5

# =========================================================================================
I_P2_arr = []
for j in phscrn_el_arr:
    # Setup initial condition for the fresnel propagation
    wf = poppy.FresnelWavefront(0.5*u.m, wavelength=wl_laser*1.0e-9, npix=N_pix, oversample=4)
    wf *= input_wave

    # 2D gaussian fit
    Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
    initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
    popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
    gauss_fit = gauss2d((Xp,Yp), *popt)
    A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
    #print("Waist size = %.4f" % (w_fit*pixel_scale), "input")

    #fig = plt.figure(figsize=(12,6))

    # plot the cutout of the initial wavefront 
    """y, x = wf.coordinates()
    ax1 = fig.add_subplot(121)
    ax1.plot(x[wf.intensity.shape[1]//2,:], wf.intensity[wf.intensity.shape[1]//2,:], label='P0')
    ax1.set_xlabel('Position [m]')
    ax1.set_ylabel('Intensity profile')
    ax1.set_xlim(-0.5,0.5)
    ax1.plot(x[wf.intensity.shape[1]//2,:], np.zeros(512)+0.1353)
    ax1.grid()
    ax2 = fig.add_subplot(122)
    ax2.plot(x[wf.phase.shape[1]//2,:], wf.phase[wf.phase.shape[1]//2,:], label='P0')
    ax2.set_xlabel('Position [m]')
    ax2.set_ylabel('Phase distribution [rad]')
    ax2.set_xlim(-0.5,0.5)

    plt.tight_layout()"""


    w_arr = []
    L_arr = []
    w_arr.append(w_fit*pixel_scale)
    L_arr.append(0.0)
    title_arr = ["After propagation(0 ~ 0.5km)",
                "After propagation(0.5 ~ 1.0km)",
                "After propagation(1.0 ~ 2.0km)",
                "After propagation(2.0 ~ 4.0km)",
                "After propagation(4.0 ~ 8.0km)",
                "After propagation(8.0 ~ 16.0km)"]
    fig_name = ["after_prop_0_05km.png",
                "after_prop_05_1km.png",
                "after_prop_1_2km.png",
                "after_prop_2_4km.png",
                "after_prop_4_8km.png",
                "after_prop_8_16km.png"]
    # example for EL=60 deg
    for i in range(len(t_alt)-1):

        # propagation distance 
        dL = (t_alt[i+1] - t_alt[i]) * X[EL_index]  # distance to the next turbulence plane in m
        L_arr.append(t_alt[i+1]*X[EL_index])

        # propagetion
        wf.propagate_fresnel(dL*u.m)
        #plt.figure(figsize=(12,5))
        #wf.display("both", colorbar=True, showpadding=False, vmin=0, vmax=0.5)
        #plt.suptitle("%s" % title_arr[i], fontsize=18)
        #plt.savefig('%s.png' % fig_name[i])

        # show the wavefront cutout
        #y, x = wf.coordinates()
        #ax1.plot(x[wf.intensity.shape[1]//2,:], wf.intensity[wf.intensity.shape[1]//2,:], label='P%d' % (i+1))
        #ax2.plot(x[wf.phase.shape[1]//2,:], wf.phase[wf.phase.shape[1]//2,:], label='P%d' % (i+1))

        # 2D gaussian fit
        Xp,Yp = np.meshgrid(np.arange(0,N_pix*4,1), np.arange(0,N_pix*4,1))
        initial_guess = (1.0, 2.0*N_pix, 2.0*N_pix, w0/pixel_scale, 1.0, 0)
        popt, pcov = opt.curve_fit(gauss2d, (Xp,Yp), wf.intensity.ravel(), p0 = initial_guess)
        gauss_fit = gauss2d((Xp,Yp), *popt)
        A_fit,x0_fit,y0_fit,w_fit,q_fit,pa_fit = popt
        w_arr.append(w_fit*pixel_scale)
        #print("Waist size = %.4f" % (w_fit*pixel_scale), t_alt[i+1])

        # add phase delay
        ph = poppy.ArrayOpticalElement(transmission=(start_image*0.0+1.0), opd=phscrn_el[1][i+1] * (wl_laser*1.0e-9/(2.0*math.pi)), name='', pixelscale=pixel_scale*u.m/u.pixel)
        #ph.display(what='both')
        wf *= ph


    # last propagation until the height of the Na layer (H=90km) 
    dL = (90.0e+3 - t_alt[len(t_alt)-1]) * X[EL_index]

    # propagetion
    wf.propagate_fresnel(dL*u.m)
    #plt.figure(figsize=(8,8))
    #wf.display("intensity", colorbar=True, showpadding=False, vmin=0, vmax=0.5)
    #wf.display("both", colorbar=True, showpadding=False, vmin=0, vmax=0.5)
    #plt.suptitle("After propagation(16.0 ~ 90.0km)", fontsize=18)
    #plt.savefig('after_prop_16_90km.png')





    P1 = 4.0 
    pix_scale = 1.0/128.0 # m/pix
    total = np.sum(wf.intensity)
    wf_n = wf.intensity / total 
    I_peak = np.max(wf_n) 
    IA = np.linspace(0,I_peak,100)

    #print(Pea)
    fla = []
    for I in IA: 
        idx = np.where(wf_n > I)
        fla.append(np.sum(wf_n[idx]))
        
    ip = interpolate.interp1d(fla, IA)

    IP2 = ip(0.5)
    I_P2_arr.append(IP2)

plt.figure(figsize=(8,8))
IP2_average = np.mean(I_P2_arr)
print(IP2_average*P1/(pix_scale**2), "[W/m^2]") # W/m^2 

plt.plot(IA,fla)



plt.show()