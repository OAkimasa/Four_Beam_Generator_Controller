import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from VectorFunctions import VectorFunctions
from optics_params import all_params
from generate_TTM_nV import generate_TTM_nV_ans

# ---- 単位は mm ----

print('\n----------------START----------------\n')


N_air = 1.0  # Refractive index of the Air
N_Fused_Silica = 1.458  # Refractive index (Fused Silica at 589nm)
N_BSL7 = 1.516  # Refractive index (BSL7 at 589nm)
t_bs = 6.0      # Beam Splitter substrate thickness [mm]
BE_power = 3  # BE magnifying power
LLT_power = 12.5  # LLT magnifying power
BE_pupill_R = 18  # BE pupil radius [mm]（仮）

IMR_rot_deg = 0  # IMR rotation angle [deg]

# target arcsec =================================================
# OK --> [4.5" <= target_arcsec <= 11.5", 14" <= target_arcsec <= 23"]
target_arcsec = 10
arcsec_ray1 = arcsec_ray2 = arcsec_ray3 = arcsec_ray4 = target_arcsec
# ===============================================================

# params
all_params = all_params()
FM_params = all_params[0]
BS_params = all_params[1]
TTM1 = all_params[2]
TTM2 = all_params[3]

TTM1_params = [[TTM1[0][0], generate_TTM_nV_ans(arcsec_ray1)[0][0], 25.4/2],
                [TTM1[1][0], generate_TTM_nV_ans(arcsec_ray2)[0][1], 25.4/2],
                [TTM1[2][0], generate_TTM_nV_ans(arcsec_ray3)[0][2], 25.4/2],
                [TTM1[3][0], generate_TTM_nV_ans(arcsec_ray4)[0][3], 25.4/2]]
TTM2_params = [[TTM2[0][0], generate_TTM_nV_ans(arcsec_ray1)[1][0], 25.4/2],
                [TTM2[1][0], generate_TTM_nV_ans(arcsec_ray2)[1][1], 25.4/2],
                [TTM2[2][0], generate_TTM_nV_ans(arcsec_ray3)[1][2], 25.4/2],
                [TTM2[3][0], generate_TTM_nV_ans(arcsec_ray4)[1][3], 25.4/2]]

for i in TTM1_params:
    nV = i[1]/np.linalg.norm(i[1])
    print(nV)
print("TTM1_params")
for i in TTM2_params:
    nV = i[1]/np.linalg.norm(i[1])
    print(nV)
print("TTM2_params")
PRISM_front_params = all_params[4]
PRISM_back_params = all_params[5]
FBM_params = all_params[6]
BE_entrance_pupill_params = all_params[7]
M5_params = all_params[8]
DM1234_params = all_params[9]
IMR_params = all_params[10]
DM56_params = all_params[11]
BE_G123_params = all_params[12]
M678_params = all_params[13]
LLT_Window_params = all_params[14]
LLT_M321_params = all_params[15]
LLT_Exit_Window_params = all_params[16]
Evaluation_Plane_params = all_params[17]




# ミラーの光線追跡を行う関数
def mirror_reflection():
    # ミラー描画
    def plot_mirror(params):
        X_center = params[0][0]
        Y_center = params[0][1]
        Z_center = params[0][2]
        normalV = params[1]
        R = params[2]

        # 円盤生成
        geneNum = 200
        x = np.linspace(X_center-R, X_center+R, geneNum)
        y = np.linspace(Y_center-R, Y_center+R, geneNum)
        X,Y = np.meshgrid(x, y)
        if normalV[2] == 0:
            Z = Z_center - (normalV[0]*(X-X_center) + normalV[1]*(Y-Y_center)) / 0.01
        else:
            Z = Z_center - (normalV[0]*(X-X_center) + normalV[1]*(Y-Y_center)) / normalV[2]
        for i in range(geneNum):
            for j in range(geneNum):
                if (X[i][j]-X_center)**2 + (Y[i][j]-Y_center)**2 + (Z[i][j]-Z_center)**2 > R**2:
                    Z[i][j] = np.nan

        ax.quiver(X_center, Y_center, Z_center, normalV[0], normalV[1], normalV[2], color='black', length=50)
        ax.plot_wireframe(X, Y, Z, color='gray',linewidth=0.3)

    # レンズ描画
    def plot_lens(params):
        geneNum = 300
        limitTheta = 2*np.pi  # theta生成数
        limitPhi = np.pi  # phi生成数
        theta = np.linspace(0, limitTheta, geneNum)
        phi = np.linspace(0, limitPhi, geneNum)
        Xs = np.outer(np.sin(theta), np.sin(phi))
        Ys = np.outer(np.ones(np.size(theta)), np.cos(phi))

        Xs1 = params[2] * Xs
        Ys1 = params[2] * Ys
        if params[3] < 0:
            Zs1 = -(params[3]**2-Xs1**2-Ys1**2)**0.5 - params[3]
            ax.plot_wireframe(Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)
        elif params[3] > 0:
            Zs1 = (params[3]**2-Xs1**2-Ys1**2)**0.5 - params[3]
            ax.plot_wireframe(Xs1+params[0][0], Ys1+params[0][1], Zs1+params[0][2], linewidth=0.1)

    # 放物線描画
    def plot_parabola(params):
        theta = np.linspace(0, 2*np.pi, 100)
        R = params[2]
        a = abs(1/(2*params[3]))
        x1 = R*np.cos(theta)
        z1 = R*np.sin(theta)
        X1,Z1 = np.meshgrid(x1, z1)
        Y1 = a*X1**2 + a*Z1**2
        for i in range(100):
            for j in range(100):
                if (X1[i][j])**2 + Z1[i][j]**2 > R**2:
                    Y1[i][j] = np.nan
                else:
                    Y1[i][j] = a*X1[i][j]**2 + a*Z1[i][j]**2
        ax.plot_wireframe(X1+params[0][0], Y1+params[0][1], Z1+params[0][2], color='b',linewidth=0.1)

    for i in FM_params:
        plot_mirror(i)
    for i in BS_params:
        plot_mirror(i)
    for i in TTM1_params:
        plot_mirror(i)
    for i in TTM2_params:
        plot_mirror(i)
    for i in PRISM_front_params:
        plot_mirror(i)
    for i in PRISM_back_params:
        plot_mirror(i)
    for i in FBM_params:
        plot_mirror(i)
    #for i in BE_entrance_pupill_params:
    #    plot_mirror(i)
    for i in M5_params:
        plot_mirror(i)
    for i in DM1234_params:
        plot_mirror(i)

    IMR_params = [
        [[430.000, 290.600, -542.000], np.array([-np.sin(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), np.cos(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), -1/np.sqrt(2)]), 20],
        [[430+10*np.sin(IMR_rot_deg*np.pi/180), 290.600-10*np.cos(IMR_rot_deg*np.pi/180), -502.100], np.array([-np.sin(IMR_rot_deg*np.pi/180), np.cos(IMR_rot_deg*np.pi/180), 0]), 30],
        [[430, 290.600, -472.200], np.array([np.sin(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), -np.cos(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), -1/np.sqrt(2)]), 20]
        ]
    for i in IMR_params:
        plot_mirror(i)
    for i in DM56_params:
        plot_mirror(i)
    # lens-------------------------------
    for i in BE_G123_params:
        plot_lens(i)
    # -----------------------------------
    for i in M678_params:
        plot_mirror(i)
    for i in LLT_Window_params:
        plot_mirror(i)
    for i in [LLT_M321_params[0]]:
        plot_mirror(i)
    # parabola---------------------------
    for i in [LLT_M321_params[1], LLT_M321_params[2]]:
        plot_parabola(i)
    # -----------------------------------
    for i in LLT_Exit_Window_params:
        plot_mirror(i)
    for i in Evaluation_Plane_params:
        plot_mirror(i)


    # 光線追跡
    VF.plotLineBlack([-200,90.6,0], [1955.5,90.6,0])  # 仮
    VF.plotLineBlack([-100,0,0], [400,0,0])  # 仮

    # 初期値
    VF.ray_start_pos = np.array([0.0, 0.0, 0.0])
    VF.ray_start_dir = np.array([1.0, 0.0, 0.0])

    # BS
    def BS_raytrace():
        last_pos = []  # 仮
        last_dir = []  # 仮
        # ----------------BS1----------------
        pos_BS1_f = BS_params[0][0]
        nV_BS1_f = BS_params[0][1]
        nV_BS1_f = nV_BS1_f / np.linalg.norm(nV_BS1_f)

        VF.raytrace_plane(pos_BS1_f, nV_BS1_f)
        VF.reflect(nV_BS1_f)
        VF.plotLineBlue()
        last_pos.append(VF.ray_end_pos)  # 仮
        last_dir.append(VF.ray_end_dir)  # 仮


        VF.refract(nV_BS1_f, N_air, N_Fused_Silica)
        pos_BS1_b = BS_params[1][0]
        nV_BS1_b = BS_params[1][1]
        nV_BS1_b = nV_BS1_b / np.linalg.norm(nV_BS1_b)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS1_b, nV_BS1_b)
        VF.refract(nV_BS1_b, N_Fused_Silica, N_air)
        VF.plotLineGreen()

        # ----------------BS2----------------
        pos_BS2_f = BS_params[2][0]
        nV_BS2_f = BS_params[2][1]
        nV_BS2_f = nV_BS2_f / np.linalg.norm(nV_BS2_f)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS2_f, nV_BS2_f)
        VF.reflect(nV_BS2_f)
        VF.plotLineGreen()
        last_pos.append(VF.ray_end_pos)  # 仮
        last_dir.append(VF.ray_end_dir)  # 仮


        VF.refract(nV_BS2_f, N_air, N_Fused_Silica)
        pos_BS2_b = BS_params[3][0]
        nV_BS2_b = BS_params[3][1]
        nV_BS2_b = nV_BS2_b / np.linalg.norm(nV_BS2_b)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS2_b, nV_BS2_b)
        VF.refract(nV_BS2_b, N_Fused_Silica, N_air)
        VF.plotLineRed()

        # ----------------BS3----------------
        pos_BS3_f = BS_params[4][0]
        nV_BS3_f = BS_params[4][1]
        nV_BS3_f = nV_BS3_f / np.linalg.norm(nV_BS3_f)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS3_f, nV_BS3_f)
        VF.reflect(nV_BS3_f)
        VF.plotLineRed()
        last_pos.append(VF.ray_end_pos)  # 仮
        last_dir.append(VF.ray_end_dir)  # 仮


        VF.refract(nV_BS3_f, N_air, N_Fused_Silica)
        pos_BS3_b = BS_params[5][0]
        nV_BS3_b = BS_params[5][1]
        nV_BS3_b = nV_BS3_b / np.linalg.norm(nV_BS3_b)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS3_b, nV_BS3_b)
        VF.refract(nV_BS3_b, N_Fused_Silica, N_air)
        VF.plotLineOrange()

        # ----------------BS4----------------
        pos_BS4_f = BS_params[6][0]
        nV_BS4_f = BS_params[6][1]
        nV_BS4_f = nV_BS4_f / np.linalg.norm(nV_BS4_f)

        VF.ray_start_pos = VF.ray_end_pos
        VF.ray_start_dir = VF.ray_end_dir
        VF.raytrace_plane(pos_BS4_f, nV_BS4_f)
        VF.reflect(nV_BS4_f)
        VF.plotLineOrange()
        last_pos.append(VF.ray_end_pos)  # 仮
        last_dir.append(VF.ray_end_dir)  # 仮

        return last_pos, last_dir

    # TTM1
    def TTM1_raytrace():
        relay_params = BS_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_TTM1 = []
        dir_TTM1 = []
        for i in range(4):
            nV_TTM1 = TTM1_params[i][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(TTM1_params[i][0], nV_TTM1)
            VF.reflect(nV_TTM1)
            pos_TTM1.append(VF.ray_end_pos)
            dir_TTM1.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_TTM1, dir_TTM1

    # TTM2
    def TTM2_raytrace():
        relay_params = TTM1_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_TTM2 = []
        dir_TTM2 = []
        for i in range(4):
            nV_TTM2 = TTM2_params[i][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(TTM2_params[i][0], nV_TTM2)
            VF.reflect(nV_TTM2)
            pos_TTM2.append(VF.ray_end_pos)
            dir_TTM2.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_TTM2, dir_TTM2

    # PRISM
    def PRISM_front_raytrace():
        relay_params = TTM2_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_PRISM_front = []
        dir_PRISM_front = []
        for i in range(4):
            nV_PRISM_front = PRISM_front_params[i][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(PRISM_front_params[i][0], nV_PRISM_front)
            VF.refract(nV_PRISM_front, N_air, N_Fused_Silica)
            pos_PRISM_front.append(VF.ray_end_pos)
            dir_PRISM_front.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_PRISM_front, dir_PRISM_front

    def PRISM_back_raytrace():
        relay_params = PRISM_front_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_PRISM_back = []
        dir_PRISM_back = []
        for i in range(4):
            nV_PRISM_back = PRISM_back_params[i][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(PRISM_back_params[i][0], nV_PRISM_back)
            VF.refract(nV_PRISM_back, N_Fused_Silica, N_air)
            pos_PRISM_back.append(VF.ray_end_pos)
            dir_PRISM_back.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_PRISM_back, dir_PRISM_back

    # FBM
    def FBM_raytrace():
        relay_params = PRISM_back_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_FBM = []
        dir_FBM = []
        for i in range(4):
            nV_FBM = FBM_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(FBM_params[0][0], nV_FBM)
            VF.reflect(nV_FBM)
            pos_FBM.append(VF.ray_end_pos)
            dir_FBM.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_FBM, dir_FBM

    # skip BE entrance pupil
    # M5
    def M5_raytrace():
        relay_params = FBM_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_M5 = []
        dir_M5 = []
        for i in range(4):
            nV_M5 = M5_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(M5_params[0][0], nV_M5)
            VF.reflect(nV_M5)
            pos_M5.append(VF.ray_end_pos)
            dir_M5.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_M5, dir_M5

    # DM1234
    def DM1234_raytrace():
        relay_params = M5_raytrace()

        # DM1
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM1234_params[0][0], nV_DM1234)
            VF.reflect(nV_DM1234)
            pos_DM1234.append(VF.ray_end_pos)
            dir_DM1234.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # DM2
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM1234_params[1][0], nV_DM1234)
            VF.reflect(nV_DM1234)
            pos_DM1234.append(VF.ray_end_pos)
            dir_DM1234.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # DM3
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[2][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM1234_params[2][0], nV_DM1234)
            VF.reflect(nV_DM1234)
            pos_DM1234.append(VF.ray_end_pos)
            dir_DM1234.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # DM4
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[3][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM1234_params[3][0], nV_DM1234)
            VF.reflect(nV_DM1234)
            pos_DM1234.append(VF.ray_end_pos)
            dir_DM1234.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_DM1234, dir_DM1234

    # IMR
    def IMR_raytrace():
        relay_params = DM1234_raytrace()

        # IMR-Front
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = np.array([-np.sin(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), np.cos(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), -1/np.sqrt(2)])
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(IMR_params[0][0], nV_IMR)
            VF.refract(nV_IMR, N_air, N_Fused_Silica)
            pos_IMR.append(VF.ray_end_pos)
            dir_IMR.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # IMR-Mid
        last_pos = pos_IMR
        last_dir = dir_IMR
        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = IMR_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(IMR_params[1][0], nV_IMR)
            VF.reflect(nV_IMR)
            pos_IMR.append(VF.ray_end_pos)
            dir_IMR.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # IMR-Back
        last_pos = pos_IMR
        last_dir = dir_IMR
        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = IMR_params[2][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(IMR_params[2][0], nV_IMR)
            VF.refract(nV_IMR, N_Fused_Silica, N_air)
            pos_IMR.append(VF.ray_end_pos)
            dir_IMR.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_IMR, dir_IMR

    # DM56
    def DM56_raytrace():
        IMR_params = IMR_raytrace()
        #IMR_params = DM1234_raytrace()  # IMR skip

        # DM5
        last_pos = IMR_params[0]
        last_dir = IMR_params[1]
        pos_DM56 = []
        dir_DM56 = []
        for i in range(4):
            nV_DM56 = DM56_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM56_params[0][0], nV_DM56)
            VF.reflect(nV_DM56)
            pos_DM56.append(VF.ray_end_pos)
            dir_DM56.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # DM6
        last_pos = pos_DM56
        last_dir = dir_DM56
        pos_DM56 = []
        dir_DM56 = []
        for i in range(4):
            nV_DM56 = DM56_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(DM56_params[1][0], nV_DM56)
            VF.reflect(nV_DM56)
            pos_DM56.append(VF.ray_end_pos)
            dir_DM56.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_DM56, dir_DM56

    # BE_G123
    def BE_G123_raytrace():
        DM56_params = DM56_raytrace()

        # BE_G1
        last_pos = DM56_params[0]
        last_dir = DM56_params[1]
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G1f = BE_G123_params[0][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[0][0], lens_R_G1f)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G1f]) - BE_G123_params[0][0]
            VF.refract(-nV_BE_G123, N_air, N_Fused_Silica)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G1b = BE_G123_params[1][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[1][0], lens_R_G1b)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G1b]) - BE_G123_params[1][0]
            VF.refract(nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # BE_G2
        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G2f = BE_G123_params[2][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[2][0], lens_R_G2f)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G2f]) - BE_G123_params[2][0]
            VF.refract(-nV_BE_G123, N_air, N_Fused_Silica)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G2b = BE_G123_params[3][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[3][0], lens_R_G2b)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G2b]) - BE_G123_params[3][0]
            VF.refract(-nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # BE_G3
        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G3f = BE_G123_params[4][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[4][0], lens_R_G3f)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G3f]) - BE_G123_params[4][0]
            VF.refract(-nV_BE_G123, N_air, N_Fused_Silica)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        lens_R_G3b = BE_G123_params[5][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_sphere(BE_G123_params[5][0], lens_R_G3b)
            nV_BE_G123 = VF.ray_end_pos + np.array([0, 0, lens_R_G3b]) - BE_G123_params[5][0]
            VF.refract(-nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(VF.ray_end_pos)
            dir_BE_G123.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_BE_G123, dir_BE_G123

    # M678
    def M678_raytrace():
        BE_G123_params = BE_G123_raytrace()

        # M6
        last_pos = BE_G123_params[0]
        last_dir = BE_G123_params[1]
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(M678_params[0][0], nV_M678)
            VF.reflect(nV_M678)
            pos_M678.append(VF.ray_end_pos)
            dir_M678.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # M7
        last_pos = pos_M678
        last_dir = dir_M678
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(M678_params[1][0], nV_M678)
            VF.reflect(nV_M678)
            pos_M678.append(VF.ray_end_pos)
            dir_M678.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # M8
        last_pos = pos_M678
        last_dir = dir_M678
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[2][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(M678_params[2][0], nV_M678)
            VF.reflect(nV_M678)
            pos_M678.append(VF.ray_end_pos)
            dir_M678.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_M678, dir_M678

    # LLT_Window
    def LLT_Window_raytrace():
        M678_params = M678_raytrace()

        last_pos = M678_params[0]
        last_dir = M678_params[1]
        pos_LLT_Window = []
        dir_LLT_Window = []
        for i in range(4):
            nV_LLT_Window = LLT_Window_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(LLT_Window_params[0][0], nV_LLT_Window)
            VF.refract(nV_LLT_Window, N_air, N_BSL7)
            pos_LLT_Window.append(VF.ray_end_pos)
            dir_LLT_Window.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        last_pos = pos_LLT_Window
        last_dir = dir_LLT_Window
        pos_LLT_Window = []
        dir_LLT_Window = []
        for i in range(4):
            nV_LLT_Window = LLT_Window_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(LLT_Window_params[1][0], nV_LLT_Window)
            VF.refract(nV_LLT_Window, N_BSL7, N_air)
            pos_LLT_Window.append(VF.ray_end_pos)
            dir_LLT_Window.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_LLT_Window, dir_LLT_Window

    # LLT_M321
    def LLT_M321_raytrace():
        LLT_Window_params = LLT_Window_raytrace()

        # LLT_M3
        last_pos = LLT_Window_params[0]
        last_dir = LLT_Window_params[1]
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        for i in range(4):
            nV_LLT_M321 = LLT_M321_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(LLT_M321_params[0][0], nV_LLT_M321)
            VF.reflect(nV_LLT_M321)
            pos_LLT_M321.append(VF.ray_end_pos)
            dir_LLT_M321.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # LLT_M2
        last_pos = pos_LLT_M321
        last_dir = dir_LLT_M321
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        parabola_R_LLT_M2 = LLT_M321_params[1][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_parabola(LLT_M321_params[1][0], parabola_R_LLT_M2)
            nV_LLT_M321 = VF.calcNormal_parabola(LLT_M321_params[1][0], parabola_R_LLT_M2)
            #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
            VF.reflect(nV_LLT_M321)
            pos_LLT_M321.append(VF.ray_end_pos)
            dir_LLT_M321.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        # LLT_M1
        last_pos = pos_LLT_M321
        last_dir = dir_LLT_M321
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        parabola_R_LLT_M1 = LLT_M321_params[2][3]
        for i in range(4):
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_parabola(LLT_M321_params[2][0], parabola_R_LLT_M1)
            nV_LLT_M321 = VF.calcNormal_parabola(LLT_M321_params[2][0], parabola_R_LLT_M1)
            #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
            VF.reflect(nV_LLT_M321)
            pos_LLT_M321.append(VF.ray_end_pos)
            dir_LLT_M321.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_LLT_M321, dir_LLT_M321

    # LLT_Exit_Window
    def LLT_Exit_Window_raytrace():
        LLT_M321_params = LLT_M321_raytrace()

        # LLT_Exit_Window
        last_pos = LLT_M321_params[0]
        last_dir = LLT_M321_params[1]
        pos_LLT_Exit_Window = []
        dir_LLT_Exit_Window = []
        for i in range(4):
            nV_LLT_Exit_Window = LLT_Exit_Window_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(LLT_Exit_Window_params[0][0], nV_LLT_Exit_Window)
            VF.refract(nV_LLT_Exit_Window, N_air, N_BSL7)
            pos_LLT_Exit_Window.append(VF.ray_end_pos)
            dir_LLT_Exit_Window.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        last_pos = pos_LLT_Exit_Window
        last_dir = dir_LLT_Exit_Window
        pos_LLT_Exit_Window = []
        dir_LLT_Exit_Window = []
        for i in range(4):
            nV_LLT_Exit_Window = LLT_Exit_Window_params[1][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(LLT_Exit_Window_params[1][0], nV_LLT_Exit_Window)
            VF.refract(nV_LLT_Exit_Window, N_BSL7, N_air)
            pos_LLT_Exit_Window.append(VF.ray_end_pos)
            dir_LLT_Exit_Window.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)

        return pos_LLT_Exit_Window, dir_LLT_Exit_Window

    # Evaluation_Plane
    def Evaluation_Plane_raytrace():
        LLT_Exit_Window_params = LLT_Exit_Window_raytrace()

        # Evaluation_Plane
        last_pos = LLT_Exit_Window_params[0]
        last_dir = LLT_Exit_Window_params[1]
        pos_Evaluation_Plane = []
        dir_Evaluation_Plane = []
        for i in range(4):
            nV_Evaluation_Plane = Evaluation_Plane_params[0][1]
            VF.ray_start_pos = last_pos[i]
            VF.ray_start_dir = last_dir[i]
            VF.raytrace_plane(Evaluation_Plane_params[0][0], nV_Evaluation_Plane)
            VF.ray_end_dir = VF.ray_start_dir
            pos_Evaluation_Plane.append(VF.ray_end_pos)
            dir_Evaluation_Plane.append(VF.ray_end_dir)
            VF.plotFourBeamLine(i)


        rad_list = []
        for i in range(4):
            rad_list.append(np.arccos(-np.dot(last_dir[i], Evaluation_Plane_params[0][1])))
        deg_list = np.degrees(rad_list)
        #print(deg_list)

        print("\n(Evaluation_Plane)")
        print("IMR_rot_deg:", IMR_rot_deg, "deg\n")
        print("ray1(blue):", pos_Evaluation_Plane[0], dir_Evaluation_Plane[0]/np.linalg.norm(dir_Evaluation_Plane[0]), np.round(deg_list[0]*3600, 6), "arcsec")
        print("ray2(green):", pos_Evaluation_Plane[1], dir_Evaluation_Plane[1]/np.linalg.norm(dir_Evaluation_Plane[1]), np.round(deg_list[1]*3600, 6), "arcsec")
        print("ray3(red):", pos_Evaluation_Plane[2], dir_Evaluation_Plane[2]/np.linalg.norm(dir_Evaluation_Plane[2]), np.round(deg_list[2]*3600, 6), "arcsec")
        print("ray4(orange):", pos_Evaluation_Plane[3], dir_Evaluation_Plane[3]/np.linalg.norm(dir_Evaluation_Plane[3]), np.round(deg_list[3]*3600, 6), "arcsec\n")

        print("ray1-ray4:", np.round(np.degrees(np.arccos(np.dot(last_dir[0], last_dir[3])))*3600, 6), "arcsec")
        print("ray2-ray3:", np.round(np.degrees(np.arccos(np.dot(last_dir[1], last_dir[2])))*3600, 6), "arcsec\n")

        """print("cf.  5 arcsec => 0.001389 deg, 10 arcsec => 0.002778 deg")
        print("cf. 15 arcsec => 0.004167 deg, 20 arcsec => 0.005556 deg")
        print("cf. 25 arcsec => 0.006944 deg, 30 arcsec => 0.008333 deg")
        print("cf. 35 arcsec => 0.009722 deg, 40 arcsec => 0.011111 deg\n")"""

        return pos_Evaluation_Plane, dir_Evaluation_Plane

    relay_param = Evaluation_Plane_raytrace()
    last_pos = relay_param[0]
    last_dir = relay_param[1]

    end_pos = []
    end_dir = []
    for i in range(4):
        VF.ray_start_pos = last_pos[i]
        VF.ray_start_dir = last_dir[i]
        VF.ray_end_pos = last_pos[i] + last_dir[i]*2000
        VF.ray_end_dir = last_dir[i]
        end_pos.append(VF.ray_end_pos)
        end_dir.append(VF.ray_end_dir)
        VF.plotFourBeamLine(i)

    VF.plotLineBlack([60.000, -250.400, -370.000], [60.000, 2000, -370.000])

    #print(Evaluation_Plane_params[0][0], "evaluation Plane")
    print(np.round(np.linalg.norm(last_pos[0]-np.array([Evaluation_Plane_params[0][0]])), 6), "radius ray1")
    print(np.round(np.linalg.norm(last_pos[1]-np.array([Evaluation_Plane_params[0][0]])), 6), "radius ray2")
    print(np.round(np.linalg.norm(last_pos[2]-np.array([Evaluation_Plane_params[0][0]])), 6), "radius ray3")
    print(np.round(np.linalg.norm(last_pos[3]-np.array([Evaluation_Plane_params[0][0]])), 6), "radius ray4\n")
    #print(last_dir, "last_dir")

    # 描画範囲
    LX = 300
    LY = 300
    LZ = 300
    ax.set_xlim(-LX+60, LX+60)
    ax.set_ylim(-LY+0, LY+0)
    ax.set_zlim(-LZ-370, LZ-370)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    ax.view_init(elev=24, azim=45)



if __name__ == "__main__":
    start = time.time()
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.aspect = 'equal'

    # インスタンス生成
    VF = VectorFunctions()
    VF.ax = ax

    mirror_reflection()

    print('\ntime =', round(time.time()-start, 5), 'sec')
    print('\n----------------END----------------\n')
    #plt.savefig("view_system_%darcsec.png" % target_arcsec, dpi=350)
    plt.show()