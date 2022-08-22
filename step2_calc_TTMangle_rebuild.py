import numpy as np
from scipy.optimize import fmin, fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


from optics_params import all_params

from step1_target_params import target_params
from step2_target_params import target_params_step2
from step2_result_viewer import result_viewer

# ---- 単位は mm ----


N_air = 1.0  # Refractive index of the Air
N_Fused_Silica = 1.458  # Refractive index (Fused Silica at 589nm)
N_BSL7 = 1.516  # Refractive index (BSL7 at 589nm)
t_bs = 6.0      # Beam Splitter substrate thickness [mm]
BE_power = 3  # BE magnifying power
LLT_power = 12.5  # LLT magnifying power
BE_pupill_R = 18  # BE pupil radius [mm]（仮）

IMR_rot_deg = 0  # IMR rotation angle [deg]

# params
all_params = all_params()
FM_params = all_params[0]
BS_params = all_params[1]
TTM1_params = all_params[2]
TTM2_params = all_params[3]
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


# 反射を計算する class
class VectorFunctions:
    # 受け取ったx,y,z座標から(x,y,z)の組を作る関数
    def makePoints(self, point0, point1, point2, shape0, shape1):
        result = [None]*(len(point0)+len(point1)+len(point2))
        result[::3] = point0
        result[1::3] = point1
        result[2::3] = point2
        result = np.array(result)
        result = result.reshape(shape0, shape1)
        return result

    # レイトレーシング平板との交点を持つときの係数Tを計算する関数
    def calcT_mirror(self, ray_pos, ray_dir, centerV, normalV):
        nV = np.array(normalV)/np.linalg.norm(normalV)
        T = (np.dot(centerV, nV)-np.dot(ray_pos, nV)) / (np.dot(ray_dir, nV))
        return T

    # レイトレーシング球面との交点を持つときの係数Tを計算する関数
    def calcT_sphere(self, ray_pos, ray_dir, lens_pos, lens_R):
        if lens_R < 0:
            shiftV = lens_pos - np.array([0, 0, lens_R])
            ray_pos = ray_pos - shiftV
            A = np.dot(ray_dir, ray_dir)
            B = np.dot(ray_dir, ray_pos)
            C = np.dot(ray_pos, ray_pos) - abs(lens_R)**2
            T = (-B + np.sqrt(B**2 - A*C)) / A
        elif lens_R > 0:
            shiftV = lens_pos - np.array([0, 0, lens_R])
            ray_pos = ray_pos - shiftV
            A = np.dot(ray_dir, ray_dir)
            B = np.dot(ray_dir, ray_pos)
            C = np.dot(ray_pos, ray_pos) - abs(lens_R)**2
            T = (-B - np.sqrt(B**2 - A*C)) / A
        return T

    # レイトレーシング放物線との交点を持つときの係数Tを計算する関数
    def calcT_parabola(self, ray_pos, ray_dir, parabola_pos, parabola_R):
        ray_pos = ray_pos - parabola_pos
        if ray_dir[0]==0 and ray_dir[2]==0:
            a = 1/(2*parabola_R)
            T = a*(ray_pos[0]**2 - ray_pos[1]/a + ray_pos[0]**2) / ray_dir[1]
            return T
        else:
            if parabola_R<0:
                a = -1/(2*parabola_R)
                A = ray_dir[0]**2 + ray_dir[2]**2
                B = ray_pos[0]*ray_dir[0] + ray_pos[2]*ray_dir[2] - ray_dir[1]/(2*a)
                C = ray_pos[0]**2 + ray_pos[2]**2 - ray_pos[1]/a
                T = (-B - np.sqrt(B**2 - A*C)) / A
                return T
            else:
                a = 1/(2*parabola_R)
                A = ray_dir[0]**2 + ray_dir[2]**2
                B = ray_pos[0]*ray_dir[0] + ray_pos[2]*ray_dir[2] - ray_dir[1]/(2*a)
                C = ray_pos[0]**2 + ray_pos[2]**2 - ray_pos[1]/a
                T = (-B + np.sqrt(B**2 - A*C)) / A
                return T

    # 放物線の法線ベクトルを計算する関数
    def calcNormal_parabola(self, ray_pos, parabola_pos, parabola_R):
        a = abs(1/(2*parabola_R))
        normalVx = 2*a*(ray_pos[0] - parabola_pos[0])
        normalVy = -1
        normalVz = 2*a*(ray_pos[2] - parabola_pos[2])
        normalV = np.array([normalVx, normalVy, normalVz])
        normalV = normalV/np.linalg.norm(normalV)
        return normalV

    # 反射後の方向ベクトルを計算する関数
    def calcReflectionV(self, ray_dir, normalV):
        ray_dir = np.array(ray_dir)/np.linalg.norm(ray_dir)
        normalV = np.array(normalV)/np.linalg.norm(normalV)
        outRayV = ray_dir - 2*(np.dot(ray_dir, normalV))*normalV
        # 正規化
        outRayV = outRayV/np.linalg.norm(outRayV)
        return outRayV

    # スネルの法則から方向ベクトルを求める関数
    def calcRefractionV(self, ray_dir, normalV, Nin, Nout):
        if np.dot(ray_dir, normalV) <= 0:
            #print("内積が負です")
            # 正規化
            ray_dir = ray_dir/np.linalg.norm(ray_dir)
            normalV = normalV/np.linalg.norm(normalV)
            # 係数A
            A = Nin/Nout
            # 入射角
            cos_t_in = abs(np.dot(ray_dir, normalV))
            # 量子化誤差対策
            if cos_t_in < -1.:
                cos_t_in = -1.
            elif cos_t_in > 1.:
                cos_t_in = 1.
            # スネルの法則
            sin_t_in = np.sqrt(1.0 - cos_t_in**2)
            sin_t_out = sin_t_in*A
            if sin_t_out > 1.0:
                # 全反射する場合
                return np.zeros(3)
            cos_t_out = np.sqrt(1 - sin_t_out**2)
            # 係数B
            B = A*cos_t_in - cos_t_out
            # 出射光線の方向ベクトル
            outRayV = A*ray_dir + B*normalV
            # 正規化
            outRayV = outRayV/np.linalg.norm(outRayV)
        else:
            #print("内積が正です")
            # スネルの法則から屈折光の方向ベクトルを求める関数(右に凸の場合)
            # 正規化
            ray_dir = ray_dir/np.linalg.norm(ray_dir)
            normalV = normalV/np.linalg.norm(normalV)
            # 係数A
            A = Nin/Nout
            # 入射角
            cos_t_in = abs(np.dot(ray_dir,normalV))
            #量子化誤差対策
            if cos_t_in<-1.:
                cos_t_in = -1.
            elif cos_t_in>1.:
                cos_t_in = 1.
            # スネルの法則
            sin_t_in = np.sqrt(1.0 - cos_t_in**2)
            sin_t_out = sin_t_in*A
            if sin_t_out>1.0:
                #全反射する場合
                return np.zeros(3)
            cos_t_out = np.sqrt(1 - sin_t_out**2)
            # 係数B
            B = -A*cos_t_in + cos_t_out
            # 出射光線の方向ベクトル
            outRayV = A*ray_dir + B*normalV
            # 正規化
            outRayV = outRayV/np.linalg.norm(outRayV)
        return outRayV

    # ２点の位置ベクトルから直線を引く関数
    def plotLineBlue(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='blue')

    def plotLineGreen(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='green')

    def plotLineRed(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='r')

    def plotLineOrange(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='orange')

    def plotLinePurple(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='purple')

    def plotLineBlack(self, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='black')

# ミラーの光線追跡を行う関数
def mirror_reflection(param_list):
    TTM1_1_nV = np.array([param_list[0], param_list[1], param_list[2]])
    TTM1_2_nV = np.array([param_list[3], param_list[4], param_list[5]])
    TTM1_3_nV = np.array([param_list[6], param_list[7], param_list[8]])
    TTM1_4_nV = np.array([param_list[9], param_list[10], param_list[11]])
    TTM2_1_nV = np.array([param_list[12], param_list[13], param_list[14]])
    TTM2_2_nV = np.array([param_list[15], param_list[16], param_list[17]])
    TTM2_3_nV = np.array([param_list[18], param_list[19], param_list[20]])
    TTM2_4_nV = np.array([param_list[21], param_list[22], param_list[23]])


    # fsolve_params
    fsolve_TTM1_nVlist = [TTM1_1_nV, TTM1_2_nV, TTM1_3_nV, TTM1_4_nV]
    fsolve_TTM2_nVlist = [TTM2_1_nV, TTM2_2_nV, TTM2_3_nV, TTM2_4_nV]


    # インスタンス生成
    VF = VectorFunctions()

    # 初期値
    s_pos = np.array([0.0, 0.0, 0.0])
    s_dir = np.array([1.0, 0.0, 0.0])

    # BS
    # ----------------BS1----------------
    pos_BS1_f = BS_params[0][0]
    nV_BS1_f = BS_params[0][1]
    nV_BS1_f = nV_BS1_f / np.linalg.norm(nV_BS1_f)
    front_pos_BS1 = s_pos + s_dir*VF.calcT_mirror(s_pos, s_dir, pos_BS1_f, nV_BS1_f)
    front_dir_BS1 = VF.calcReflectionV(s_dir, nV_BS1_f)
    VF.plotLineBlue(s_pos, front_pos_BS1)

    glass_dir_BS1 = VF.calcRefractionV(s_dir, nV_BS1_f, N_air, N_Fused_Silica)
    pos_BS1_b = BS_params[1][0]
    nV_BS1_b = BS_params[1][1]
    nV_BS1_b = nV_BS1_b / np.linalg.norm(nV_BS1_b)
    back_pos_BS1 = front_pos_BS1 + glass_dir_BS1*VF.calcT_mirror(front_pos_BS1, glass_dir_BS1, pos_BS1_b, nV_BS1_b)
    back_dir_BS1 = VF.calcRefractionV(glass_dir_BS1, nV_BS1_b, N_Fused_Silica, N_air)
    VF.plotLineGreen(front_pos_BS1, back_pos_BS1)

    # ----------------BS2----------------
    pos_BS2_f = BS_params[2][0]
    nV_BS2_f = BS_params[2][1]
    nV_BS2_f = nV_BS2_f / np.linalg.norm(nV_BS2_f)
    front_pos_BS2 = back_pos_BS1 + back_dir_BS1*VF.calcT_mirror(back_pos_BS1, back_dir_BS1, pos_BS2_f, nV_BS2_f)
    front_dir_BS2 = VF.calcReflectionV(back_dir_BS1, nV_BS2_f)
    VF.plotLineGreen(back_pos_BS1, front_pos_BS2)

    glass_dir_BS2 = VF.calcRefractionV(back_dir_BS1, nV_BS2_f, N_air, N_Fused_Silica)
    pos_BS2_b = BS_params[3][0]
    nV_BS2_b = BS_params[3][1]
    nV_BS2_b = nV_BS2_b / np.linalg.norm(nV_BS2_b)
    back_pos_BS2 = front_pos_BS2 + glass_dir_BS2*VF.calcT_mirror(front_pos_BS2, glass_dir_BS2, pos_BS2_b, nV_BS2_b)
    back_dir_BS2 = VF.calcRefractionV(glass_dir_BS2, nV_BS2_b, N_Fused_Silica, N_air)
    VF.plotLineRed(front_pos_BS2, back_pos_BS2)

    # ----------------BS3----------------
    pos_BS3_f = BS_params[4][0]
    nV_BS3_f = BS_params[4][1]
    nV_BS3_f = nV_BS3_f / np.linalg.norm(nV_BS3_f)
    front_pos_BS3 = back_pos_BS2 + back_dir_BS2*VF.calcT_mirror(back_pos_BS2, back_dir_BS2, pos_BS3_f, nV_BS3_f)
    front_dir_BS3 = VF.calcReflectionV(back_dir_BS2, nV_BS3_f)
    VF.plotLineRed(back_pos_BS2, front_pos_BS3)

    glass_dir_BS3 = VF.calcRefractionV(back_dir_BS2, nV_BS3_f, N_air, N_Fused_Silica)
    pos_BS3_b = BS_params[5][0]
    nV_BS3_b = BS_params[5][1]
    nV_BS3_b = nV_BS3_b / np.linalg.norm(nV_BS3_b)
    back_pos_BS3 = front_pos_BS3 + glass_dir_BS3*VF.calcT_mirror(front_pos_BS3, glass_dir_BS3, pos_BS3_b, nV_BS3_b)
    back_dir_BS3 = VF.calcRefractionV(glass_dir_BS3, nV_BS3_b, N_Fused_Silica, N_air)
    VF.plotLineOrange(front_pos_BS3, back_pos_BS3)

    # ----------------BS4----------------
    pos_BS4_f = BS_params[6][0]
    nV_BS4_f = BS_params[6][1]
    nV_BS4_f = nV_BS4_f / np.linalg.norm(nV_BS4_f)
    front_pos_BS4 = back_pos_BS3 + back_dir_BS3*VF.calcT_mirror(back_pos_BS3, back_dir_BS3, pos_BS4_f, nV_BS4_f)
    front_dir_BS4 = VF.calcReflectionV(back_dir_BS3, nV_BS4_f)
    VF.plotLineOrange(back_pos_BS3, front_pos_BS4)

    last_pos = np.array([front_pos_BS1, front_pos_BS2, front_pos_BS3, front_pos_BS4])
    last_dir = np.array([front_dir_BS1, front_dir_BS2, front_dir_BS3, front_dir_BS4])


    # TTM1
    pos_TTM1 = []
    dir_TTM1 = []
    for i in range(4):
        nV_TTM1 = fsolve_TTM1_nVlist[i]
        pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], TTM1_params[i][0], nV_TTM1)
        dir = VF.calcReflectionV(last_dir[i], nV_TTM1)
        pos_TTM1.append(pos)
        dir_TTM1.append(dir)

    VF.plotLineBlue(last_pos[0], pos_TTM1[0])
    VF.plotLineGreen(last_pos[1], pos_TTM1[1])
    VF.plotLineRed(last_pos[2], pos_TTM1[2])
    VF.plotLineOrange(last_pos[3], pos_TTM1[3])

    last_pos = np.array(pos_TTM1)
    last_dir = np.array(dir_TTM1)


    # TTM2
    pos_TTM2 = []
    dir_TTM2 = []
    for i in range(4):
        nV_TTM2 = fsolve_TTM2_nVlist[i]
        pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], TTM2_params[i][0], nV_TTM2)
        dir = VF.calcReflectionV(last_dir[i], nV_TTM2)
        pos_TTM2.append(pos)
        dir_TTM2.append(dir)

    L1 = np.linalg.norm(pos_TTM2[0] - TTM2_params[0][0])
    L2 = np.linalg.norm(pos_TTM2[1] - TTM2_params[1][0])
    L3 = np.linalg.norm(pos_TTM2[2] - TTM2_params[2][0])
    L4 = np.linalg.norm(pos_TTM2[3] - TTM2_params[3][0])
    #print(L1, L2, L3, L4, "at TTM2 limits")

    if L1>12.7 or L2>12.7 or L3>12.7 or L4>12.7:
        return [1000000000000000000]*24
    else:
        VF.plotLineBlue(last_pos[0], pos_TTM2[0])
        VF.plotLineGreen(last_pos[1], pos_TTM2[1])
        VF.plotLineRed(last_pos[2], pos_TTM2[2])
        VF.plotLineOrange(last_pos[3], pos_TTM2[3])

        last_pos = np.array(pos_TTM2)
        last_dir = np.array(dir_TTM2)


        # PRISM
        pos_PRISM_front = []
        dir_PRISM_front = []
        for i in range(4):
            nV_PRISM_front = PRISM_front_params[i][1]
            #print(nV_PRISM)
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], PRISM_front_params[i][0], PRISM_front_params[i][1])
            dir = VF.calcRefractionV(last_dir[i], nV_PRISM_front, N_air, N_Fused_Silica)
            pos_PRISM_front.append(pos)
            dir_PRISM_front.append(dir)

        VF.plotLineBlue(last_pos[0], pos_PRISM_front[0])
        VF.plotLineGreen(last_pos[1], pos_PRISM_front[1])
        VF.plotLineRed(last_pos[2], pos_PRISM_front[2])
        VF.plotLineOrange(last_pos[3], pos_PRISM_front[3])

        last_pos = np.array(pos_PRISM_front)
        last_dir = np.array(dir_PRISM_front)


        # PRISM_back
        pos_PRISM_back = []
        dir_PRISM_back = []
        for i in range(4):
            nV_PRISM_back = PRISM_back_params[i][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], PRISM_back_params[i][0], PRISM_back_params[i][1])
            dir = VF.calcRefractionV(last_dir[i], nV_PRISM_back, N_Fused_Silica, N_air)
            pos_PRISM_back.append(pos)
            dir_PRISM_back.append(dir)

        VF.plotLineBlue(last_pos[0], pos_PRISM_back[0])
        VF.plotLineGreen(last_pos[1], pos_PRISM_back[1])
        VF.plotLineRed(last_pos[2], pos_PRISM_back[2])
        VF.plotLineOrange(last_pos[3], pos_PRISM_back[3])

        last_pos = np.array(pos_PRISM_back)
        last_dir = np.array(dir_PRISM_back)


        # FBM
        pos_FBM = []
        dir_FBM = []
        for i in range(4):
            nV_FBM = FBM_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], FBM_params[0][0], FBM_params[0][1])
            dir = VF.calcReflectionV(last_dir[i], nV_FBM)
            pos_FBM.append(pos)
            dir_FBM.append(dir)

        VF.plotLineBlue(last_pos[0], pos_FBM[0])
        VF.plotLineGreen(last_pos[1], pos_FBM[1])
        VF.plotLineRed(last_pos[2], pos_FBM[2])
        VF.plotLineOrange(last_pos[3], pos_FBM[3])

        last_pos = np.array(pos_FBM)
        last_dir = np.array(dir_FBM)

        # => out
        after_FBM_ray_pos = last_pos
        after_FBM_ray_dir = last_dir


        # skip BE entrance pupil
        # M5
        pos_M5 = []
        dir_M5 = []
        for i in range(4):
            nV_M5 = M5_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], M5_params[0][0], M5_params[0][1])
            dir = VF.calcReflectionV(last_dir[i], nV_M5)
            pos_M5.append(pos)
            dir_M5.append(dir)

        VF.plotLineBlue(last_pos[0], pos_M5[0])
        VF.plotLineGreen(last_pos[1], pos_M5[1])
        VF.plotLineRed(last_pos[2], pos_M5[2])
        VF.plotLineOrange(last_pos[3], pos_M5[3])

        last_pos = np.array(pos_M5)
        last_dir = np.array(dir_M5)


        # DM1234
        # DM1
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM1234_params[0][0], nV_DM1234)
            dir = VF.calcReflectionV(last_dir[i], nV_DM1234)
            pos_DM1234.append(pos)
            dir_DM1234.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM1234[0])
        VF.plotLineGreen(last_pos[1], pos_DM1234[1])
        VF.plotLineRed(last_pos[2], pos_DM1234[2])
        VF.plotLineOrange(last_pos[3], pos_DM1234[3])

        last_pos = np.array(pos_DM1234)
        last_dir = np.array(dir_DM1234)


        # DM2
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM1234_params[1][0], nV_DM1234)
            dir = VF.calcReflectionV(last_dir[i], nV_DM1234)
            pos_DM1234.append(pos)
            dir_DM1234.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM1234[0])
        VF.plotLineGreen(last_pos[1], pos_DM1234[1])
        VF.plotLineRed(last_pos[2], pos_DM1234[2])
        VF.plotLineOrange(last_pos[3], pos_DM1234[3])

        last_pos = np.array(pos_DM1234)
        last_dir = np.array(dir_DM1234)


        # DM3
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[2][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM1234_params[2][0], nV_DM1234)
            dir = VF.calcReflectionV(last_dir[i], nV_DM1234)
            pos_DM1234.append(pos)
            dir_DM1234.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM1234[0])
        VF.plotLineGreen(last_pos[1], pos_DM1234[1])
        VF.plotLineRed(last_pos[2], pos_DM1234[2])
        VF.plotLineOrange(last_pos[3], pos_DM1234[3])

        last_pos = np.array(pos_DM1234)
        last_dir = np.array(dir_DM1234)


        # DM4
        last_pos = pos_DM1234
        last_dir = dir_DM1234
        pos_DM1234 = []
        dir_DM1234 = []
        for i in range(4):
            nV_DM1234 = DM1234_params[3][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM1234_params[3][0], nV_DM1234)
            dir = VF.calcReflectionV(last_dir[i], nV_DM1234)
            pos_DM1234.append(pos)
            dir_DM1234.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM1234[0])
        VF.plotLineGreen(last_pos[1], pos_DM1234[1])
        VF.plotLineRed(last_pos[2], pos_DM1234[2])
        VF.plotLineOrange(last_pos[3], pos_DM1234[3])

        last_pos = np.array(pos_DM1234)
        last_dir = np.array(dir_DM1234)


        # IMR
        # IMR-Front
        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = np.array([-np.sin(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), np.cos(IMR_rot_deg*np.pi/180)*(1/np.sqrt(2)), -1/np.sqrt(2)])
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], IMR_params[0][0], nV_IMR)
            dir = VF.calcRefractionV(last_dir[i], nV_IMR, N_air, N_Fused_Silica)
            pos_IMR.append(pos)
            dir_IMR.append(dir)

        VF.plotLineBlue(last_pos[0], pos_IMR[0])
        VF.plotLineGreen(last_pos[1], pos_IMR[1])
        VF.plotLineRed(last_pos[2], pos_IMR[2])
        VF.plotLineOrange(last_pos[3], pos_IMR[3])

        last_pos = np.array(pos_IMR)
        last_dir = np.array(dir_IMR)


        # IMR-Mid
        last_pos = pos_IMR
        last_dir = dir_IMR
        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = IMR_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], IMR_params[1][0], nV_IMR)
            dir = VF.calcReflectionV(last_dir[i], nV_IMR)
            pos_IMR.append(pos)
            dir_IMR.append(dir)

        VF.plotLineBlue(last_pos[0], pos_IMR[0])
        VF.plotLineGreen(last_pos[1], pos_IMR[1])
        VF.plotLineRed(last_pos[2], pos_IMR[2])
        VF.plotLineOrange(last_pos[3], pos_IMR[3])

        last_pos = np.array(pos_IMR)
        last_dir = np.array(dir_IMR)


        # IMR-Back
        last_pos = pos_IMR
        last_dir = dir_IMR
        pos_IMR = []
        dir_IMR = []
        for i in range(4):
            nV_IMR = IMR_params[2][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], IMR_params[2][0], nV_IMR)
            dir = VF.calcRefractionV(last_dir[i], nV_IMR, N_Fused_Silica, N_air)
            pos_IMR.append(pos)
            dir_IMR.append(dir)

        VF.plotLineBlue(last_pos[0], pos_IMR[0])
        VF.plotLineGreen(last_pos[1], pos_IMR[1])
        VF.plotLineRed(last_pos[2], pos_IMR[2])
        VF.plotLineOrange(last_pos[3], pos_IMR[3])

        last_pos = np.array(pos_IMR)
        last_dir = np.array(dir_IMR)


        # DM56
        # DM5
        pos_DM56 = []
        dir_DM56 = []
        for i in range(4):
            nV_DM56 = DM56_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM56_params[0][0], nV_DM56)
            dir = VF.calcReflectionV(last_dir[i], nV_DM56)
            pos_DM56.append(pos)
            dir_DM56.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM56[0])
        VF.plotLineGreen(last_pos[1], pos_DM56[1])
        VF.plotLineRed(last_pos[2], pos_DM56[2])
        VF.plotLineOrange(last_pos[3], pos_DM56[3])

        last_pos = np.array(pos_DM56)
        last_dir = np.array(dir_DM56)


        # DM6
        last_pos = pos_DM56
        last_dir = dir_DM56
        pos_DM56 = []
        dir_DM56 = []
        for i in range(4):
            nV_DM56 = DM56_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], DM56_params[1][0], nV_DM56)
            dir = VF.calcReflectionV(last_dir[i], nV_DM56)
            pos_DM56.append(pos)
            dir_DM56.append(dir)

        VF.plotLineBlue(last_pos[0], pos_DM56[0])
        VF.plotLineGreen(last_pos[1], pos_DM56[1])
        VF.plotLineRed(last_pos[2], pos_DM56[2])
        VF.plotLineOrange(last_pos[3], pos_DM56[3])

        last_pos = np.array(pos_DM56)
        last_dir = np.array(dir_DM56)


        # BE_G123
        # BE_G1
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[0][0], BE_G123_params[0][3])
            if (pos[0]-430)**2+(pos[1]-90.6)**2>8**2:
                nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[0][3]]) - BE_G123_params[0][0]
                dir = np.array([0, 0, 0])
                #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
                #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
                pos_BE_G123.append(pos)
                dir_BE_G123.append(dir)
            else:
                nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[0][3]]) - BE_G123_params[0][0]
                dir = VF.calcRefractionV(last_dir[i], -nV_BE_G123, N_air, N_Fused_Silica)
                #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
                #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
                pos_BE_G123.append(pos)
                dir_BE_G123.append(dir)

        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[1][0], BE_G123_params[1][3])
            nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[1][3]]) - BE_G123_params[1][0]
            dir = VF.calcRefractionV(last_dir[i], nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(pos)
            dir_BE_G123.append(dir)

        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        # BE_G2
        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[2][0], BE_G123_params[2][3])
            if (pos[0]-430)**2+(pos[1]-90.6)**2>17**2:
                nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[2][3]]) - BE_G123_params[2][0]
                dir = np.array([0, 0, 0])
                #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
                #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
                pos_BE_G123.append(pos)
                dir_BE_G123.append(dir)
            else:
                nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[2][3]]) - BE_G123_params[2][0]
                dir = VF.calcRefractionV(last_dir[i], -nV_BE_G123, N_air, N_Fused_Silica)
                #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
                #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
                pos_BE_G123.append(pos)
                dir_BE_G123.append(dir)

        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[3][0], BE_G123_params[3][3])
            nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[3][3]]) - BE_G123_params[3][0]
            dir = VF.calcRefractionV(last_dir[i], -nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(pos)
            dir_BE_G123.append(dir)

        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        # BE_G3
        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[4][0], BE_G123_params[4][3])
            nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[4][3]]) - BE_G123_params[4][0]
            dir = VF.calcRefractionV(last_dir[i], -nV_BE_G123, N_air, N_Fused_Silica)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(pos)
            dir_BE_G123.append(dir)
        
        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        last_pos = pos_BE_G123
        last_dir = dir_BE_G123
        pos_BE_G123 = []
        dir_BE_G123 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_sphere(last_pos[i], last_dir[i], BE_G123_params[5][0], BE_G123_params[5][3])
            nV_BE_G123 = pos + np.array([0, 0, BE_G123_params[5][3]]) - BE_G123_params[5][0]
            dir = VF.calcRefractionV(last_dir[i], -nV_BE_G123, N_Fused_Silica, N_air)
            #ax.quiver(pos[0], pos[1], pos[2], nV_BE_G123[0], nV_BE_G123[1], nV_BE_G123[2], color='blue', length=0.1)
            #ax.quiver(pos[0], pos[1], pos[2], -nV_BE_G123[0], -nV_BE_G123[1], -nV_BE_G123[2], color='blue', length=0.1)
            pos_BE_G123.append(pos)
            dir_BE_G123.append(dir)

        VF.plotLineBlue(last_pos[0], pos_BE_G123[0])
        VF.plotLineGreen(last_pos[1], pos_BE_G123[1])
        VF.plotLineRed(last_pos[2], pos_BE_G123[2])
        VF.plotLineOrange(last_pos[3], pos_BE_G123[3])

        last_pos = np.array(pos_BE_G123)
        last_dir = np.array(dir_BE_G123)


        # M678
        # M6
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], M678_params[0][0], nV_M678)
            dir = VF.calcReflectionV(last_dir[i], nV_M678)
            pos_M678.append(pos)
            dir_M678.append(dir)

        VF.plotLineBlue(last_pos[0], pos_M678[0])
        VF.plotLineGreen(last_pos[1], pos_M678[1])
        VF.plotLineRed(last_pos[2], pos_M678[2])
        VF.plotLineOrange(last_pos[3], pos_M678[3])

        last_pos = np.array(pos_M678)
        last_dir = np.array(dir_M678)


        # M7
        last_pos = pos_M678
        last_dir = dir_M678
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], M678_params[1][0], nV_M678)
            dir = VF.calcReflectionV(last_dir[i], nV_M678)
            pos_M678.append(pos)
            dir_M678.append(dir)
        
        VF.plotLineBlue(last_pos[0], pos_M678[0])
        VF.plotLineGreen(last_pos[1], pos_M678[1])
        VF.plotLineRed(last_pos[2], pos_M678[2])
        VF.plotLineOrange(last_pos[3], pos_M678[3])

        last_pos = np.array(pos_M678)
        last_dir = np.array(dir_M678)


        # M8
        last_pos = pos_M678
        last_dir = dir_M678
        pos_M678 = []
        dir_M678 = []
        for i in range(4):
            nV_M678 = M678_params[2][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], M678_params[2][0], nV_M678)
            dir = VF.calcReflectionV(last_dir[i], nV_M678)
            pos_M678.append(pos)
            dir_M678.append(dir)

        VF.plotLineBlue(last_pos[0], pos_M678[0])
        VF.plotLineGreen(last_pos[1], pos_M678[1])
        VF.plotLineRed(last_pos[2], pos_M678[2])
        VF.plotLineOrange(last_pos[3], pos_M678[3])

        last_pos = np.array(pos_M678)
        last_dir = np.array(dir_M678)


        # LLT_Window
        pos_LLT_Window = []
        dir_LLT_Window = []
        for i in range(4):
            nV_LLT_Window = LLT_Window_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_Window_params[0][0], nV_LLT_Window)
            dir = VF.calcRefractionV(last_dir[i], nV_LLT_Window, N_air, N_BSL7)
            pos_LLT_Window.append(pos)
            dir_LLT_Window.append(dir)

        VF.plotLineBlue(last_pos[0], pos_LLT_Window[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_Window[1])
        VF.plotLineRed(last_pos[2], pos_LLT_Window[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_Window[3])

        last_pos = np.array(pos_LLT_Window)
        last_dir = np.array(dir_LLT_Window)

        # LLT_Window_back
        pos_LLT_Window = []
        dir_LLT_Window = []
        for i in range(4):
            nV_LLT_Window = LLT_Window_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_Window_params[1][0], nV_LLT_Window)
            dir = VF.calcRefractionV(last_dir[i], nV_LLT_Window, N_BSL7, N_air)
            pos_LLT_Window.append(pos)
            dir_LLT_Window.append(dir)
        
        VF.plotLineBlue(last_pos[0], pos_LLT_Window[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_Window[1])
        VF.plotLineRed(last_pos[2], pos_LLT_Window[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_Window[3])

        last_pos1 = np.array(pos_LLT_Window)
        last_dir1 = np.array(dir_LLT_Window)


        # LLT_M321
        # LLT_M3
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        for i in range(4):
            nV_LLT_M321 = LLT_M321_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_M321_params[0][0], nV_LLT_M321)
            dir = VF.calcReflectionV(last_dir[i], nV_LLT_M321)
            pos_LLT_M321.append(pos)
            dir_LLT_M321.append(dir)

        VF.plotLineBlue(last_pos[0], pos_LLT_M321[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_M321[1])
        VF.plotLineRed(last_pos[2], pos_LLT_M321[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_M321[3])

        last_pos = np.array(pos_LLT_M321)
        last_dir = np.array(dir_LLT_M321)


        # LLT_M2
        last_pos = pos_LLT_M321
        last_dir = dir_LLT_M321
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_parabola(last_pos[i], last_dir[i], LLT_M321_params[1][0], LLT_M321_params[1][3])
            if (pos[0]-60)**2+(pos[2]+370)**2>20**2:
                nV_LLT_M321 = VF.calcNormal_parabola(pos, LLT_M321_params[1][0], LLT_M321_params[1][3])
                #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
                dir = np.array([0,0,0])
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)
            else:
                nV_LLT_M321 = VF.calcNormal_parabola(pos, LLT_M321_params[1][0], LLT_M321_params[1][3])
                #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
                dir = VF.calcReflectionV(last_dir[i], nV_LLT_M321)
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)

        VF.plotLineBlue(last_pos[0], pos_LLT_M321[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_M321[1])
        VF.plotLineRed(last_pos[2], pos_LLT_M321[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_M321[3])

        last_pos = np.array(pos_LLT_M321)
        last_dir = np.array(dir_LLT_M321)

        # block
        last_pos = pos_LLT_M321
        last_dir = dir_LLT_M321
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_M321_params[0][0], np.array([0,1,0]))
            if (pos[0]-60)**2+(pos[2]+370)**2<21.4**2:
                dir = np.array([0,0,0])
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)
            else:
                dir = last_dir[i]
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)

        # LLT_M1
        last_pos = pos_LLT_M321
        last_dir = dir_LLT_M321
        pos_LLT_M321 = []
        dir_LLT_M321 = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_parabola(last_pos[i], last_dir[i], LLT_M321_params[2][0], LLT_M321_params[2][3])
            if (pos[0]-60)**2+(pos[2]+370)**2>252**2:
                nV_LLT_M321 = VF.calcNormal_parabola(pos, LLT_M321_params[2][0], LLT_M321_params[2][3])
                #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
                dir = np.array([0,0,0])
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)
            else:
                nV_LLT_M321 = VF.calcNormal_parabola(pos, LLT_M321_params[2][0], LLT_M321_params[2][3])
                #ax.quiver(pos[0], pos[1], pos[2], nV_LLT_M321[0], nV_LLT_M321[1], nV_LLT_M321[2], color='r', length=10)
                dir = VF.calcReflectionV(last_dir[i], nV_LLT_M321)
                pos_LLT_M321.append(pos)
                dir_LLT_M321.append(dir)
        
        VF.plotLineBlue(last_pos[0], pos_LLT_M321[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_M321[1])
        VF.plotLineRed(last_pos[2], pos_LLT_M321[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_M321[3])

        last_pos = np.array(pos_LLT_M321)
        last_dir = np.array(dir_LLT_M321)


        # LLT_Exit_Window
        pos_LLT_Exit_Window = []
        dir_LLT_Exit_Window = []
        for i in range(4):
            nV_LLT_Exit_Window = LLT_Exit_Window_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_Exit_Window_params[0][0], nV_LLT_Exit_Window)
            dir = VF.calcRefractionV(last_dir[i], nV_LLT_Exit_Window, N_air, N_BSL7)
            pos_LLT_Exit_Window.append(pos)
            dir_LLT_Exit_Window.append(dir)

        VF.plotLineBlue(last_pos[0], pos_LLT_Exit_Window[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_Exit_Window[1])
        VF.plotLineRed(last_pos[2], pos_LLT_Exit_Window[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_Exit_Window[3])

        last_pos = np.array(pos_LLT_Exit_Window)
        last_dir = np.array(dir_LLT_Exit_Window)


        # LLT_Exit_Window_back
        last_pos = pos_LLT_Exit_Window
        last_dir = dir_LLT_Exit_Window
        pos_LLT_Exit_Window = []
        dir_LLT_Exit_Window = []
        for i in range(4):
            nV_LLT_Exit_Window = LLT_Exit_Window_params[1][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], LLT_Exit_Window_params[1][0], nV_LLT_Exit_Window)
            dir = VF.calcRefractionV(last_dir[i], nV_LLT_Exit_Window, N_BSL7, N_air)
            pos_LLT_Exit_Window.append(pos)
            dir_LLT_Exit_Window.append(dir)

        VF.plotLineBlue(last_pos[0], pos_LLT_Exit_Window[0])
        VF.plotLineGreen(last_pos[1], pos_LLT_Exit_Window[1])
        VF.plotLineRed(last_pos[2], pos_LLT_Exit_Window[2])
        VF.plotLineOrange(last_pos[3], pos_LLT_Exit_Window[3])

        last_pos = np.array(pos_LLT_Exit_Window)
        last_dir = np.array(dir_LLT_Exit_Window)


        # Evaluation_Plane
        pos_Evaluation_Plane = []
        dir_Evaluation_Plane = []
        for i in range(4):
            nV_Evaluation_Plane = Evaluation_Plane_params[0][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], Evaluation_Plane_params[0][0], nV_Evaluation_Plane)
            dir = last_dir[i]
            pos_Evaluation_Plane.append(pos)
            dir_Evaluation_Plane.append(dir)

        VF.plotLineBlue(last_pos[0], pos_Evaluation_Plane[0])
        VF.plotLineGreen(last_pos[1], pos_Evaluation_Plane[1])
        VF.plotLineRed(last_pos[2], pos_Evaluation_Plane[2])
        VF.plotLineOrange(last_pos[3], pos_Evaluation_Plane[3])

        rad_list = []
        for i in range(4):
            rad_list.append(np.arccos(-np.dot(last_dir[i], Evaluation_Plane_params[0][1])))
        deg_list = np.degrees(rad_list)
        print(deg_list, "deg_list")

        print("(Evaluation_Plane)")
        print("IMR_rot_deg:", IMR_rot_deg, "deg")
        print("ray1:", pos_Evaluation_Plane[0], dir_Evaluation_Plane[0]/np.linalg.norm(dir_Evaluation_Plane[0]), np.round(deg_list[0], 6)*3600, "arcsec")
        print("ray2:", pos_Evaluation_Plane[1], dir_Evaluation_Plane[1]/np.linalg.norm(dir_Evaluation_Plane[1]), np.round(deg_list[1], 6)*3600, "arcsec")
        print("ray3:", pos_Evaluation_Plane[2], dir_Evaluation_Plane[2]/np.linalg.norm(dir_Evaluation_Plane[2]), np.round(deg_list[2], 6)*3600, "arcsec")
        print("ray4:", pos_Evaluation_Plane[3], dir_Evaluation_Plane[3]/np.linalg.norm(dir_Evaluation_Plane[3]), np.round(deg_list[3], 6)*3600, "arcsec")
        
        print("ray1-ray4:", np.round(np.degrees(np.arccos(np.dot(last_dir[0], last_dir[3]))), 6)*3600, "arcsec")
        print("ray2-ray3:", np.round(np.degrees(np.arccos(np.dot(last_dir[1], last_dir[2]))), 6)*3600, "arcsec\n")
        """print("cf.  5 arcsec => 0.001389 deg, 10 arcsec => 0.002778 deg")
        print("cf. 15 arcsec => 0.004167 deg, 20 arcsec => 0.005556 deg")
        print("cf. 25 arcsec => 0.006944 deg, 30 arcsec => 0.008333 deg")
        print("cf. 35 arcsec => 0.009722 deg, 40 arcsec => 0.011111 deg\n")"""

        for i in dir_Evaluation_Plane:
            if i[1]>0.9999999:
                print("OK")
            else:
                print("NG")


        last_pos = np.array(pos_Evaluation_Plane)
        last_dir = np.array(dir_Evaluation_Plane)

        end_pos = []
        end_dir = []
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*2000
            dir = last_dir[i]
            end_pos.append(pos)
            end_dir.append(dir)

        VF.plotLineBlue(last_pos[0], end_pos[0])
        VF.plotLineGreen(last_pos[1], end_pos[1])
        VF.plotLineRed(last_pos[2], end_pos[2])
        VF.plotLineOrange(last_pos[3], end_pos[3])
        VF.plotLineBlack([60.000, -250.400, -370.000], [60.000, 2000, -370.000])

        print(Evaluation_Plane_params[0][0], "evaluation Plane")
        L1 = np.linalg.norm(last_pos[0]-np.array([Evaluation_Plane_params[0][0]]))
        L2 = np.linalg.norm(last_pos[1]-np.array([Evaluation_Plane_params[0][0]]))
        L3 = np.linalg.norm(last_pos[2]-np.array([Evaluation_Plane_params[0][0]]))
        L4 = np.linalg.norm(last_pos[3]-np.array([Evaluation_Plane_params[0][0]]))
        print(L1, "r1")
        print(L2, "r2")
        print(L3, "r3")
        print(L4, "r4")

        if L1>170 or L2>170 or L3>170 or L4>170:
            return [10**10]*24
        else:
            # 描画範囲
            LX = 500
            LY = 500
            LZ = 500
            ax.set_xlim(-LX+60, LX+60)
            ax.set_ylim(-LY+90.6, LY+90.6)
            ax.set_zlim(-LZ-350, LZ-350)
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_zlabel('z [mm]')
            ax.view_init(elev=0, azim=90)

            print(last_dir, "last_dir")
            last_dir[0] = last_dir[0]/np.linalg.norm(last_dir[0])
            last_dir[1] = last_dir[1]/np.linalg.norm(last_dir[1])
            last_dir[2] = last_dir[2]/np.linalg.norm(last_dir[2])
            last_dir[3] = last_dir[3]/np.linalg.norm(last_dir[3])
            print(last_dir, "last_dir_re")

            target_index = 3  # 0:10arcsec, 1:20arcsec, 2:30arcsec, 3:40arcsec
            pupill_pos = 2500  # 4905.5, 3480.5, 1380.5

            ref = target_params_step2()
            target_ray_dir = ref[target_index]
            print(target_ray_dir, "target_ray_dir")

            step1_target_ray_dir = target_params()
            target_ray1_dir = step1_target_ray_dir[target_index][0][1]
            target_ray2_dir = step1_target_ray_dir[target_index][1][1]
            target_ray3_dir = step1_target_ray_dir[target_index][2][1]
            target_ray4_dir = step1_target_ray_dir[target_index][3][1]

            T_y = (90.6-after_FBM_ray_pos[0][1])/after_FBM_ray_dir[0][1]
            T_z = (0-after_FBM_ray_pos[0][2])/after_FBM_ray_dir[0][2]
            step1_pupill_ray1y = after_FBM_ray_pos[0][0] + after_FBM_ray_dir[0][0]*T_y
            step1_pupill_ray1z = after_FBM_ray_pos[0][0] + after_FBM_ray_dir[0][0]*T_z
            T_y = (90.6-after_FBM_ray_pos[1][1])/after_FBM_ray_dir[1][1]
            T_z = (0-after_FBM_ray_pos[1][2])/after_FBM_ray_dir[1][2]
            step1_pupill_ray2y = after_FBM_ray_pos[1][0] + after_FBM_ray_dir[1][0]*T_y
            step1_pupill_ray2z = after_FBM_ray_pos[1][0] + after_FBM_ray_dir[1][0]*T_z
            T_y = (90.6-after_FBM_ray_pos[2][1])/after_FBM_ray_dir[2][1]
            T_z = (0-after_FBM_ray_pos[2][2])/after_FBM_ray_dir[2][2]
            step1_pupill_ray3y = after_FBM_ray_pos[2][0] + after_FBM_ray_dir[2][0]*T_y
            step1_pupill_ray3z = after_FBM_ray_pos[2][0] + after_FBM_ray_dir[2][0]*T_z
            T_y = (90.6-after_FBM_ray_pos[3][1])/after_FBM_ray_dir[3][1]
            T_z = (0-after_FBM_ray_pos[3][2])/after_FBM_ray_dir[3][2]
            step1_pupill_ray4y = after_FBM_ray_pos[3][0] + after_FBM_ray_dir[3][0]*T_y
            step1_pupill_ray4z = after_FBM_ray_pos[3][0] + after_FBM_ray_dir[3][0]*T_z

            out = np.array([(pupill_pos-step1_pupill_ray1y)**2,
                            (pupill_pos-step1_pupill_ray1z)**2,
                            (pupill_pos-step1_pupill_ray2y)**2,
                            (pupill_pos-step1_pupill_ray2z)**2,
                            (pupill_pos-step1_pupill_ray3y)**2,
                            (pupill_pos-step1_pupill_ray3z)**2,
                            (pupill_pos-step1_pupill_ray4y)**2,
                            (pupill_pos-step1_pupill_ray4z)**2,
                            (120-((last_pos[0][0]-60)**2+(last_pos[0][2]+370)**2))*10**0,
                            (120-((last_pos[1][0]-60)**2+(last_pos[1][2]+370)**2))*10**0,
                            (120-((last_pos[2][0]-60)**2+(last_pos[2][2]+370)**2))*10**0,
                            (120-((last_pos[3][0]-60)**2+(last_pos[3][2]+370)**2))*10**0,
                            (target_ray1_dir[1]-after_FBM_ray_dir[0][1])**3,
                            (target_ray1_dir[2]-after_FBM_ray_dir[0][2])**3,
                            (target_ray2_dir[1]-after_FBM_ray_dir[1][1])**3,
                            (target_ray2_dir[2]-after_FBM_ray_dir[1][2])**3,
                            (target_ray3_dir[1]-after_FBM_ray_dir[2][1])**3,
                            (target_ray3_dir[2]-after_FBM_ray_dir[2][2])**3,
                            (target_ray4_dir[1]-after_FBM_ray_dir[3][1])**3,
                            (target_ray4_dir[2]-after_FBM_ray_dir[3][2])**3,
                            ((last_pos[0][0]-60)+(last_pos[0][2]+370))**4,
                            ((last_pos[1][0]-60)-(last_pos[1][2]+370))**4,
                            ((last_pos[2][0]-60)-(last_pos[2][2]+370))**4,
                            ((last_pos[3][0]-60)+(last_pos[3][2]+370))**4])

            #out = np.linalg.norm(out)
            check_dir1_1 = np.linalg.norm(TTM1_1_nV-TTM1_params[0][1])*10**15
            check_dir1_2 = np.linalg.norm(TTM1_2_nV-TTM1_params[1][1])*10**15
            check_dir1_3 = np.linalg.norm(TTM1_3_nV-TTM1_params[2][1])*10**15
            check_dir1_4 = np.linalg.norm(TTM1_4_nV-TTM1_params[3][1])*10**15
            check_dir2_1 = np.linalg.norm(TTM2_1_nV-TTM2_params[0][1])*10**15
            check_dir2_2 = np.linalg.norm(TTM2_2_nV-TTM2_params[1][1])*10**15
            check_dir2_3 = np.linalg.norm(TTM2_3_nV-TTM2_params[2][1])*10**15
            check_dir2_4 = np.linalg.norm(TTM2_4_nV-TTM2_params[3][1])*10**15
            if check_dir1_1==0 and check_dir1_2==0 and check_dir1_3==0 and check_dir1_4==0 and check_dir2_1==0 and check_dir2_2==0 and check_dir2_3==0 and check_dir2_4==0:
                out = [10000000]*24
                return out
            else:
                print(out, "out")
                return out

if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()
    fig = plt.figure(figsize=(8, 8))

    """init_params = np.array([TTM1_params[0][1][0], TTM1_params[0][1][1], TTM1_params[0][1][2],
                        TTM1_params[1][1][0], TTM1_params[1][1][1], TTM1_params[1][1][2],
                        TTM1_params[2][1][0], TTM1_params[2][1][1], TTM1_params[2][1][2],
                        TTM1_params[3][1][0], TTM1_params[3][1][1], TTM1_params[3][1][2],
                        TTM2_params[0][1][0], TTM2_params[0][1][1], TTM2_params[0][1][2],
                        TTM2_params[1][1][0], TTM2_params[1][1][1], TTM2_params[1][1][2],
                        TTM2_params[2][1][0], TTM2_params[2][1][1], TTM2_params[2][1][2],
                        TTM2_params[3][1][0], TTM2_params[3][1][1], TTM2_params[3][1][2]])  # init_step1"""

    """init_params = np.array([-0.7070022961, -0.7045547709, -0.0612399220,
                        TTM1_params[1][1][0], TTM1_params[1][1][1], TTM1_params[1][1][2],
                        TTM1_params[2][1][0], TTM1_params[2][1][1], TTM1_params[2][1][2],
                        TTM1_params[3][1][0], TTM1_params[3][1][1], TTM1_params[3][1][2],
                        0.6715791688, -0.7380057630, -0.0657944812,
                        TTM2_params[1][1][0], TTM2_params[1][1][1], TTM2_params[1][1][2],
                        TTM2_params[2][1][0], TTM2_params[2][1][1], TTM2_params[2][1][2],
                        TTM2_params[3][1][0], TTM2_params[3][1][1], TTM2_params[3][1][2]])  # init_step1"""

    init_params = np.array([-0.7070, -0.7046, -0.0612,
                    TTM1_params[1][1][0], TTM1_params[1][1][1], TTM1_params[1][1][2],
                    TTM1_params[2][1][0], TTM1_params[2][1][1], TTM1_params[2][1][2],
                    TTM1_params[3][1][0], TTM1_params[3][1][1], TTM1_params[3][1][2],
                    0.6716, -0.7380, -0.0658,
                    TTM2_params[1][1][0], TTM2_params[1][1][1], TTM2_params[1][1][2],
                    TTM2_params[2][1][0], TTM2_params[2][1][1], TTM2_params[2][1][2],
                    TTM2_params[3][1][0], TTM2_params[3][1][1], TTM2_params[3][1][2]])  # init_step1

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    solver_nV = fsolve(mirror_reflection, init_params)
    #solver_nV = fmin(mirror_reflection, *init_params)

    print(init_params, "init_params")
    print(TTM1_params, "optics_paramsTTM1")
    print(TTM2_params, "optics_paramsTTM2")

    TTM1_1 = np.array([solver_nV[0], solver_nV[1], solver_nV[2]])
    TTM1_1 = TTM1_1/np.linalg.norm(TTM1_1)
    TTM1_2 = np.array([solver_nV[3], solver_nV[4], solver_nV[5]])
    TTM1_2 = TTM1_2/np.linalg.norm(TTM1_2)
    TTM1_3 = np.array([solver_nV[6], solver_nV[7], solver_nV[8]])
    TTM1_3 = TTM1_3/np.linalg.norm(TTM1_3)
    TTM1_4 = np.array([solver_nV[9], solver_nV[10], solver_nV[11]])
    TTM1_4 = TTM1_4/np.linalg.norm(TTM1_4)
    TTM2_1 = np.array([solver_nV[12], solver_nV[13], solver_nV[14]])
    TTM2_1 = TTM2_1/np.linalg.norm(TTM2_1)
    TTM2_2 = np.array([solver_nV[15], solver_nV[16], solver_nV[17]])
    TTM2_2 = TTM2_2/np.linalg.norm(TTM2_2)
    TTM2_3 = np.array([solver_nV[18], solver_nV[19], solver_nV[20]])
    TTM2_3 = TTM2_3/np.linalg.norm(TTM2_3)
    TTM2_4 = np.array([solver_nV[21], solver_nV[22], solver_nV[23]])
    TTM2_4 = TTM2_4/np.linalg.norm(TTM2_4)
    print(TTM1_1, "TTM1_1_nV_solved")
    print(TTM1_2, "TTM1_2_nV_solved")
    print(TTM1_3, "TTM1_3_nV_solved")
    print(TTM1_4, "TTM1_4_nV_solved")
    print(TTM2_1, "TTM2_1_nV_solved")
    print(TTM2_2, "TTM2_2_nV_solved")
    print(TTM2_3, "TTM2_3_nV_solved")
    print(TTM2_4, "TTM2_4_nV_solved\n")

    result_viewer([TTM1_1, TTM1_2, TTM1_3, TTM1_4, TTM2_1, TTM2_2, TTM2_3, TTM2_4])

    print('\ntime =', round(time.time()-start, 5), 'sec')
    print('\n----------------END----------------\n')
    plt.show()