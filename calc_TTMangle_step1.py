from math import radians
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# ---- 単位は mm ----


N_air = 1.0  # Refractive index of the Air
N_glass = 1.458  # Refractive index (Fused Silica at 589nm)
t_bs = 6.0      # Beam Splitter substrate thickness [mm]
BE_power = 3  # BE magnifying power
LLT_power = 12.5  # LLT magnifying power
BE_pupill_R = 18  # BE pupil radius [mm]

# ミラーパラメータ [[X, Y, Z], [nV_x, nV_y, nV_z], R]
FM_params = [
    [[0.000, 0.000, 0.000], [0.70710678, -0.70710678, 0.00000000], 12.7]]  # FM4-1

BS_params = [
    [[125.000, 0.000, 0.000], [-0.70613770, 0.70613770, 0.05233590], 12.7],
    [[129.159, -4.159, -0.308], [-0.70616460, 0.70616460, 0.05233790], 12.7],
    [[179.202, -1.064, 0.000], [-0.70864990, 0.70361660, 0.05233560], 12.7],
    [[183.375, -5.207, -0.308], [-0.70867680, 0.70364340, 0.05233770], 12.7],
    [[228.072, -1.910, 0.000], [-0.71113880, 0.70104420, -0.05309080], 12.7],
    [[232.260, -6.038, 0.313], [-0.71116590, 0.70107090, -0.05309280], 12.7],
    [[275.287, -2.233, 0.000], [-0.71366120, 0.69847630, -0.05309050], 12.7]
    ]  # BS1_f, BS1_b, BS2_f, BS2_b, BS3_f, BS3_b, BS4

TTM1_params = [
    [[125.000, 315.000, 20.000], [-0.70444000, -0.70664860, -0.06642240], 12.7],
    [[179.202, 270.000, 20.000], [-0.71138250, -0.69999730, -0.06275900], 12.7],
    [[228.072, 315.000, -20.000], [-0.70464120, -0.70675270, 0.06309880], 12.7],
    [[275.287, 270.000, -20.000], [-0.70981330, -0.70173860, 0.06105730], 12.7]
    ]  # TTM1-1, TTM1-2, TTM1-3, TTM1-4

TTM2_params = [
    [[-103.000, 315.000, 20.000], [0.67442920, -0.73592180, -0.05970220], 12.7],
    [[-149.000, 270.000, 20.000], [0.75861100, -0.64887440, -0.05891950], 12.7],
    [[-103.000, 315.000, -20.000], [0.67429800, -0.73574620, 0.06324330], 12.7],
    [[-149.000, 270.000, -20.000], [0.76008120, -0.64700820, 0.06047280], 12.7]
    ]  # TTM2-1, TTM2-2, TTM2-3, TTM2-4

PRISM_front_params = [
    [[-112.300, 166.150, 12.700], [-0.20355850, 0.95766790, -0.20355860], 12.7],
    [[-137.700, 166.490, 12.700], [0.31073190, 0.92867950, -0.20248490], 12.7],
    [[-112.300, 166.150, -12.700], [-0.20355850, 0.95766790, 0.20355860], 12.7],
    [[-137.700, 166.490, -12.700], [0.31073190, 0.92867950, 0.20248490], 12.7]
    ]  # PRISM-1_Front, PRISM-2_Front, PRISM-3_Front, PRISM-4_Front, PRISM_Back

PRISM_back_params = [
    [[-112.300, 150.000, 12.700], [0.00000000, 1.00000000, 0.00000000], 12.7],
    [[-137.700, 150.000, 12.700], [0.00000000, 1.00000000, 0.00000000], 12.7],
    [[-112.300, 150.000, -12.700], [0.00000000, 1.00000000, 0.00000000], 12.7],
    [[-137.700, 150.000, -12.700], [0.00000000, 1.00000000, 0.00000000], 12.7]
    ]  # PRISM-1_Back, PRISM-2_Back, PRISM-3_Back, PRISM-4_Back

FBM_params = [
    [[-125.000, 90.600, 0.000], [0.70710680, 0.70710680, 0.00000000], 25.4]]  # FBM

BE_entrance_pupill_params = [
    [[1955.500, 90.600, 0.000], [-1.00000000, 0.00000000, 0.00000000], BE_pupill_R]]  # BE_entrance_pupil


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

    # レイトレーシング平板の交点を持つときの係数Tを計算する関数
    def calcT_mirror(self, ray_pos, ray_dir, centerV, normalV):
        nV = np.array(normalV)/np.linalg.norm(normalV)
        T = (np.dot(centerV, nV)-np.dot(ray_pos, nV)) / (np.dot(ray_dir, nV))
        return T

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
            print("error")
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
def mirror_reflection(mode_flag=0):
    # インスタンス生成
    VF = VectorFunctions()

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
    for i in BE_entrance_pupill_params:
        plot_mirror(i)


    # 光線追跡
    VF.plotLineBlack([-200,90.6,0], [1955.5,90.6,0])  # 仮
    VF.plotLineBlack([-100,0,0], [400,0,0])  # 仮

    # 初期値
    s_pos = np.array([0.0, 0.0, 0.0])
    s_dir = np.array([1.0, 0.0, 0.0])

    # BS
    def BS_raytrace():
        # ----------------BS1----------------
        pos_BS1_f = BS_params[0][0]
        nV_BS1_f = BS_params[0][1]
        nV_BS1_f = nV_BS1_f / np.linalg.norm(nV_BS1_f)
        front_pos_BS1 = s_pos + s_dir*VF.calcT_mirror(s_pos, s_dir, pos_BS1_f, nV_BS1_f)
        front_dir_BS1 = VF.calcReflectionV(s_dir, nV_BS1_f)
        VF.plotLineBlue(s_pos, front_pos_BS1)

        glass_dir_BS1 = VF.calcRefractionV(s_dir, nV_BS1_f, N_air, N_glass)
        pos_BS1_b = BS_params[1][0]
        nV_BS1_b = BS_params[1][1]
        nV_BS1_b = nV_BS1_b / np.linalg.norm(nV_BS1_b)
        print("0.5deg")
        print("BS1_deg", np.degrees(np.arccos(np.dot(nV_BS1_f, nV_BS1_b))))
        back_pos_BS1 = front_pos_BS1 + glass_dir_BS1*VF.calcT_mirror(front_pos_BS1, glass_dir_BS1, pos_BS1_b, nV_BS1_b)
        back_dir_BS1 = VF.calcRefractionV(glass_dir_BS1, nV_BS1_b, N_glass, N_air)
        VF.plotLineGreen(front_pos_BS1, back_pos_BS1)

        # ----------------BS2----------------
        pos_BS2_f = BS_params[2][0]
        nV_BS2_f = BS_params[2][1]
        nV_BS2_f = nV_BS2_f / np.linalg.norm(nV_BS2_f)
        front_pos_BS2 = back_pos_BS1 + back_dir_BS1*VF.calcT_mirror(back_pos_BS1, back_dir_BS1, pos_BS2_f, nV_BS2_f)
        front_dir_BS2 = VF.calcReflectionV(back_dir_BS1, nV_BS2_f)
        VF.plotLineGreen(back_pos_BS1, front_pos_BS2)

        glass_dir_BS2 = VF.calcRefractionV(back_dir_BS1, nV_BS2_f, N_air, N_glass)
        pos_BS2_b = BS_params[3][0]
        nV_BS2_b = BS_params[3][1]
        nV_BS2_b = nV_BS2_b / np.linalg.norm(nV_BS2_b)
        print("BS2_deg", np.degrees(np.arccos(np.dot(nV_BS2_f, nV_BS2_b))))
        back_pos_BS2 = front_pos_BS2 + glass_dir_BS2*VF.calcT_mirror(front_pos_BS2, glass_dir_BS2, pos_BS2_b, nV_BS2_b)
        back_dir_BS2 = VF.calcRefractionV(glass_dir_BS2, nV_BS2_b, N_glass, N_air)
        VF.plotLineRed(front_pos_BS2, back_pos_BS2)

        # ----------------BS3----------------
        pos_BS3_f = BS_params[4][0]
        nV_BS3_f = BS_params[4][1]
        nV_BS3_f = nV_BS3_f / np.linalg.norm(nV_BS3_f)
        front_pos_BS3 = back_pos_BS2 + back_dir_BS2*VF.calcT_mirror(back_pos_BS2, back_dir_BS2, pos_BS3_f, nV_BS3_f)
        front_dir_BS3 = VF.calcReflectionV(back_dir_BS2, nV_BS3_f)
        VF.plotLineRed(back_pos_BS2, front_pos_BS3)

        glass_dir_BS3 = VF.calcRefractionV(back_dir_BS2, nV_BS3_f, N_air, N_glass)
        pos_BS3_b = BS_params[5][0]
        nV_BS3_b = BS_params[5][1]
        nV_BS3_b = nV_BS3_b / np.linalg.norm(nV_BS3_b)
        print("BS3_deg", np.degrees(np.arccos(np.dot(nV_BS3_f, nV_BS3_b))))
        back_pos_BS3 = front_pos_BS3 + glass_dir_BS3*VF.calcT_mirror(front_pos_BS3, glass_dir_BS3, pos_BS3_b, nV_BS3_b)
        back_dir_BS3 = VF.calcRefractionV(glass_dir_BS3, nV_BS3_b, N_glass, N_air)
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
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], TTM1_params[i][0], TTM1_params[i][1])
            dir = VF.calcReflectionV(last_dir[i], nV_TTM1)
            pos_TTM1.append(pos)
            dir_TTM1.append(dir)

        VF.plotLineBlue(last_pos[0], pos_TTM1[0])
        VF.plotLineGreen(last_pos[1], pos_TTM1[1])
        VF.plotLineRed(last_pos[2], pos_TTM1[2])
        VF.plotLineOrange(last_pos[3], pos_TTM1[3])
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
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], TTM2_params[i][0], TTM2_params[i][1])
            dir = VF.calcReflectionV(last_dir[i], nV_TTM2)
            pos_TTM2.append(pos)
            dir_TTM2.append(dir)

        VF.plotLineBlue(last_pos[0], pos_TTM2[0])
        VF.plotLineGreen(last_pos[1], pos_TTM2[1])
        VF.plotLineRed(last_pos[2], pos_TTM2[2])
        VF.plotLineOrange(last_pos[3], pos_TTM2[3])
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
            #print(nV_PRISM)
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], PRISM_front_params[i][0], PRISM_front_params[i][1])
            dir = VF.calcRefractionV(last_dir[i], nV_PRISM_front, N_air, N_glass)
            pos_PRISM_front.append(pos)
            dir_PRISM_front.append(dir)

        VF.plotLineBlue(last_pos[0], pos_PRISM_front[0])
        VF.plotLineGreen(last_pos[1], pos_PRISM_front[1])
        VF.plotLineRed(last_pos[2], pos_PRISM_front[2])
        VF.plotLineOrange(last_pos[3], pos_PRISM_front[3])
        return pos_PRISM_front, dir_PRISM_front

    def PRISM_back_raytrace():
        relay_params = PRISM_front_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_PRISM_back = []
        dir_PRISM_back = []
        for i in range(4):
            nV_PRISM_back = PRISM_back_params[i][1]
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], PRISM_back_params[i][0], PRISM_back_params[i][1])
            dir = VF.calcRefractionV(last_dir[i], nV_PRISM_back, N_glass, N_air)
            pos_PRISM_back.append(pos)
            dir_PRISM_back.append(dir)

        VF.plotLineBlue(last_pos[0], pos_PRISM_back[0])
        VF.plotLineGreen(last_pos[1], pos_PRISM_back[1])
        VF.plotLineRed(last_pos[2], pos_PRISM_back[2])
        VF.plotLineOrange(last_pos[3], pos_PRISM_back[3])
        return pos_PRISM_back, dir_PRISM_back

    def FBM_raytrace():
        relay_params = PRISM_back_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

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
        return pos_FBM, dir_FBM

    def BE_entrance_pupill_raytrace():
        relay_params = FBM_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        pos_BE_entrance_pupill = []
        dir_BE_entrance_pupill = last_dir
        for i in range(4):
            pos = last_pos[i] + last_dir[i]*VF.calcT_mirror(last_pos[i], last_dir[i], BE_entrance_pupill_params[0][0], BE_entrance_pupill_params[0][1])
            pos_BE_entrance_pupill.append(pos)

        VF.plotLineBlue(last_pos[0], pos_BE_entrance_pupill[0])
        VF.plotLineGreen(last_pos[1], pos_BE_entrance_pupill[1])
        VF.plotLineRed(last_pos[2], pos_BE_entrance_pupill[2])
        VF.plotLineOrange(last_pos[3], pos_BE_entrance_pupill[3])
        return pos_BE_entrance_pupill, dir_BE_entrance_pupill

    def beam_angle():
        relay_params = BE_entrance_pupill_raytrace()
        last_pos = relay_params[0]
        last_dir = relay_params[1]

        nV_BE_entrance_pupill = BE_entrance_pupill_params[0][1]
        rad_list = []
        for i in range(4):
            rad_list.append(np.arccos(-np.dot(last_dir[i], nV_BE_entrance_pupill)))

        deg_list = np.degrees(rad_list)

        print("pupill_center : ", BE_entrance_pupill_params[0][0], "mm(瞳位置)")

        print("beam1 : ", last_pos[0], "mm(瞳位置), ", round(deg_list[0], 5), "deg(瞳位置), ",
            round(deg_list[0]*3600/(BE_power*LLT_power), 5), "sec(打ち上げ位置)")
        if (last_pos[0][1]>0 or last_pos[0][2]<0 or last_pos[0][1]**2+last_pos[0][2]**2>BE_pupill_R**2).any():
            #print((last_pos[0][1]-BE_entrance_pupill_params[0][0][1])**2+(last_pos[0][2]-BE_entrance_pupill_params[0][0][2])**2)
            print("    ### beam1_ERORR (blue) ###")
        print("beam2 : ", last_pos[1], "mm(瞳位置), ", round(deg_list[1], 5), "deg(瞳位置), ",
            round(deg_list[1]*3600/(BE_power*LLT_power), 5), "sec(打ち上げ位置)")
        if (last_pos[1][1]<0 or last_pos[1][2]<0 or last_pos[1][1]**2+last_pos[1][2]**2>BE_pupill_R**2).any():
            print("    ### beam2_ERORR (green) ###")
        print("beam3 : ", last_pos[2], "mm(瞳位置), ", round(deg_list[2], 5), "deg(瞳位置), ",
            round(deg_list[2]*3600/(BE_power*LLT_power), 5), "sec(打ち上げ位置)")
        if (last_pos[2][1]>0 or last_pos[2][2]>0 or last_pos[2][1]**2+last_pos[2][2]**2>BE_pupill_R**2).any():
            print("    ### beam3_ERORR (red) ###")
        print("beam4 : ", last_pos[3], "mm(瞳位置), ", round(deg_list[3], 5), "deg(瞳位置), ",
            round(deg_list[3]*3600/(BE_power*LLT_power), 5), "sec(打ち上げ位置)")
        if (last_pos[3][1]<0 or last_pos[3][2]>0 or last_pos[3][1]**2+last_pos[3][2]**2>BE_pupill_R**2).any():
            print("    ### beam4_ERORR (orange) ###")

        return deg_list

    if mode_flag == 0:
        beam_angle()
        # 描画範囲
        LX = 200
        LY = 200
        LZ = 200
        ax.set_xlim(-LX+100, LX+100)  # bird's-eye: +100, BE_entrance_pupill: +1955.5
        ax.view_init(elev=25, azim=-60)
    else:
        BE_entrance_pupill_raytrace()
        # 描画範囲
        LX = 20
        LY = 20
        LZ = 20
        ax.set_xlim(-LX+1955.5, LX+1955.5)  # bird's-eye: +100, BE_entrance_pupill: +1955.5
        ax.view_init(elev=20, azim=-38)

    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    ax.set_ylim(-LY+90.6, LY+90.6)
    ax.set_zlim(-LZ, LZ)



if __name__ == "__main__":
    print('\n----------------START----------------\n')
    start = time.time()
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    mirror_reflection(mode_flag=0)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    mirror_reflection(mode_flag=1)

    print('\ntime =', round(time.time()-start, 5), 'sec')
    print('\n----------------END----------------\n')
    plt.show()