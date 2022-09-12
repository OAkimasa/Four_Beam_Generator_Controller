import numpy as np

# 反射を計算する class
class VectorFunctions:
    # next
    def __init__(self):
        self.ax = None  # plotline関数のため

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
    def calcT_plane(self, ray_pos, ray_dir, centerV, normalV):
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
    def plotLineBlue(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='blue')

    def plotLineGreen(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='green')

    def plotLineRed(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='r')

    def plotLineOrange(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='orange')

    def plotLinePurple(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='purple')

    def plotLineBlack(self, ax, startPointV, endPointV):
        startX = startPointV[0]
        startY = startPointV[1]
        startZ = startPointV[2]
        endX = endPointV[0]
        endY = endPointV[1]
        endZ = endPointV[2]
        ax.plot([startX, endX], [startY, endY], [startZ, endZ],
                'o-', ms='2', linewidth=0.5, color='black')