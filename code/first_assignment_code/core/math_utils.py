"""
数学工具模块
包含矩阵变换、四元数等数学功能
"""

import numpy as np

def LookAt(eye, center, up):

    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward

    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)

    return view

def Ortho(left, right, bottom, top, near, far):

    matrix = np.eye(4)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (far - near)
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(far + near) / (far - near)
    
    return matrix

def Perspective(fov, aspect=1.0, near=0.1, far=10.0):
    """构建透视投影矩阵
    参数：
        fov (float): 垂直视野角度（Field of View Y），单位为度（degrees）
        aspect (float): 宽高比（aspect ratio），默认值为1.0
        near (float): 近平面，视锥体近端平面到相机的距离 （正值）
        far (float): 远平面，视锥体远端平面到相机的距离   (正值， near < far)
    返回：
        4x4透视投影矩阵
    """
    fov_rad = np.deg2rad(fov)
    f = 1.0 / np.tan(fov_rad / 2)
    
    matrix = np.eye(4)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2 * far * near) / (near - far)
    matrix[3, 2] = -1
    matrix[3, 3] = 0
    
    return matrix

class Quaternion:
    """四元数类，用于表示旋转"""
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w  # 实部
        self.x = x  # 虚部i
        self.y = y  # 虚部j
        self.z = z  # 虚部k
    
    @classmethod
    def from_axis_angle(cls, axis, angle):
        """从轴角表示创建四元数"""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        
        return cls(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)
    
    @classmethod
    def from_euler(cls, rx, ry, rz):
        """从欧拉角创建四元数（ZYX顺序）"""
        qx = cls.from_axis_angle(np.array([1, 0, 0]), rx)
        qy = cls.from_axis_angle(np.array([0, 1, 0]), ry)
        qz = cls.from_axis_angle(np.array([0, 0, 1]), rz)
        
        return qz * qy * qx
    
    def normalize(self):
        """归一化四元数"""
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm > 1e-8:
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
        return self
    
    def conjugate(self):
        """四元数共轭"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def __mul__(self, other):
        """四元数乘法"""
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        else:
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def __add__(self, other):
        """四元数加法"""
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
    
    def dot(self, other):
        """四元数点积"""
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    
    def to_rotation_matrix(self):
        """转换为3x3旋转矩阵"""
        self.normalize()
        R = np.eye(3)
        R[0,0] = 1 - 2*(self.y**2 + self.z**2)
        R[0,1] = 2*(self.x*self.y - self.z*self.w)
        R[0,2] = 2*(self.x*self.z + self.y*self.w)
        R[1,0] = 2*(self.x*self.y + self.z*self.w)
        R[1,1] = 1 - 2*(self.x**2 + self.z**2)
        R[1,2] = 2*(self.y*self.z - self.x*self.w)
        R[2,0] = 2*(self.x*self.z - self.y*self.w)
        R[2,1] = 2*(self.y*self.z + self.x*self.w)
        R[2,2] = 1 - 2*(self.x**2 + self.y**2)

        return R
    
    @staticmethod
    def slerp(q1, q2, t):
        """球面线性插值"""
        q1.normalize()
        q2.normalize()
        
        d = q1.dot(q2)
        if d < 0:
            q2 = Quaternion(-q2.w, -q2.x, -q2.y, -q2.z)
            d = -d

        if d > 0.9995:
            result=Quaternion(
                q1.w + t * (q2.w - q1.w),
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z)
            )
            return result.normalize()

        theta = np.arccos(d)
        sin_theta = np.sin(theta)
        t0 = np.sin((1 - t) * theta) / sin_theta
        t1 = np.sin(t * theta) / sin_theta

        return Quaternion(
            t0 * q1.w + t1 * q2.w,
            t0 * q1.x + t1 * q2.x,
            t0 * q1.y + t1 * q2.y,
            t0 * q1.z + t1 * q2.z
        ).normalize()
