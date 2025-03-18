#include <apriltag_math.h>
#include <cuda_runtime.h>
static inline void cpu_mat33_chol(const double* A,
    double* R)
{
    // A[0] = R[0]*R[0]
    R[0] = sqrt(A[0]);

    // A[1] = R[0]*R[3];
    R[3] = A[1] / R[0];

    // A[2] = R[0]*R[6];
    R[6] = A[2] / R[0];

    // A[4] = R[3]*R[3] + R[4]*R[4]
    R[4] = sqrt(A[4] - R[3] * R[3]);

    // A[5] = R[3]*R[6] + R[4]*R[7]
    R[7] = (A[5] - R[3] * R[6]) / R[4];

    // A[8] = R[6]*R[6] + R[7]*R[7] + R[8]*R[8]
    R[8] = sqrt(A[8] - R[6] * R[6] - R[7] * R[7]);

    R[1] = 0;
    R[2] = 0;
    R[5] = 0;
}

static inline void cpu_mat33_lower_tri_inv(const double* A,
    double* R)
{
    // A[0]*R[0] = 1
    R[0] = 1 / A[0];

    // A[3]*R[0] + A[4]*R[3] = 0
    R[3] = -A[3] * R[0] / A[4];

    // A[4]*R[4] = 1
    R[4] = 1 / A[4];

    // A[6]*R[0] + A[7]*R[3] + A[8]*R[6] = 0
    R[6] = (-A[6] * R[0] - A[7] * R[3]) / A[8];

    // A[7]*R[4] + A[8]*R[7] = 0
    R[7] = -A[7] * R[4] / A[8];

    // A[8]*R[8] = 1
    R[8] = 1 / A[8];
}

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// CUDA 错误检查宏
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    std::cerr << "Error: " << cudaGetErrorString(x) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE);}} while(0)

// CUDA 设备端 Cholesky 分解
__device__
void device_mat33_chol(const double* A, double* R) {
    R[0] = sqrt(A[0]);
    R[3] = A[1] / R[0];
    R[6] = A[2] / R[0];
    R[4] = sqrt(A[4] - R[3] * R[3]);
    R[7] = (A[5] - R[3] * R[6]) / R[4];
    R[8] = sqrt(A[8] - R[6] * R[6] - R[7] * R[7]);
    R[1] = R[2] = R[5] = 0;
}

// CUDA 核函数：Cholesky 分解
__global__
void kernel_mat33_chol(const double* A, double* R) {
    device_mat33_chol(A, R);
}

// CUDA 设备端下三角矩阵求逆
__device__
void device_mat33_lower_tri_inv(const double* A, double* R) {
    R[0] = 1 / A[0];
    R[3] = -A[3] * R[0] / A[4];
    R[4] = 1 / A[4];
    R[6] = (-A[6] * R[0] - A[7] * R[3]) / A[8];
    R[7] = -A[7] * R[4] / A[8];
    R[8] = 1 / A[8];
}

// CUDA 核函数：下三角矩阵求逆
__global__
void kernel_mat33_lower_tri_inv(const double* A, double* R) {
    device_mat33_lower_tri_inv(A, R);
}

// CUDA 设备端对称矩阵求解
__device__
void device_mat33_sym_solve(const double* A, const double* B, double* R) {
    double L[9];
    device_mat33_chol(A, L); // Cholesky 分解

    double M[9];
    device_mat33_lower_tri_inv(L, M); // 下三角矩阵求逆

    double tmp[3];
    tmp[0] = M[0] * B[0];
    tmp[1] = M[3] * B[0] + M[4] * B[1];
    tmp[2] = M[6] * B[0] + M[7] * B[1] + M[8] * B[2];

    R[0] = M[0] * tmp[0] + M[3] * tmp[1] + M[6] * tmp[2];
    R[1] = M[4] * tmp[1] + M[7] * tmp[2];
    R[2] = M[8] * tmp[2];
}

// CUDA 核函数：对称矩阵求解
__global__
void kernel_mat33_sym_solve(const double* A, const double* B, double* R) {
    device_mat33_sym_solve(A, B, R);
}

// 主机端接口：Cholesky 分解
void mat33_chol(const double* h_A, double* h_R) {
    double *d_A, *d_R;

    // 分配设备内存
    CUDA_CALL(cudaMalloc((void**)&d_A, sizeof(double) * 9));
    CUDA_CALL(cudaMalloc((void**)&d_R, sizeof(double) * 9));

    // 主机数据复制到设备
    CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(double) * 9, cudaMemcpyHostToDevice));

    // 调用 CUDA 核函数
    kernel_mat33_chol<<<1, 1>>>(d_A, d_R);
    CUDA_CALL(cudaDeviceSynchronize());

    // 设备数据复制回主机
    CUDA_CALL(cudaMemcpy(h_R, d_R, sizeof(double) * 9, cudaMemcpyDeviceToHost));

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_R);
}

// 主机端接口：下三角矩阵求逆
void mat33_lower_tri_inv(const double* h_A, double* h_R) {
    double *d_A, *d_R;

    // 分配设备内存
    CUDA_CALL(cudaMalloc((void**)&d_A, sizeof(double) * 9));
    CUDA_CALL(cudaMalloc((void**)&d_R, sizeof(double) * 9));

    // 主机数据复制到设备
    CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(double) * 9, cudaMemcpyHostToDevice));

    // 调用 CUDA 核函数
    kernel_mat33_lower_tri_inv<<<1, 1>>>(d_A, d_R);
    CUDA_CALL(cudaDeviceSynchronize());

    // 设备数据复制回主机
    CUDA_CALL(cudaMemcpy(h_R, d_R, sizeof(double) * 9, cudaMemcpyDeviceToHost));

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_R);
}

// 主机端接口：对称矩阵求解
void mat33_sym_solve(const double* h_A, const double* h_B, double* h_R) {
    double *d_A, *d_B, *d_R;

    // 分配设备内存
    CUDA_CALL(cudaMalloc((void**)&d_A, sizeof(double) * 9));
    CUDA_CALL(cudaMalloc((void**)&d_B, sizeof(double) * 3));
    CUDA_CALL(cudaMalloc((void**)&d_R, sizeof(double) * 3));

    // 主机数据复制到设备
    CUDA_CALL(cudaMemcpy(d_A, h_A, sizeof(double) * 9, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, sizeof(double) * 3, cudaMemcpyHostToDevice));

    // 调用 CUDA 核函数
    kernel_mat33_sym_solve<<<1, 1>>>(d_A, d_B, d_R);
    CUDA_CALL(cudaDeviceSynchronize());

    // 设备数据复制回主机
    CUDA_CALL(cudaMemcpy(h_R, d_R, sizeof(double) * 3, cudaMemcpyDeviceToHost));

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);
}


