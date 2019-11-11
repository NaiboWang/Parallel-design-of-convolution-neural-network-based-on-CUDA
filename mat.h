// 这里库文件主要存在关于二维矩阵数组的操作
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "add.cuh"
#define full 0//完全卷积
#define same 1//输出相同大小的那种卷积,MATLAB CONV函数有这三种类型
#define valid 2//正常卷积

typedef struct Mat2DSize{//定义矩阵大小的结构体,c和r表示列数和行数
	int c; // 列数（宽度）
	int r; // 行数（高度）
}nSize;

float** rotate180(float* mat, nSize matSize);// 矩阵翻转180度
float** rotate180(float** mat, nSize matSize);// 矩阵翻转180度

// 两个矩阵各元素对应位置相加,mat1加mat2得到res,矩阵大小不变
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// 矩阵相加

float** correlation(float** map,nSize mapSize,float* inputData,nSize inSize,int type);// 互相关,协方差?
float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);// 互相关,协方差?

// 卷积操作,map代表卷积核,mapSize为卷积核大小,inputData是要卷积的数据,inSize是要卷积数据的大小,type为卷积类型
float** cov(float** map,nSize mapSize,float* inputData,nSize inSize,int type); // 卷积操作
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type); // 卷积操作

// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
float** UpSample(float** mat,nSize matSize,int upc,int upr);

// 给二维矩阵边缘扩大，增加addw大小的0值边,用于完全卷积
float** matEdgeExpand(float* mat,nSize matSize,int addc,int addr);
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);

// 给二维矩阵边缘缩小，擦除shrinkc大小的边,用于完全卷积之后的还原
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr);

void savemat(float** mat,nSize matSize,const char* filename);// 保存矩阵数据

void multifactor(float** res, float** mat, nSize matSize, float factor);// 矩阵乘以系数

float summat(float** mat,nSize matSize);// 矩阵各元素的和

char * combine_strings(char *a, char *b);

char* intTochar(int i);

#endif