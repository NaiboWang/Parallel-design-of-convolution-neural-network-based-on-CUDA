// ������ļ���Ҫ���ڹ��ڶ�ά��������Ĳ���
#ifndef __MAT_
#define __MAT_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "add.cuh"
#define full 0//��ȫ���
#define same 1//�����ͬ��С�����־��,MATLAB CONV����������������
#define valid 2//�������

typedef struct Mat2DSize{//��������С�Ľṹ��,c��r��ʾ����������
	int c; // ��������ȣ�
	int r; // �������߶ȣ�
}nSize;

float** rotate180(float* mat, nSize matSize);// ����ת180��
float** rotate180(float** mat, nSize matSize);// ����ת180��

// ���������Ԫ�ض�Ӧλ�����,mat1��mat2�õ�res,�����С����
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// �������

float** correlation(float** map,nSize mapSize,float* inputData,nSize inSize,int type);// �����,Э����?
float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);// �����,Э����?

// �������,map��������,mapSizeΪ����˴�С,inputData��Ҫ���������,inSize��Ҫ������ݵĴ�С,typeΪ�������
float** cov(float** map,nSize mapSize,float* inputData,nSize inSize,int type); // �������
float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type); // �������

// ����Ǿ�����ϲ�������ֵ�ڲ壩��upc��upr���ڲ屶��
float** UpSample(float** mat,nSize matSize,int upc,int upr);

// ����ά�����Ե��������addw��С��0ֵ��,������ȫ���
float** matEdgeExpand(float* mat,nSize matSize,int addc,int addr);
float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);

// ����ά�����Ե��С������shrinkc��С�ı�,������ȫ���֮��Ļ�ԭ
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr);

void savemat(float** mat,nSize matSize,const char* filename);// �����������

void multifactor(float** res, float** mat, nSize matSize, float factor);// �������ϵ��

float summat(float** mat,nSize matSize);// �����Ԫ�صĺ�

char * combine_strings(char *a, char *b);

char* intTochar(int i);

#endif