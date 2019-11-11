#ifndef __CNN_
#define __CNN_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "mat.h"
#include "minst.h"

#define AvePool 0//����ػ�����,ƽ���ػ�
#define MaxPool 1//���ػ�
#define MinPool 2//��С�ػ�
#define BATCHSIZE 1000

typedef struct cnn_network {//����CNN����������һ��,�������������,�����Ŀ
	int layerNum;//����Ŀ
	int c1inputWidth;   //����ͼ��Ŀ�
	int c1inputHeight;  //����ͼ��ĳ�
	int c1mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
	int c1inChannels;   //����ͼ�����Ŀ
	int c1outChannels;  //���ͼ�����Ŀ
	float* c1mapData;     //�������ģ�������
	float* c1dmapData;    //�������ģ������ݵľֲ��ݶ�
	float* c1basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	float* c1dbasicData;   //ƫ�õ��ݶȣ�ƫ�õĴ�С��ΪoutChannels
	bool c1isFullConnect; //�Ƿ�Ϊȫ����
	bool* c1connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�
	float* c1v; // ���뼤���������ֵ
	float* c1y; // ���������Ԫ�����
	float* c1d; // ����ľֲ��ݶ�,��ֵ  
	int c3inputWidth;   //����ͼ��Ŀ�
	int c3inputHeight;  //����ͼ��ĳ�
	int c3mapSize;      //����ģ��Ĵ�С��ģ��һ�㶼��������
	int c3inChannels;   //����ͼ�����Ŀ
	int c3outChannels;  //���ͼ�����Ŀ
	float* c3mapData;     //�������ģ�������
	float* c3dmapData;    //�������ģ������ݵľֲ��ݶ�
	float* c3basicData;   //ƫ�ã�ƫ�õĴ�С��ΪoutChannels
	float* c3dbasicData;   //ƫ�õ��ݶȣ�ƫ�õĴ�С��ΪoutChannels
	bool c3isFullConnect; //�Ƿ�Ϊȫ����
	bool* c3connectModel; //����ģʽ��Ĭ��Ϊȫ���ӣ�
	int s2inputWidth;   //����ͼ��Ŀ�
	int s2inputHeight;  //����ͼ��ĳ�
	int s2mapSize;      //����ģ��Ĵ�С
	int s2inChannels;   //����ͼ�����Ŀ
	int s2outChannels;  //���ͼ�����Ŀ
	int s2poolType;     //Pooling�ķ���
	float* s2basicData;   //ƫ��,ʵ����û���õ�
	float* s2y; // ������������Ԫ�����,�޼����
	float* s2d; // ����ľֲ��ݶ�,��ֵ
	int s4inputWidth;   //����ͼ��Ŀ�
	int s4inputHeight;  //����ͼ��ĳ�
	int s4mapSize;      //����ģ��Ĵ�С
	int s4inChannels;   //����ͼ�����Ŀ
	int s4outChannels;  //���ͼ�����Ŀ
	int s4poolType;     //Pooling�ķ���
	float* s4basicData;   //ƫ��,ʵ����û���õ�
	float* s4y; // ������������Ԫ�����,�޼����
	float* s4d; // ����ľֲ��ݶ�,��ֵ
	float* c3v; // ���뼤���������ֵ
	float* c3y; // ���������Ԫ�����
	float* c3d; // ����ľֲ��ݶ�,��ֵ  
	int oinputNum;   //�������ݵ���Ŀ
	int ooutputNum;  //������ݵ���Ŀ
	float* owData; // Ȩ�����ݣ�Ϊһ��inputNum*outputNum��С
	float* obasicData;   //ƫ�ã���СΪoutputNum��С
	float* odwData; // Ȩ�������ݶȣ�Ϊһ��inputNum*outputNum��С
	float* odbasicData;   //ƫ���ݶȣ���СΪoutputNum��С
	float* ov; // ���뼤���������ֵ
	float* oy; // ���������Ԫ�����
	float* od; // ����ľֲ��ݶ�,��ֵ
	bool isFullConnect; //�Ƿ�Ϊȫ����
	float* e; // ѵ�����
	float* L; // ˲ʱ�������
}CNN;
//ѵ������
typedef struct train_opts {
	int numepochs; // ѵ���ĵ�������
	float alpha; // ѧϰ����
}CNNOpts;

void cnnsetup(CNN** cnn, nSize inputSize, int outputSize, FILE *fp);//��ʼ��CNN�Ĳ���
																   /*
																   CNN�����ѵ������
																   inputData��outputData�ֱ����ѵ������
																   dataNum����������Ŀ
																   */
void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum);
// ����cnn����,inputData��outputData�ֱ������Լ����ݵ�x��y
float cnntest(CNN** cnn, ImgArr inputData, LabelArr outputData, int testNum, FILE *fp);
// ����cnn
void savecnn(CNN* cnn, const char* filename);
// ����cnn������
void importcnn(CNN* cnn, const char* filename);

void cnnupdategrad(CNN** cnnarray);
// ��ʼ�������
// ����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
__host__ __device__ float activation_Sigma(float input, float bas); // sigma�����
__global__ void testcnn(CNN** cnn, float* inputData, float* LabelData, int* wrongnum);
void cnnff(CNN* cnn, float* inputData); // �����ǰ�򴫲�
void cnnbp(CNN* cnn, float* outputData); // ����ĺ��򴫲�
void cnnapplygrads(CNN* cnn, CNNOpts opts, float* inputData);//���������Ȩֵ
void cnnclear(CNN* cnn); // ������vyd����

						 /*
						 Pooling Function
						 input ��������
						 inputNum ����������Ŀ
						 mapSize ��ƽ����ģ������
						 */
void avgPooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize); // ��ƽ��ֵ

																								/*
																								����ȫ����������Ĵ���
																								nnSize������Ĵ�С
																								*/
void nnff(float* output, float* input, float** wdata, nSize nnSize); // ����ȫ�����������ǰ�򴫲�

void savecnndata(CNN* cnn, const char* filename, float** inputdata); // ����CNN�����е��������
void initCovLayer(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void initCovLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void initPoolLayer(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
void initPoolLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
void initOutLayer(CNN* cnn, int inputNum, int outputNum);
void cnncpy(CNN** cnnarray);
__global__ void cnntrains(CNN** cnn, float* IData, float* LData, int bs);
#endif
