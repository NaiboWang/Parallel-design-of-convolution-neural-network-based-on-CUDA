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

#define AvePool 0//定义池化类型,平均池化
#define MaxPool 1//最大池化
#define MinPool 2//最小池化
#define BATCHSIZE 1000

typedef struct cnn_network {//整个CNN的最外面那一层,包括五个层和误差,层的数目
	int layerNum;//层数目
	int c1inputWidth;   //输入图像的宽
	int c1inputHeight;  //输入图像的长
	int c1mapSize;      //特征模板的大小，模板一般都是正方形
	int c1inChannels;   //输入图像的数目
	int c1outChannels;  //输出图像的数目
	float* c1mapData;     //存放特征模块的数据
	float* c1dmapData;    //存放特征模块的数据的局部梯度
	float* c1basicData;   //偏置，偏置的大小，为outChannels
	float* c1dbasicData;   //偏置的梯度，偏置的大小，为outChannels
	bool c1isFullConnect; //是否为全连接
	bool* c1connectModel; //连接模式（默认为全连接）
	float* c1v; // 进入激活函数的输入值
	float* c1y; // 激活函数后神经元的输出
	float* c1d; // 网络的局部梯度,δ值  
	int c3inputWidth;   //输入图像的宽
	int c3inputHeight;  //输入图像的长
	int c3mapSize;      //特征模板的大小，模板一般都是正方形
	int c3inChannels;   //输入图像的数目
	int c3outChannels;  //输出图像的数目
	float* c3mapData;     //存放特征模块的数据
	float* c3dmapData;    //存放特征模块的数据的局部梯度
	float* c3basicData;   //偏置，偏置的大小，为outChannels
	float* c3dbasicData;   //偏置的梯度，偏置的大小，为outChannels
	bool c3isFullConnect; //是否为全连接
	bool* c3connectModel; //连接模式（默认为全连接）
	int s2inputWidth;   //输入图像的宽
	int s2inputHeight;  //输入图像的长
	int s2mapSize;      //特征模板的大小
	int s2inChannels;   //输入图像的数目
	int s2outChannels;  //输出图像的数目
	int s2poolType;     //Pooling的方法
	float* s2basicData;   //偏置,实际上没有用到
	float* s2y; // 采样函数后神经元的输出,无激活函数
	float* s2d; // 网络的局部梯度,δ值
	int s4inputWidth;   //输入图像的宽
	int s4inputHeight;  //输入图像的长
	int s4mapSize;      //特征模板的大小
	int s4inChannels;   //输入图像的数目
	int s4outChannels;  //输出图像的数目
	int s4poolType;     //Pooling的方法
	float* s4basicData;   //偏置,实际上没有用到
	float* s4y; // 采样函数后神经元的输出,无激活函数
	float* s4d; // 网络的局部梯度,δ值
	float* c3v; // 进入激活函数的输入值
	float* c3y; // 激活函数后神经元的输出
	float* c3d; // 网络的局部梯度,δ值  
	int oinputNum;   //输入数据的数目
	int ooutputNum;  //输出数据的数目
	float* owData; // 权重数据，为一个inputNum*outputNum大小
	float* obasicData;   //偏置，大小为outputNum大小
	float* odwData; // 权重数据梯度，为一个inputNum*outputNum大小
	float* odbasicData;   //偏置梯度，大小为outputNum大小
	float* ov; // 进入激活函数的输入值
	float* oy; // 激活函数后神经元的输出
	float* od; // 网络的局部梯度,δ值
	bool isFullConnect; //是否为全连接
	float* e; // 训练误差
	float* L; // 瞬时误差能量
}CNN;
//训练参数
typedef struct train_opts {
	int numepochs; // 训练的迭代次数
	float alpha; // 学习速率
}CNNOpts;

void cnnsetup(CNN** cnn, nSize inputSize, int outputSize, FILE *fp);//初始化CNN的参数
																   /*
																   CNN网络的训练函数
																   inputData，outputData分别存入训练数据
																   dataNum表明数据数目
																   */
void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum);
// 测试cnn函数,inputData，outputData分别存入测试集数据的x和y
float cnntest(CNN** cnn, ImgArr inputData, LabelArr outputData, int testNum, FILE *fp);
// 保存cnn
void savecnn(CNN* cnn, const char* filename);
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename);

void cnnupdategrad(CNN** cnnarray);
// 初始化卷积层
// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
__host__ __device__ float activation_Sigma(float input, float bas); // sigma激活函数
__global__ void testcnn(CNN** cnn, float* inputData, float* LabelData, int* wrongnum);
void cnnff(CNN* cnn, float* inputData); // 网络的前向传播
void cnnbp(CNN* cnn, float* outputData); // 网络的后向传播
void cnnapplygrads(CNN* cnn, CNNOpts opts, float* inputData);//更新网络的权值
void cnnclear(CNN* cnn); // 将数据vyd清零

						 /*
						 Pooling Function
						 input 输入数据
						 inputNum 输入数据数目
						 mapSize 求平均的模块区域
						 */
void avgPooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize); // 求平均值

																								/*
																								单层全连接神经网络的处理
																								nnSize是网络的大小
																								*/
void nnff(float* output, float* input, float** wdata, nSize nnSize); // 单层全连接神经网络的前向传播

void savecnndata(CNN* cnn, const char* filename, float** inputdata); // 保存CNN网络中的相关数据
void initCovLayer(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void initCovLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
void initPoolLayer(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
void initPoolLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType);
void initOutLayer(CNN* cnn, int inputNum, int outputNum);
void cnncpy(CNN** cnnarray);
__global__ void cnntrains(CNN** cnn, float* IData, float* LData, int bs);
#endif
