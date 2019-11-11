#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"//导入CNN文件
#include "minst.h"//导入手写数字输入文件
#include "add.cuh"
#include <iostream>
using namespace std;
//#define S
#define TRAIN 0//此值为1则有训练过程
#define DEBUG 1
/*主函数*/
#ifdef S

void conv2(float* src, float* dst, float* filter, int imageOutSize, int imageInSize, int filterSize)
{
	int fSize = filterSize* filterSize;
	for (int col = 0;col < imageInSize;col++) {
		for (int row = 0;row < imageInSize;row++) {
			fSize = filterSize* filterSize;
			int dstIndex = col * imageOutSize + row;
			dst[dstIndex] = 0;
			for (int fy = 0; fy < filterSize; fy++) {
				for (int fx = 0; fx < filterSize; fx++) {
					float filterItem = filter[--fSize];
					float imageItem = src[row + fx + (fy + col)*imageInSize];
					dst[dstIndex] += filterItem*imageItem;
				}
			}
		}
	}

}
#endif // DEBUG
int main()
{
#if DEBUG
	initCuda();//初始化cuda
	clock_t start, finish, start1, finish1;//计算时间用的
	double  duration;
	start = clock();//开始计时
	FILE  *fp = NULL;
	fp = fopen("E:\\CNNData\\test1.txt", "w");
	LabelArr trainLabel = read_Lable("E:\\CNN\\train-labels.idx1-ubyte");//读入训练集的标签y
	ImgArr trainImg = read_Img("E:\\CNN\\train-images.idx3-ubyte");//读入训练集的原始图像x
	LabelArr testLabel = read_Lable("E:\\CNN\\t10k-labels.idx1-ubyte");//读入测试集的标签y
	ImgArr testImg = read_Img("E:\\CNN\\t10k-images.idx3-ubyte");//读入训练集的原始图像x
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("readtime:%f seconds\n", duration);
	fprintf(fp, "readtime:%f seconds\n", duration);
	nSize inputSize = { testImg->c,testImg->r };//记录图像大小为28x28
	int outSize = testLabel->l;//记录标签大小为10
	CNN** cnnarray;//分配二维数组来保存BATCHSIZE个CNN网络
	cudaMallocManaged(&cnnarray, sizeof(CNN*)*BATCHSIZE);
	for (int i = 0; i < BATCHSIZE; i++)
	{
		cudaMallocManaged(&cnnarray[i], sizeof(CNN));
	}

	cnnsetup(cnnarray, inputSize, outSize, fp);//初始化CNN网络

										  // CNN训练

	CNNOpts opts;
	opts.numepochs = 10;//训练次数,默认为1
	opts.alpha = 1.0;//学习率
	int trainNum = 60000;//暂时未知,应该指的是训练集的训练数量

#if TRAIN 
	cnntrain(cnn, trainImg, trainLabel, opts, trainNum, fp, testImg, testLabel, 100);//训练CNN网络

	savecnn(cnn, "E:\\minst.cnn");//保存CNN网络
								  // 保存训练误差
	FILE  *fp2 = NULL;
	fp2 = fopen("E:\\cnnL.ma", "wb");
	if (fp2 == NULL)
		printf("write file failed\n");
	fwrite(cnn->L, sizeof(float), trainNum, fp2);
	fclose(fp2);
#endif // TRAIN
	// CNN测试
	importcnn(cnnarray[0], "E:\\mnist.cnn");
	cnncpy(cnnarray);
	int testNum = 10000;//测试集的测试数量
	float incorrectRatio = 0.0;//错误率,默认为0
	start1 = clock();
	incorrectRatio = cnntest(cnnarray, testImg, testLabel, testNum, fp);//测试CNN网络,输出错误率
	finish1 = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("testtime:%f seconds\n", duration);
	fprintf(fp, "testtime:%f seconds\n", duration);
	printf("test finished!!,error=%f\n", incorrectRatio);
	fprintf(fp, "test finished!!,error=%f\n", incorrectRatio);
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("totaltime:%f seconds\n", duration);
	fprintf(fp, "totaltime:%f seconds\n", duration);
	fclose(fp);
#endif // !DEBUG

	system("pause");
	return 0;
}


