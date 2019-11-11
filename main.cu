#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"//����CNN�ļ�
#include "minst.h"//������д���������ļ�
#include "add.cuh"
#include <iostream>
using namespace std;
//#define S
#define TRAIN 0//��ֵΪ1����ѵ������
#define DEBUG 1
/*������*/
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
	initCuda();//��ʼ��cuda
	clock_t start, finish, start1, finish1;//����ʱ���õ�
	double  duration;
	start = clock();//��ʼ��ʱ
	FILE  *fp = NULL;
	fp = fopen("E:\\CNNData\\test1.txt", "w");
	LabelArr trainLabel = read_Lable("E:\\CNN\\train-labels.idx1-ubyte");//����ѵ�����ı�ǩy
	ImgArr trainImg = read_Img("E:\\CNN\\train-images.idx3-ubyte");//����ѵ������ԭʼͼ��x
	LabelArr testLabel = read_Lable("E:\\CNN\\t10k-labels.idx1-ubyte");//������Լ��ı�ǩy
	ImgArr testImg = read_Img("E:\\CNN\\t10k-images.idx3-ubyte");//����ѵ������ԭʼͼ��x
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("readtime:%f seconds\n", duration);
	fprintf(fp, "readtime:%f seconds\n", duration);
	nSize inputSize = { testImg->c,testImg->r };//��¼ͼ���СΪ28x28
	int outSize = testLabel->l;//��¼��ǩ��СΪ10
	CNN** cnnarray;//�����ά����������BATCHSIZE��CNN����
	cudaMallocManaged(&cnnarray, sizeof(CNN*)*BATCHSIZE);
	for (int i = 0; i < BATCHSIZE; i++)
	{
		cudaMallocManaged(&cnnarray[i], sizeof(CNN));
	}

	cnnsetup(cnnarray, inputSize, outSize, fp);//��ʼ��CNN����

										  // CNNѵ��

	CNNOpts opts;
	opts.numepochs = 10;//ѵ������,Ĭ��Ϊ1
	opts.alpha = 1.0;//ѧϰ��
	int trainNum = 60000;//��ʱδ֪,Ӧ��ָ����ѵ������ѵ������

#if TRAIN 
	cnntrain(cnn, trainImg, trainLabel, opts, trainNum, fp, testImg, testLabel, 100);//ѵ��CNN����

	savecnn(cnn, "E:\\minst.cnn");//����CNN����
								  // ����ѵ�����
	FILE  *fp2 = NULL;
	fp2 = fopen("E:\\cnnL.ma", "wb");
	if (fp2 == NULL)
		printf("write file failed\n");
	fwrite(cnn->L, sizeof(float), trainNum, fp2);
	fclose(fp2);
#endif // TRAIN
	// CNN����
	importcnn(cnnarray[0], "E:\\mnist.cnn");
	cnncpy(cnnarray);
	int testNum = 10000;//���Լ��Ĳ�������
	float incorrectRatio = 0.0;//������,Ĭ��Ϊ0
	start1 = clock();
	incorrectRatio = cnntest(cnnarray, testImg, testLabel, testNum, fp);//����CNN����,���������
	finish1 = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("testtime:%f seconds\n", duration);
	fprintf(fp, "testtime:%f seconds\n", duration);
	printf("test finished!!,error=%f\n", incorrectRatio);
	fprintf(fp, "test finished!!,error=%f\n", incorrectRatio);
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("totaltime:%f seconds\n", duration);
	fprintf(fp, "totaltime:%f seconds\n", duration);
	fclose(fp);
#endif // !DEBUG

	system("pause");
	return 0;
}


