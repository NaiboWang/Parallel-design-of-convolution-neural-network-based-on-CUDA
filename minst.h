#ifndef __MINST_//���û�б�������,����,��Ȼ������
#define __MINST_
/*
MINST���ݿ���һ����дͼ�����ݿ⣬����Ľṹ���������뿴��http://m.blog.csdn.net/article/details?id=53257185
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
//60000��ԭʼͼ��
typedef struct MinstImgArr{
	int ImgNum;        // �洢ͼ�����Ŀ,����ѵ������60000,���Լ�Ϊ10000
	int c;
	int r;
	float* ImgPtr;  // �洢ͼ��ָ������,ÿһ�ž��������imgNumX28X28�Ľṹ��
}*ImgArr;              // �洢ͼ�����ݵ�����,ע����ָ������
//�������ǩ�Ľṹ��
typedef struct MinstLabelArr{
	int LabelNum;//�洢��ǩ��Ŀ,����ѵ������60000,���Լ�Ϊ10000
	int l;
	float* LabelPtr;// �洢��ǩָ������,ÿһ�ž�������1����ǩ�ṹ�� labelNum*l
}*LabelArr;              // �洢ͼ���ǵ�����

class Time
{
	double s, e;
public:
	Time() {};
	void start()
	{
		s = omp_get_wtime();
	}
	void end()
	{
		e = omp_get_wtime();
		printf_s("start = %.1f,end = %.1f,diff = %.1fs\n",
			s, e, e - s);
	}
};
class GPUT
{
	cudaEvent_t     start, stop;
public:
	GPUT()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	void startT()
	{
		cudaEventRecord(start, 0);
	}
	void endT()
	{
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float   elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Time to generate:  %3.1f ms\n", elapsedTime);
	}
};
LabelArr read_Lable(const char* filename); // ����ͼ����

ImgArr read_Img(const char* filename); // ����ͼ��

void save_Img(ImgArr imgarr,char* filedir); // ��ͼ�����ݱ�����ļ�

#endif