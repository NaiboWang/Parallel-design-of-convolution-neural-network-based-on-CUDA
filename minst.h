#ifndef __MINST_//如果没有编译过这个,编译,不然不编译
#define __MINST_
/*
MINST数据库是一个手写图像数据库，里面的结构数据详情请看：http://m.blog.csdn.net/article/details?id=53257185
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
//60000张原始图像
typedef struct MinstImgArr{
	int ImgNum;        // 存储图像的数目,这里训练集是60000,测试集为10000
	int c;
	int r;
	float* ImgPtr;  // 存储图像指针数组,每一张就是上面的imgNumX28X28的结构体
}*ImgArr;              // 存储图像数据的数组,注意是指针类型
//用来存标签的结构体
typedef struct MinstLabelArr{
	int LabelNum;//存储标签数目,这里训练集是60000,测试集为10000
	int l;
	float* LabelPtr;// 存储标签指针数组,每一张就是上面1个标签结构体 labelNum*l
}*LabelArr;              // 存储图像标记的数组

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
LabelArr read_Lable(const char* filename); // 读入图像标记

ImgArr read_Img(const char* filename); // 读入图像

void save_Img(ImgArr imgarr,char* filedir); // 将图像数据保存成文件

#endif