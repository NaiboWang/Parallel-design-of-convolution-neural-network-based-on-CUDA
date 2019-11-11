#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include<iostream>
using namespace std;
clock_t start, finish;//计算时间用的
double  duration;

void cnnsetup(CNN** cnn, nSize inputSize, int outputSize, FILE *fp)
{
	start = clock();//开始计时
	for (int i = 0;i < BATCHSIZE;i++)//初始化BATCHSIZE组cnn[i]网络
	{
		cnn[i]->layerNum = 5;//设置cnn[i][i]层数为5
		nSize inSize;//输入图像大小
		int mapSize = 5;//定义卷积核大小为5
		inSize.c = inputSize.c;//输入图像大小为28X28
		inSize.r = inputSize.r;//输入图像大小为28X28
		initCovLayer(cnn[i], inSize.c, inSize.r, 5, 1, 6);//以输入图像大小为28X28,卷积核大小为5X5,输入图像数为1,输出MAP数为6初始化C1层,具体初始化过程见initCovLayer函数定义
		inSize.c = inSize.c - mapSize + 1;//S2层的输入MAP的大小为28-5+1=24,即24X24
		inSize.r = inSize.r - mapSize + 1;//S2层的输入MAP的大小为28-5+1=24,即24X24
		initPoolLayer(cnn[i], inSize.c, inSize.r, 2, 6, 6, AvePool); //以输入图像大小为24X24, 池化大小为2X2, 输入图像数为6, 输出MAP数为6,池化方法为平均池化初始化S2层, 具体初始化过程见initPoolLayer函数定义
		inSize.c = inSize.c / 2;//C3层的输入图像大小为24/2=12,即12X12
		inSize.r = inSize.r / 2;//C3层的输入图像大小为24/2=12,即12X12
		initCovLayer2(cnn[i], inSize.c, inSize.r, 5, 6, 12);//以输入图像大小为12X12,卷积核大小为5X5,输入图像数为6,输出MAP数为12初始化C3层,具体初始化过程见initCovLayer函数定义
		inSize.c = inSize.c - mapSize + 1;//S4层输入图像大小为12-5+1=8,即8X8
		inSize.r = inSize.r - mapSize + 1;//S4层输入图像大小为12-5+1=8,即8X8
		initPoolLayer2(cnn[i], inSize.c, inSize.r, 2, 12, 12, AvePool);//以输入图像大小为8X8, 池化大小为2X2, 输入图像数为12, 输出MAP数为12,池化方法为平均池化初始化S4层, 具体初始化过程见initPoolLayer函数定义
		inSize.c = inSize.c / 2;//全连接输出层输入图像大小为8/2=4,即4X4
		inSize.r = inSize.r / 2;//全连接输出层输入图像大小为8/2=4,即4X4
		initOutLayer(cnn[i], inSize.c*inSize.r * 12, outputSize);//以输入图像大小为4*4*12=192,输出图像为10初始化输出层,具体初始化过程见initOutLayer函数定义
		cudaMallocManaged(&cnn[i]->e, cnn[i]->ooutputNum*sizeof(float));
		for (int j = 0;j < cnn[i]->ooutputNum;j++)
			cnn[i]->e[i] = 0.0;
	}
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("setuptime:%f seconds\n", duration);
	fprintf(fp, "setuptime:%f seconds\n", duration);
}
//初始化卷积层,参数为输入图像的大小inputWidth,inputHeight,卷积核大小mapSize,输入图像个数inChannels,输出图像个数outChannels
void initCovLayer(CNN* cnn,int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{

	cnn->c1inputHeight = inputHeight;//输入图像高度为inputHeight
	cnn->c1inputWidth = inputWidth;//输入图像宽度为inputWidth
	cnn->c1mapSize = mapSize;//卷积核大小为mapSize

	cnn->c1inChannels = inChannels;//输入图像个数
	cnn->c1outChannels = outChannels;//输出MAP个数

	cnn->c1isFullConnect = true; // 默认为全连接

								// 权重空间的初始化，先行再列调用，[r][c]
	int i, j, c, r;
	srand((unsigned)time(NULL));//随机化初始化种子以便每次初始化得到的随机数都不相同
	cudaMallocManaged(&cnn->c1mapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	cudaMallocManaged(&cnn->c1dmapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	for (i = 0;i<inChannels;i++) {//对一副输入图像
		for (j = 0;j<outChannels;j++) {//对一副输出图像
			for (r = 0;r<mapSize;r++) {
				for (c = 0;c<mapSize;c++) {
					float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2; 
					cnn->c1mapData[i*outChannels * mapSize * mapSize + j*mapSize * mapSize+r*mapSize + c] = randnum*sqrt((float)6.0 / (float)(mapSize*mapSize*(inChannels + outChannels)));
				}
			}
		}
	}
	cudaMallocManaged(&cnn->c1basicData, outChannels* sizeof(float));
	cudaMallocManaged(&cnn->c1dbasicData, outChannels * sizeof(float));
	int outW = inputWidth - mapSize + 1;//输出MAP大小的宽度
	int outH = inputHeight - mapSize + 1;//输出MAP大小的高度
	cudaMallocManaged(&cnn->c1d, outChannels *outH*outW* sizeof(float));
	cudaMallocManaged(&cnn->c1v, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c1y, outChannels *outH*outW * sizeof(float));
}
void initCovLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{

	cnn->c3inputHeight = inputHeight;//输入图像高度为inputHeight
	cnn->c3inputWidth = inputWidth;//输入图像宽度为inputWidth
	cnn->c3mapSize = mapSize;//卷积核大小为mapSize

	cnn->c3inChannels = inChannels;//输入图像个数
	cnn->c3outChannels = outChannels;//输出MAP个数

	cnn->c3isFullConnect = true; // 默认为全连接

								 // 权重空间的初始化，先行再列调用，[r][c]
	int i, j, c, r;
	srand((unsigned)time(NULL));//随机化初始化种子以便每次初始化得到的随机数都不相同
	cudaMallocManaged(&cnn->c3mapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	cudaMallocManaged(&cnn->c3dmapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	for (i = 0;i<inChannels;i++) {//对一副输入图像
		for (j = 0;j<outChannels;j++) {//对一副输出图像
			for (r = 0;r<mapSize;r++) {
				for (c = 0;c<mapSize;c++) {
					float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2;
					cnn->c3mapData[i*outChannels * mapSize * mapSize + j*mapSize * mapSize + r*mapSize + c] = randnum*sqrt((float)6.0 / (float)(mapSize*mapSize*(inChannels + outChannels)));
				}
			}
		}
	}
	cudaMallocManaged(&cnn->c3basicData, outChannels * sizeof(float));
	cudaMallocManaged(&cnn->c3dbasicData, outChannels * sizeof(float));
	int outW = inputWidth - mapSize + 1;//输出MAP大小的宽度
	int outH = inputHeight - mapSize + 1;//输出MAP大小的高度
	cudaMallocManaged(&cnn->c3d, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c3v, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c3y, outChannels *outH*outW * sizeof(float));
}
//初始化池化层,参数为输入图像的大小inputWidth,inputHeight,池化大小mapSize,输入图像个数inChannels,输出图像个数outChannels,池化类型poolType
void initPoolLayer(CNN* cnn,int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	cnn->s2inputHeight = inputHeight;
	cnn->s2inputWidth = inputWidth;
	cnn->s2mapSize = mapSize;
	cnn->s2inChannels = inChannels;
	cnn->s2outChannels = outChannels;
	cnn->s2poolType = poolType;
	cudaMallocManaged(&cnn->s2basicData, outChannels * sizeof(float));
																  //下面定义输出图像大小,如S2层为24/2=12
	int outW = inputWidth / mapSize;
	int outH = inputHeight / mapSize;

	int j, r;
	cudaMallocManaged(&cnn->s2d, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->s2y, outChannels *outH*outW * sizeof(float));
}
void initPoolLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	cnn->s4inputHeight = inputHeight;
	cnn->s4inputWidth = inputWidth;
	cnn->s4mapSize = mapSize;
	cnn->s4inChannels = inChannels;
	cnn->s4outChannels = outChannels;
	cnn->s4poolType = poolType;

	cudaMallocManaged(&cnn->s4basicData, outChannels * sizeof(float));
	//下面定义输出图像大小,如S2层为24/2=12
	int outW = inputWidth / mapSize;
	int outH = inputHeight / mapSize;

	int j, r;
	cudaMallocManaged(&cnn->s4d, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->s4y, outChannels *outH*outW * sizeof(float));
}
//初始化最后一层的参数,把他视为普通神经网络,参数为输入节点数inputNum和输出节点数outputNum
void initOutLayer(CNN* cnn,int inputNum, int outputNum)
{
	cnn->oinputNum = inputNum;
	cnn->ooutputNum = outputNum;

	cudaMallocManaged(&cnn->obasicData, outputNum* sizeof(float));
	cudaMallocManaged(&cnn->odbasicData, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->od, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->ov, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->oy, outputNum * sizeof(float));

	// 权重的初始化
	cudaMallocManaged(&cnn->owData, outputNum *inputNum * sizeof(float));
	cudaMallocManaged(&cnn->odwData, outputNum *inputNum * sizeof(float));
	int i, j;
	//以下初始化权值矩阵
	srand((unsigned)time(NULL));
	for (i = 0;i<outputNum;i++) {
		for (j = 0;j<inputNum;j++) {
			float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2; // 产生一个-1到1的随机数
			cnn->owData[i*outputNum + j] = randnum*sqrt((float)6.0 / (float)(inputNum + outputNum));//公式仍然不清楚为什么这样定义

		}
	}
	cnn->isFullConnect = true;//设置为全连接
}
//到这里,MATLAB版本cnnsetup的内容就全部结束

// 返回向量最大数的序号,注意这里是向量,所以是一维数组
int vecmaxIndex(float* vec, int veclength)
{
	//下面就是一个最简单的在veclength个长度数组中找最大元素的算法,最后返回的是最大的那个序号
	//这个函数的作用是最后比较测试结果和正确的标签结果是不是同一个值,然后计算误差
	int i;
	float maxnum = -1.0;
	int maxIndex = 0;
	for (i = 0;i<veclength;i++) {
		if (maxnum<vec[i]) {
			maxnum = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;//返回相似度最大的那个神经元的序号
}

// 测试cnn函数,这里的参数为:cnn代表训练好的cnn网络,inputData为测试集的原始图像数据,outputData为测试集实际的正确结果,testNum为测试集数量,这里为10000
float cnntest(CNN** cnn, ImgArr inputData, LabelArr outputDat, int testNum, FILE *fp)
{
	Time t;
	//t.start();
	int n = 0, i;
	int incorrectnum = 0;  //错误预测的数目
	int *p;
	cudaMallocManaged(&p,sizeof(int));
	int tab = inputData->c * inputData->r;
	int tab2 = outputDat->l;
	int size = testNum / BATCHSIZE;
	GPUT test;
	for (n = 0;n<size;n++) {
		test.startT();
		testcnn << <50,20>> > (cnn, inputData->ImgPtr + n*BATCHSIZE*tab, outputDat->LabelPtr + n*BATCHSIZE*tab2,p);
			cudaDeviceSynchronize();
		test.endT();
	}
	//t.end();
	return (float)*p / (float)testNum;//返回错误率
}

// 保存cnn
void savecnn(CNN* cnn, const char* filename)//用来把CNN网络中每一层的权值(卷积核)和偏置存储到文件中
{
	FILE  *fp = NULL;
	fp = fopen(filename, "wb");
	if (fp == NULL)
		printf("write file failed\n");
	int i, j, r;
	// C1的数据
	fwrite(cnn->c1mapData, sizeof(float), cnn->c1inChannels * cnn->c1outChannels *cnn->c1mapSize * cnn->c1mapSize, fp);
	fwrite(cnn->c1basicData, sizeof(float), cnn->c1outChannels, fp);
	// C3网络
	fwrite(cnn->c3mapData, sizeof(float), cnn->c3inChannels * cnn->c3outChannels *cnn->c3mapSize * cnn->c3mapSize, fp);
	fwrite(cnn->c3basicData, sizeof(float), cnn->c3outChannels, fp);
	// O5输出层
	fwrite(cnn->owData, sizeof(float), cnn->ooutputNum * cnn->oinputNum, fp);
	fwrite(cnn->obasicData, sizeof(float), cnn->ooutputNum, fp);
	fclose(fp);
}
// 导入cnn的数据
void importcnn(CNN* cnn, const char* filename)//用来从文件中导入每一层的权值(卷积核)和偏置到CNN网络
{
	FILE  *fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("write file failed\n");

	int i, j, c, r;
	// C1的数据
	for (i = 0;i<cnn->c1inChannels;i++)
		for (j = 0;j<cnn->c1outChannels;j++)
			for (r = 0;r<cnn->c1mapSize;r++)
				for (c = 0;c<cnn->c1mapSize;c++) {
					float* in = (float*)malloc(sizeof(float));//分配一个长度为1的数组?为什么这样做
					fread(in, sizeof(float), 1, fp);
					cnn->c1mapData[i * cnn->c1outChannels * cnn->c1mapSize * cnn->c1mapSize +j * cnn->c1mapSize*cnn->c1mapSize+r*cnn->c1mapSize+c] = *in;
				}

	for (i = 0;i<cnn->c1outChannels;i++)
		fread(&cnn->c1basicData[i], sizeof(float), 1, fp);//读取偏置值,一共6个

															// C3网络
	for (i = 0;i<cnn->c3inChannels;i++)
		for (j = 0;j<cnn->c3outChannels;j++)
			for (r = 0;r<cnn->c3mapSize;r++)
				for (c = 0;c<cnn->c3mapSize;c++)
					fread(&cnn->c3mapData[i*cnn->c3outChannels*cnn->c3mapSize*cnn->c3mapSize+j*cnn->c3mapSize*cnn->c3mapSize+r*cnn->c3mapSize+c], sizeof(float), 1, fp);//同上,读取参数值

	for (i = 0;i<cnn->c3outChannels;i++)
		fread(&cnn->c3basicData[i], sizeof(float), 1, fp);//读取偏置值,一共12个

	// O5输出层
	for (i = 0;i<cnn->ooutputNum;i++)
		for (j = 0;j<cnn->oinputNum;j++)
			fread(&cnn->owData[i*cnn->oinputNum + j], sizeof(float), 1, fp);//读取输出层的权值矩阵

	for (i = 0;i<cnn->ooutputNum;i++)
		fread(&cnn->obasicData[i], sizeof(float), 1, fp);//读取输出层的偏置值,一共10个

	fclose(fp);
}
//用来训练CNN的网络,根据传入的原始图像inputData,图像的正确值(标签)outputData,训练的参数opts以及训练集的数量trainNum来训练网络,这里trainNum为55000,inputData为60000幅原始图像,outputData为60000幅标签
void cnntrain(CNN** cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum)
{
	int testtime = 0;
	//这里并没有打乱原始数据,而是顺序训练的,可能是因为打乱的成本太高
	// 学习训练误差曲线,个数为55000个
	cnn[0]->L = (float*)malloc(trainNum * sizeof(float));//第一个cnn来保存学习误差
	int e;
	if (trainNum % BATCHSIZE != 0)
	{
		cout << "对不起,批次数量不能被全样本个数整除,不能进行训练!" << endl;
		exit(-1);
	}
	for (e = 0;e<opts.numepochs;e++) {//训练次数
		float incorrectRatio = 0.0;//错误率,默认为0
		string t;
		int train = trainNum / BATCHSIZE;//批训练次数
		for (int n = 0;n<train;n++) {//批训练
			cnncpy(cnn);//把第一个CNN的信息复制给后面BATCHSIZE-1个
			int bs = n*BATCHSIZE;
			//cout << bs << endl;
			cnntrains << <BATCHSIZE, 1 >> > (cnn, inputData->ImgPtr, outputData->LabelPtr, bs);
			cudaDeviceSynchronize();
			cnnupdategrad(cnn);//批量更新整个网络的梯度
			float l = 0.0;
			int i;
			for (i = 0; i<cnn[0]->ooutputNum; i++)
				l = l + cnn[0]->e[i] * cnn[0]->e[i];//计算均方误差e[i]^2,下面除以2才是真正的均方误差E,e[i] = t[i] - y[i],见cnnbp函数
			if (n == 0)
				cnn[0]->L[n] = l / (float)2.0;//第一次让误差值为l(L)/2
			else
				cnn[0]->L[n] = cnn[0]->L[n - 1] * 0.99 + 0.01*l / (float)2.0;//第二次开始让误差值等于这个函数
			if (n % 20 == 0)
			{
				char* filedir = "E:\\CNNData\\";//先把cnn原来的权值保存到这个目录下
				const char* filename = combine_strings(filedir, combine_strings(intTochar(testtime++), ".cnn"));//文件名字是n.cnn
				savecnn(cnn[0], filename);//把卷积神经网络保存下来
				incorrectRatio = cnntest(cnn, inputData1, outputData1, testNum, fp);//测试CNN网络,输出错误率,用的是第一个CNN网络,后面的和第一个是一样的
				cout << "test" << "error:" << incorrectRatio << endl;
				fprintf(fp, "testerror:%f\n", incorrectRatio);
				cout << "test" << e << "error:" << incorrectRatio << endl;
			}
		}
	}
}

// 这里InputData是图像数据，inputData[r][c],r行c列，这里跟各权重模板是一致的
//注意这里采用的是在线学习,也就是一个图像一个图像的学习,每个图像都会生成一堆权值,然后马上更新
void cnnff(CNN* cnn, float* inputData)
{
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn->s2inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn->s2inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
										// 第一层的传播
	int i, j, r, c, t, k, m, n;
	// 第一层输出数据
	nSize mapSize = { cnn->c1mapSize,cnn->c1mapSize };//卷积核大小,5X5
	nSize inSize = { cnn->c1inputWidth,cnn->c1inputHeight };//输入图像大小,28X28
	nSize outSize = { cnn->s2inputWidth,cnn->s2inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnn->c1outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnn->c1inChannels); j++) {//对C1层的每一个输入MAP,这里为1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->c1mapData[j * cnn->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r+(mapSize.r - 1 - r)*mapSize.r +mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] += tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//截止到这一步，没有任何逻辑问题，但是，S2出现了问题。
	//继续查找S2错误


	// 第二层的输出传播S2，采样层

	outSize.c = cnn->c3inputWidth;//输出图像大小,12X12
	outSize.r = cnn->c3inputHeight;//输出图像大小,12X12
	inSize.c = cnn->s2inputWidth;//输入图像大小,24X24
	inSize.r = cnn->s2inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i < (cnn->s2outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												  //下采样池化
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->c1y[i *inSize.c *inSize.r +m*inSize.c + n];
						//cout << m << " " << n <<" "<< cnn->c1y[i *inSize.c *inSize.r + m*inSize.c + n]<<endl;
					}
				}
				cnn->s2y[i*outSize.c*outSize.r  + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i << " " << t << " " << j << " :" << cnn->s2y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}
	// 第三层输出传播,这里是全连接
	outSize.c = cnn->s4inputWidth;//输出图像大小,8X8
	outSize.r = cnn->s4inputHeight;//输出图像大小,8X8
	inSize.c = cnn->c3inputWidth;//输入图像大小,12X12
	inSize.r = cnn->c3inputHeight;//输入图像大小,12X12
	mapSize.c = cnn->c3mapSize;//卷积核大小,5X5
	mapSize.r = cnn->c3mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnn->c3outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnn->c3inChannels); j++) {//对C3层的每一个输入MAP,这里为6
												   //初始化卷积用数组
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->c3mapData[j *cnn->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c+(mapSize.r - 1 - r)*mapSize.r+mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->s2y[j * is * is + (t + r)* is + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn->c3basicData[i]);//得到C3层最后的输出MAP
				//cout << i << " " << r << " " << c << " " << cnn->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnn->s4inputWidth;//输入图像大小,8X8
	inSize.r = cnn->s4inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnn->s4mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnn->s4mapSize;//输出图像大小,4X4
	for (i = 0; i<(cnn->s4outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn->c3y[i *inSize.c *inSize.r + m*inSize.r + n];
						//
					}
				}
				cnn->s4y[i*outSize.c*outSize.r + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i <<" "<< t << " " << j << " " << cnn->s4y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn->s4outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
				//cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn->oinputNum,cnn->ooutputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
															//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->owData[i*nnSize.c+j];//向量相乘之后相加,然后返回结果
		cnn->ov[i] = o;
	}
	for (i = 0; i<cnn->ooutputNum; i++)//最后用sigmoid函数
		cnn->oy[i] = activation_Sigma(cnn->ov[i], cnn->obasicData[i]);//计算sigmoid函数,即输出层的输出值

}

// sigmoid激活函数 input是数据，inputNum说明数据数目，bas表明偏置
__host__ __device__ float activation_Sigma(float input, float bas) // sigma激活函数
{
	float temp = input + bas;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
}
//求一块矩阵平均值的函数,用于S层池化,参数:output是输出的池化矩阵,outputSize是输出池化矩阵的大小.input是输入矩阵,inputsize是输入矩阵大小,mapSize是池化区域的大小
//如S2层就是输入一个24X24大小的矩阵,然后以2X2大小为一个区域求平均值,最后输出12X12大小的矩阵
void avgPooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize) // 求平均值
{
	int outputW = inputSize.c / mapSize;//输出宽度
	int outputH = inputSize.r / mapSize;//输出高度
	if (outputSize.c != outputW || outputSize.r != outputH)//计算出来的输出大小和给定的输出大小不相同的时候,报错
		printf("ERROR: output size is wrong!!");

	int i, j, m, n;
	//以下计算平均值,加起来求平均,很简单不做解释,注意把int类型的mapsize转化成float来计算
	for (i = 0;i<outputH;i++)
		for (j = 0;j<outputW;j++)
		{
			float sum = 0.0;
			for (m = i*mapSize;m<i*mapSize + mapSize;m++)
				for (n = j*mapSize;n<j*mapSize + mapSize;n++)
					sum = sum + input[m][n];

			output[i][j] = sum / (float)(mapSize*mapSize);
		}
}

// 单层全连接神经网络的前向传播
// 两向量相乘,即各位置对应元素相乘然后求和,注意这里的乘是点乘操作,不是矩阵相乘操作
__host__ __device__ float vecMulti(float* vec1, float* vec2, int vecL)
{
	int i;
	float m = 0;
	for (i = 0;i<vecL;i++)
		m = m + vec1[i] * vec2[i];//向量相乘之后相加,然后返回结果
	return m;
}
//此函数用来定义普通神经网络的前向传播过程,即最后一层的输出map的计算方法,参数说明:把input矩阵的每一行数据和wdata矩阵的每一行数据点乘然后求和,最后得到的结果放入output数组里,nnSize是两个相乘矩阵的大小,要求两个相乘矩阵的大小相同,均为nnSize
void nnff(float* output, float* input, float** wdata, nSize nnSize)
{
	int w = nnSize.c;//宽度,即列数,192
	int h = nnSize.r;//高度,即行数,10

	int i;
	for (i = 0;i<h;i++)//对每一行数据,vecMulti定义把两个矩阵对应位置的192个元素分别相乘然后求和,类似于矩阵运算,再加上一个偏置b就得到了一个神经元的输入z,这里一共有10个神经元
		output[i] = vecMulti(input, wdata[i], w);
}

__host__ __device__ float sigma_derivation(float y) { // Logic激活函数的自变量微分,即sigmoid函数的导数
	return y*(1 - y); // 这里y是指经过激活函数的输出值，而不是自变量
}
// 网络的后向传播
void cnnbp(CNN* cnn, float* outputData)
{
	//nSize outSize,inSize,mapSize;
	//int i, j, c, r,t,k,m,n; // 将误差保存到网络中
	//for (i = 0; i<cnn->ooutputNum; i++)
	//	cnn->e[i] = cnn->oy[i] - outputData[i];//误差是实际输出减去真正正确的输出,对应公式为ai-yi=-(yi-ai),注意这里的y[i]是ai,而yi是outputData[i]
	//											 // 输出层O5的灵敏度
	//for (i = 0; i<cnn->ooutputNum; i++)
	//	cnn->od[i] = cnn->e[i] * sigma_derivation(cnn->oy[i]);//对10个神经元来说,每个神经元的输出层的灵敏度公式为-(yi-ai)(ai*(1-ai)),注意这里的y[i]是ai,而yi是outputData[i]
	//																// S4层，传递到S4层的误差
	//																// 这里没有激活函数
	//outSize.r = cnn->s4inputWidth / cnn->s4mapSize;
	//outSize.c = cnn->s4inputHeight / cnn->s4mapSize;//S4层的输出矩阵大小,这里是4X4
	//for (i = 0; i < cnn->s4outChannels; i++) {//对每一个输出矩阵,都有一个和输出矩阵一样大小的敏感度矩阵与之对应
	//	for (r = 0; r < outSize.r; r++) {
	//		for (c = 0; c < outSize.c; c++) {
	//			for (j = 0; j < cnn->ooutputNum; j++) {//这里对应公式是普通神经网络非输出层的残差计算公式,详解见MATLAB版本各变量说明那篇文章fvd变量的说明
	//				int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt用来定位权值,S4层第i个输出MAP第r行第c列与第j个神经元的权值为[j][i*outSize.c*outSize.r + r*outSize.c + c],因为他是二维行优先存储矩阵,第一维代表了他链接的输出层的第j个神经元,第二维代表的是那条边上的权值
	//				cnn->s4d[i][r][c] = cnn->s4d[i][r][c] + cnn->od[j] * cnn->owData[j][wInt];
	//			}
	//		}
	//	}
	//}
	//int mapdata = cnn->s4mapSize;//这里需要进行上采样操作,因此需要扩充mapSize大小的上采样,这里是2X2
	//nSize S4dSize = { cnn->s4inputWidth / cnn->s4mapSize,cnn->s4inputHeight / cnn->s4mapSize };//S4层的敏感度矩阵大小,这里是4X4,也就是S4层输出矩阵大小
	//float C3e[8][8];
	//for (i = 0; i<cnn->c3outChannels; i++) {//C3层每一个输出MAP都对应一个敏感度矩阵
	//										  //S4dSize12 mapSize2
	//	for (j = 0; j<S4dSize.r*cnn->s4mapSize; j = j + cnn->s4mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
	//		for (t = 0; t<S4dSize.c*cnn->s4mapSize; t = t + cnn->s4mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
	//			for (m = 0; m<cnn->s4mapSize; m++) {//每次对连续的upc个元素赋值
	//				C3e[j][t + m] = cnn->s4d[i][j / cnn->s4mapSize][t / cnn->s4mapSize];//填充行
	//			}
	//		}
	//		for (n = 1; n < cnn->s4mapSize; n++) {     //  高的扩充,第二行到最后一行
	//			for (t = 0; t < S4dSize.c*cnn->s4mapSize; t++) {//列方向切换
	//				C3e[j + n][t] = C3e[j][t];//填充刚才第一行的结果
	//			}
	//		}
	//	}
	//	for (r = 0; r<cnn->s4inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是8
	//		for (c = 0; c<cnn->s4inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是8
	//			cnn->c3d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->c3y[i][r][c]) / (float)(cnn->s4mapSize*cnn->s4mapSize);//注意这里需要除以(float)(cnn->s4mapSize*cnn->s4mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C3层的敏感度矩阵
	//}
	//// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	//// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	//outSize.c = cnn->c3inputWidth;//S2层敏感度矩阵大小为12X12
	//outSize.r = cnn->c3inputHeight;//S2层敏感度矩阵大小为12X12
	//inSize.r = cnn->s4inputWidth;
	//inSize.c = cnn->s4inputHeight;//C3层敏感度矩阵的大小
	//mapSize.r = cnn->c3mapSize;
	//mapSize.c = cnn->c3mapSize;//C3层卷积核大小5X5
	//float corr[12][12];//存储相关计算结果
	//float exData[16][16];//存储full之后的临时变量
	//int addr, addc;

	//addr = addc = mapSize.r - 1;//要扩展的边长
	//for (i = 0; i<cnn->s2outChannels; i++) {//对于S2层每一个输出MAP,6
	//	for (j = 0; j<cnn->c3outChannels; j++) {//对于C3层每一个输出MAP,由于这里是全连接结构,因此S2层的每一副图像与C3层的每一副图像都有关,12
	//											  //float** corr = correlation(cnn->c3mapData[i][j], mapSize, cnn->c3d[j], inSize, full);//这里本来要把C3层对应的卷积核在先旋转180度然后在进行卷积操作,而实际上卷积操作又把卷积核旋转了180度,因此这里直接就不旋转卷积核,而是直接和卷积核相乘,full类型相乘
	//		int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数,12
	//		int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数,12
	//		int newSize = outSizeW - 1 + mapSize.c;//exInputData大小,16
	//											   //扩展矩阵
	//		for (t = 0; t<inSize.r + 2 * addr; t++) {
	//			for (k = 0; k<inSize.c + 2 * addc; k++) {
	//				if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//如果是在新扩充的边缘处,设置为0
	//					exData[t][k] = (float)0.0;
	//				else
	//					exData[t][k] = cnn->c3d[j][t - addr][k - addc]; // 不然,复制原向量的数据
	//			}
	//		}
	//		//卷积操作
	//		for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
	//			for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
	//				corr[t][k] = 0.0;
	//			}
	//		}
	//		for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
	//			for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
	//				for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
	//					for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
	//						corr[t][k] = corr[t][k] + cnn->c3mapData[i][j][r][c] * exData[t + r][k + c];
	//						//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
	//					}
	//				}
	//			}
	//		}
	//		for (t = 0; t<outSize.r; t++) {
	//			for (k = 0; k<outSize.c; k++) {
	//				cnn->s2d[i][t][k] = cnn->s2d[i][t][k] + corr[t][k];//相加然后返回给res
	//			}
	//		}
	//	}
	//}
	//// C1层，卷积层
	//mapdata = cnn->s2mapSize;//C1层灵敏度map的大小,24X24
	//nSize S2dSize = { cnn->s2inputWidth / cnn->s2mapSize,cnn->s2inputHeight / cnn->s2mapSize };//S2层灵敏度MAP的大小,12X12里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	//float C1e[24][24];
	//for (i = 0; i<cnn->c1outChannels; i++) {//C1层每一个输出MAP都对应一个敏感度矩阵
	//	for (j = 0; j<S2dSize.r*cnn->s2mapSize; j = j + cnn->s2mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
	//		for (t = 0; t<S2dSize.c*cnn->s2mapSize; t = t + cnn->s2mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
	//			for (m = 0; m<cnn->s2mapSize; m++) {//每次对连续的upc个元素赋值
	//				C1e[j][t + m] = cnn->s2d[i][j / cnn->s2mapSize][t / cnn->s2mapSize];//填充行
	//			}
	//		}
	//		for (n = 1; n < cnn->s2mapSize; n++) {     //  高的扩充,第二行到最后一行
	//			for (t = 0; t < S2dSize.c*cnn->s2mapSize; t++) {//列方向切换
	//				C1e[j + n][t] = C1e[j][t];//填充刚才第一行的结果
	//			}
	//		}
	//	}
	//	for (r = 0; r<cnn->s2inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是24
	//		for (c = 0; c<cnn->s2inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是24
	//			cnn->c1d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->c1y[i][r][c]) / (float)(cnn->s2mapSize*cnn->s2mapSize);//注意这里需要除以(float)(cnn->s2mapSize*cnn->s2mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C1层的敏感度矩阵
	//}
}

// 更新权重
void cnnapplygrads(CNN* cnn, CNNOpts opts, float* inputData)
{
	//// 这里存在权重的主要是卷积层和输出层
	//// 更新这两个地方的权重就可以了
	//int i, j, r, c,t,k;
	//nSize mapSize;
	//nSize dSize = { cnn->s2inputHeight,cnn->s2inputWidth };//C1层灵敏度矩阵大小,24X24
	//nSize ySize = { cnn->c1inputHeight,cnn->c1inputWidth };//C1层输入矩阵大小,28X28
	//mapSize.r = cnn->c1mapSize;
	//mapSize.c = cnn->c1mapSize;//C1层卷积核大小
	//float cov[24][24];
	////float cmout[5][5];
	//float tins[28][28];
	//float tin[28][28];
	//for (i = 0; i<cnn->c1outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
	//	for (j = 0; j<cnn->c1inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
	//											 //首先,一维转二维计算,旋转180度似乎不对
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tins[r][c] = inputData[r*ySize.c + c];
	//			}
	//		}
	//		//这里之所以会出错,是数组交换最简单的问题,a=b,b=a不能直接写,要用C做中转!!!!
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
	//																   //cout << tin[r][c] << " ";
	//			}
	//			//cout << endl;
	//		}
	//		//system("pause");
	//		//旋转卷积核
	//		for (r = 0; r<dSize.r; r++) {
	//			for (c = 0; c<dSize.c; c++) {
	//				cov[r][c] = cnn->c1d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
	//			}
	//		}

	//		//计算卷积
	//		for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
	//			for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
	//				for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
	//					for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
	//						cnn->c1dmapData[j][i][t][k] = cnn->c1dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
	//						//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
	//					}
	//				}
	//			}
	//		}
	//		for (t = 0; t<mapSize.r; t++)
	//			for (k = 0; k<mapSize.c; k++)
	//				cnn->c1dmapData[j][i][t][k] = cnn->c1dmapData[j][i][t][k] * -1 * 1.0;
	//	}
	//	float sum = 0.0;
	//	for (t = 0; t<dSize.r; t++)
	//		for (j = 0; j<dSize.c; j++)
	//			sum = sum + cnn->c1d[i][t][j];
	//	cnn->c1dbasicData[i] = -1 * 1.0*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	//}
	//// C3层的权重更新
	//dSize.c = cnn->s4inputWidth;//C3层灵敏度矩阵大小,8X8
	//dSize.r = cnn->s4inputHeight;//C3层灵敏度矩阵大小,8X8
	//ySize.c = cnn->c3inputWidth;//C3层输入矩阵大小,12X12
	//ySize.r = cnn->c3inputHeight;//C3层输入矩阵大小,12X12
	//mapSize.c = cnn->c3mapSize;//C3层卷积核大小,5X5
	//mapSize.r = cnn->c3mapSize;//C3层卷积核大小,5X5
	//float cov2[8][8];
	//float tin2[12][12];
	//for (i = 0; i<cnn->c3outChannels; i++) {//对于每一副输出MAP,这里是12,大小8X8
	//	for (j = 0; j<cnn->c3inChannels; j++) {//对于每一副输入图像,这里是8,大小12X12
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tin2[r][c] = cnn->s2y[j][ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
	//			}
	//		}
	//		//旋转卷积核
	//		for (r = 0; r<dSize.r; r++) {
	//			for (c = 0; c<dSize.c; c++) {
	//				cov2[r][c] = cnn->c3d[i][dSize.r - 1 - r][dSize.c - 1 - c];//旋转180度,一目了然
	//			}
	//		}
	//		//计算卷积
	//		for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
	//			for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
	//				for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
	//					for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
	//						cnn->c3dmapData[j][i][t][k] = cnn->c3dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
	//						//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
	//					}
	//				}
	//			}
	//		}
	//		for (t = 0; t<mapSize.r; t++)
	//			for (k = 0; k<mapSize.c; k++)
	//				cnn->c3dmapData[j][i][t][k] = cnn->c3dmapData[j][i][t][k] * -1 * 1.0;
	//	}
	//	float sum = 0.0;
	//	for (t = 0; t<dSize.r; t++)
	//		for (j = 0; j<dSize.c; j++)
	//			sum = sum + cnn->c3d[i][t][j];
	//	cnn->c3dbasicData[i] = -1 * 1.0*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	//}
	//float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	//for (i = 0; i < (cnn->s4outChannels); i++) {//S4层的12个输出矩阵
	//	for (r = 0; r < 4; r++) {//对每一个4X4的MAP
	//		for (c = 0; c < 4; c++) {
	//			O5inData[i*4*4 + r*4 + c] = cnn->s4y[i][r][c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
	//		}
	//	}
	//}
	//// 输出层
	//// 首先需要将前面的多维输出展开成一维向量
	//for (j = 0; j<cnn->ooutputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
	//	for (i = 0; i<cnn->oinputNum; i++)//对192个输入更新梯度
	//		cnn->odwData[j][i] = -1 * 1.0*cnn->od[j] * O5inData[i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
	//	cnn->odbasicData[j] = -1 * 1.0*cnn->od[j];//对b更新梯度,b的梯度就是敏感度delta
	//}
}

void cnnclear(CNN* cnn)
{
	// 将神经元的部分数据清除,主要清楚的是中间保存变量v,每一层的输出y以及敏感误差值d,清空这些值为0.0
	int j, c, r;
	// C1网络
	for (j = 0;j<cnn->c1outChannels;j++) {
		for (r = 0;r<cnn->s2inputHeight;r++) {
			for (c = 0;c<cnn->s2inputWidth;c++) {
				cnn->c1d[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
				cnn->c1v[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
				cnn->c1y[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
			}
		}
	}
	// S2网络
	for (j = 0;j<cnn->s2outChannels;j++) {
		for (r = 0;r<cnn->c3inputHeight;r++) {
			for (c = 0;c<cnn->c3inputWidth;c++) {
				cnn->s2d[j*cnn->c3inputHeight*cnn->c3inputWidth+r*cnn->c3inputWidth+c] = (float)0.0;
				cnn->s2y[j*cnn->c3inputHeight*cnn->c3inputWidth + r*cnn->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3网络
	for (j = 0;j<cnn->c3outChannels;j++) {
		for (r = 0;r<cnn->s4inputHeight;r++) {
			for (c = 0;c<cnn->s4inputWidth;c++) {
				cnn->c3d[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth +c] = (float)0.0;
				cnn->c3v[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth + c] = (float)0.0;
				cnn->c3y[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4网络
	for (j = 0;j<cnn->s4outChannels;j++) {
		for (r = 0;r<cnn->s4inputHeight / cnn->s4mapSize;r++) {
			for (c = 0;c<cnn->s4inputWidth / cnn->s4mapSize;c++) {
				cnn->s4d[j * (cnn->s4inputHeight / cnn->s4mapSize) * (cnn->s4inputWidth / cnn->s4mapSize) + r* (cnn->s4inputWidth / cnn->s4mapSize)+c] = (float)0.0;
				cnn->s4y[j * (cnn->s4inputHeight / cnn->s4mapSize) * (cnn->s4inputWidth / cnn->s4mapSize) + r* (cnn->s4inputWidth / cnn->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5输出
	for (j = 0;j<cnn->ooutputNum;j++) {
		cnn->od[j] = (float)0.0;
		cnn->ov[j] = (float)0.0;
		cnn->oy[j] = (float)0.0;
	}
}

// 这是用于测试的函数,用来以二进制的方式把训练好的CNN网络的所有数据保存到文件中
void savecnndata(CNN* cnn, const char* filename, float** inputdata) // 保存CNN网络中的相关数据
{
	
}
void cnnupdategrad(CNN** cnnarray)
{
	//int i, j;
	//nSize mapSize = { cnnarray[0]->c1mapSize,cnnarray[0]->c1mapSize };//C1层卷积核大小
	//for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//	cnnarray[0]->e[i] *= cnnarray[0]->e[i];//均方误差先求平均再求和
	//for (int s = 1; s < BATCHSIZE; s++)
	//{
	//	//累加误差
	//	for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//		cnnarray[0]->e[i] += cnnarray[s]->e[i] * cnnarray[s]->e[i];
	//	//C1层梯度累加
	//	for (i = 0; i < cnnarray[0]->c1outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
	//		for (j = 0; j < cnnarray[0]->c1inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
	//			addmat(cnnarray[0]->c1dmapData[j][i], cnnarray[0]->c1dmapData[j][i], mapSize, cnnarray[s]->c1dmapData[j][i], mapSize);//累加卷积核梯度
	//		}
	//	}
	//	for (int j = 0; j < cnnarray[0]->c1outChannels; j++) {//对于每一副输出MAP,累加偏置梯度这里是6,大小24X24
	//		cnnarray[0]->c1dbasicData[j] += cnnarray[s]->c1dbasicData[j];
	//	}
	//	//C3层梯度累加
	//	for (i = 0; i < cnnarray[0]->c3outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
	//		for (j = 0; j < cnnarray[0]->c3inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
	//			addmat(cnnarray[0]->c3dmapData[j][i], cnnarray[0]->c3dmapData[j][i], mapSize, cnnarray[s]->c3dmapData[j][i], mapSize);//累加卷积核梯度
	//		}
	//	}
	//	for (int j = 0; j < cnnarray[0]->c3outChannels; j++) {//对于每一副输出MAP,累加偏置梯度这里是6,大小24X24
	//		cnnarray[0]->c3dbasicData[j] += cnnarray[s]->c3dbasicData[j];
	//	}
	//	//输出层梯度累加
	//	for (j = 0; j<cnnarray[0]->ooutputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
	//		for (i = 0; i<cnnarray[0]->oinputNum; i++)//对192个输入更新梯度
	//			cnnarray[0]->odwData[j][i] += cnnarray[s]->odwData[j][i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
	//		cnnarray[0]->odbasicData[j] += cnnarray[s]->odbasicData[j];//对b更新梯度,b的梯度就是敏感度delta
	//	}
	//}
	////以下求权重平均并更新权重
	//for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//	cnnarray[0]->e[i] /= (float)BATCHSIZE;//计算均方误差平均值
	//for (i = 0; i < cnnarray[0]->c1outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
	//	for (j = 0; j < cnnarray[0]->c1inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
	//		multifactor(cnnarray[0]->c1dmapData[j][i], cnnarray[0]->c1dmapData[j][i], mapSize, 1.0 / BATCHSIZE);//卷积核梯度求平均
	//		addmat(cnnarray[0]->c1mapData[j][i], cnnarray[0]->c1mapData[j][i], mapSize, cnnarray[0]->c1dmapData[j][i], mapSize);//更新梯度
	//	}
	//}
	//for (int j = 0; j < cnnarray[0]->c1outChannels; j++) {
	//	cnnarray[0]->c1dbasicData[j] /= (float)BATCHSIZE;//偏置求平均
	//	cnnarray[0]->c1basicData[j] += cnnarray[0]->c1dbasicData[j];
	//}
	////C3层梯度求平均
	//for (i = 0; i < cnnarray[0]->c3outChannels; i++) {//对于每一副输出MAP
	//	for (j = 0; j < cnnarray[0]->c3inChannels; j++) {//对于每一副输入图像
	//		multifactor(cnnarray[0]->c3dmapData[j][i], cnnarray[0]->c3dmapData[j][i], mapSize, 1.0 / (float)BATCHSIZE);//卷积核梯度求平均
	//		addmat(cnnarray[0]->c3mapData[j][i], cnnarray[0]->c3mapData[j][i], mapSize, cnnarray[0]->c3dmapData[j][i], mapSize);//更新梯度
	//	}
	//}
	//for (int j = 0; j < cnnarray[0]->c3outChannels; j++) {
	//	cnnarray[0]->c3dbasicData[j] /= (float)BATCHSIZE;
	//	cnnarray[0]->c3basicData[j] += cnnarray[0]->c3dbasicData[j];
	//}
	////输出层求平均梯度
	//for (j = 0; j<cnnarray[0]->ooutputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
	//	for (i = 0; i < cnnarray[0]->oinputNum; i++)//对192个输入更新梯度
	//	{
	//		cnnarray[0]->odwData[j][i] /= (float)BATCHSIZE;//求平均
	//		cnnarray[0]->owData[j][i] += cnnarray[0]->odwData[j][i];//更新梯度
	//	}
	//	cnnarray[0]->odbasicData[j] /= (float)BATCHSIZE;//求平均
	//	cnnarray[0]->obasicData[j] += cnnarray[0]->odbasicData[j];//更新梯度
	//}

}
void cnncpy(CNN** cnnarray)
{
	start = clock();//开始计时
	for (int k = 1; k < BATCHSIZE; k++)
	{
		int i, j, r, s;
		int t1 = cnnarray[0]->c1inChannels *cnnarray[0]->c1outChannels*cnnarray[0]->c1mapSize*cnnarray[0]->c1mapSize;
		int t2 = cnnarray[0]->c3inChannels *cnnarray[0]->c3outChannels*cnnarray[0]->c3mapSize*cnnarray[0]->c3mapSize;
		// 复制C1的数据inChannels*outChannels*mapSize*mapSize
		for (i = 0; i < t1; i++)
			cnnarray[k]->c1mapData[i] = cnnarray[0]->c1mapData[i];
		for (i = 0; i < cnnarray[0]->c1outChannels; i++)
			cnnarray[k]->c1basicData[i] = cnnarray[0]->c1basicData[i];
		//C3层信息复制
		for (i = 0; i < t2; i++)
				cnnarray[k]->c3mapData[i] = cnnarray[0]->c3mapData[i];
		for (i = 0; i < cnnarray[0]->c3outChannels; i++)
			cnnarray[k]->c3basicData[i] = cnnarray[0]->c3basicData[i];
		//输出层信息复制
		for (i = 0; i<cnnarray[0]->ooutputNum; i++)
			for (j = 0; j < cnnarray[0]->oinputNum; j++)
				cnnarray[k]->owData[i*cnnarray[0]->oinputNum + j] = cnnarray[0]->owData[i*cnnarray[0]->oinputNum + j];
		for (i = 0; i < cnnarray[0]->ooutputNum; i++)
			cnnarray[k]->obasicData[i] = cnnarray[0]->obasicData[i];
	}
	finish = clock();//结束计时,单位毫秒
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//单位换成秒
	printf("copytime:%f seconds\n", duration);
}
__global__ void testcnn(CNN** cnn, float* inputData,float* LabelData,int* wrongnum)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	wrongnum[x] = 0;
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn[x]->s2inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn[x]->s2inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
									  // 第一层的传播
	int i, j, r, c, t, k, m, n;
	for (j = 0;j<cnn[x]->c1outChannels;j++) {
		for (r = 0;r<cnn[x]->s2inputHeight;r++) {
			for (c = 0;c<cnn[x]->s2inputWidth;c++) {
				cnn[x]->c1d[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
				cnn[x]->c1v[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
				cnn[x]->c1y[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
			}
		}
	}
	// S2网络
	for (j = 0;j<cnn[x]->s2outChannels;j++) {
		for (r = 0;r<cnn[x]->c3inputHeight;r++) {
			for (c = 0;c<cnn[x]->c3inputWidth;c++) {
				cnn[x]->s2d[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
				cnn[x]->s2y[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3网络
	for (j = 0;j<cnn[x]->c3outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth;c++) {
				cnn[x]->c3d[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3v[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3y[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4网络
	for (j = 0;j<cnn[x]->s4outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight / cnn[x]->s4mapSize;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth / cnn[x]->s4mapSize;c++) {
				cnn[x]->s4d[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
				cnn[x]->s4y[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5输出
	for (j = 0;j<cnn[x]->ooutputNum;j++) {
		cnn[x]->od[j] = (float)0.0;
		cnn[x]->ov[j] = (float)0.0;
		cnn[x]->oy[j] = (float)0.0;
	}
	// 第一层输出数据
	nSize mapSize = { cnn[x]->c1mapSize,cnn[x]->c1mapSize };//卷积核大小,5X5
	nSize inSize = { cnn[x]->c1inputWidth,cnn[x]->c1inputHeight };//输入图像大小,28X28
	nSize outSize = { cnn[x]->s2inputWidth,cnn[x]->s2inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnn[x]->c1outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnn[x]->c1inChannels); j++) {//对C1层的每一个输入MAP,这里为1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c1mapData[j * cnn[x]->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] += tempconv[r][c] * inputData[x*inSize.r*inSize.c+(t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//截止到这一步，没有任何逻辑问题，但是，S2出现了问题。
	//继续查找S2错误


	// 第二层的输出传播S2，采样层

	outSize.c = cnn[x]->c3inputWidth;//输出图像大小,12X12
	outSize.r = cnn[x]->c3inputHeight;//输出图像大小,12X12
	inSize.c = cnn[x]->s2inputWidth;//输入图像大小,24X24
	inSize.r = cnn[x]->s2inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i < (cnn[x]->s2outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												//下采样池化
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn[x]->c1y[i *inSize.c *inSize.r + m*inSize.c + n];
						//cout << m << " " << n <<" "<< cnn[x]->c1y[i *inSize.c *inSize.r + m*inSize.c + n]<<endl;
					}
				}
				cnn[x]->s2y[i*outSize.c*outSize.r + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i << " " << t << " " << j << " :" << cnn[x]->s2y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}
	// 第三层输出传播,这里是全连接
	outSize.c = cnn[x]->s4inputWidth;//输出图像大小,8X8
	outSize.r = cnn[x]->s4inputHeight;//输出图像大小,8X8
	inSize.c = cnn[x]->c3inputWidth;//输入图像大小,12X12
	inSize.r = cnn[x]->c3inputHeight;//输入图像大小,12X12
	mapSize.c = cnn[x]->c3mapSize;//卷积核大小,5X5
	mapSize.r = cnn[x]->c3mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnn[x]->c3outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnn[x]->c3inChannels); j++) {//对C3层的每一个输入MAP,这里为6
												 //初始化卷积用数组
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c3mapData[j *cnn[x]->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn[x]->s2y[j * is * is + (t + r)* is + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn[x]->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c3basicData[i]);//得到C3层最后的输出MAP
																																								 //cout << i << " " << r << " " << c << " " << cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnn[x]->s4inputWidth;//输入图像大小,8X8
	inSize.r = cnn[x]->s4inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnn[x]->s4mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnn[x]->s4mapSize;//输出图像大小,4X4
	for (i = 0; i<(cnn[x]->s4outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn[x]->c3y[i *inSize.c *inSize.r + m*inSize.r + n];
						//
					}
				}
				cnn[x]->s4y[i*outSize.c*outSize.r + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i <<" "<< t << " " << j << " " << cnn[x]->s4y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn[x]->s4outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn[x]->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
																													  //cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn[x]->oinputNum,cnn[x]->ooutputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
													  //nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn[x]->owData[i*nnSize.c + j];//向量相乘之后相加,然后返回结果
		cnn[x]->ov[i] = o;
	}
	for (i = 0; i<cnn[x]->ooutputNum; i++)//最后用sigmoid函数
		cnn[x]->oy[i] = activation_Sigma(cnn[x]->ov[i], cnn[x]->obasicData[i]);//计算sigmoid函数,即输出层的输出值
	float maxnum = -1.0;
	int maxIndex = 0;
	for (i = 0; i<cnn[x]->ooutputNum; i++) {
		if (maxnum<cnn[x]->oy[i]) {
			maxnum = cnn[x]->oy[i];
			maxIndex = i;
		}
	}
	maxnum = -1.0;
	int maxIndex2 = 0;
	for (i = 0; i<cnn[x]->ooutputNum; i++) {
		if (maxnum<LabelData[x*10+i]) {
			maxnum = LabelData[x * 10 + i];
			maxIndex2 = i;
		}
	}
	//printf("%d %d\n",x, maxIndex);
	if (maxIndex != maxIndex2)
		atomicAdd(wrongnum, 1);
}



__global__ void cnntrains(CNN** cnn, float* IData, float* LData, int bs) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int py = bs + x;
	//if(py>58999)
	//printf("%d %d %d\n",py,bs,x);
	//由于结构体中没有定义当前层输出MAP的大小,因此获得当前层输出MAP的大小只能通过下一层输入MAP的大小来获得
	int outSizeW = cnn[x]->s2inputWidth;//定义第一层的输出MAP矩阵的大小,这里是24X24
	int outSizeH = cnn[x]->s2inputHeight;//定义第一层的输出MAP矩阵的大小,这里是24X24
										 // 第一层的传播
	int i, j, r, c, t, k, m, n;
	for (j = 0;j<cnn[x]->c1outChannels;j++) {
		for (r = 0;r<cnn[x]->s2inputHeight;r++) {
			for (c = 0;c<cnn[x]->s2inputWidth;c++) {
				cnn[x]->c1d[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
				cnn[x]->c1v[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
				cnn[x]->c1y[j*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c] = (float)0.0;
			}
		}
	}
	// S2网络
	for (j = 0;j<cnn[x]->s2outChannels;j++) {
		for (r = 0;r<cnn[x]->c3inputHeight;r++) {
			for (c = 0;c<cnn[x]->c3inputWidth;c++) {
				cnn[x]->s2d[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
				cnn[x]->s2y[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3网络
	for (j = 0;j<cnn[x]->c3outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth;c++) {
				cnn[x]->c3d[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3v[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3y[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4网络
	for (j = 0;j<cnn[x]->s4outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight / cnn[x]->s4mapSize;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth / cnn[x]->s4mapSize;c++) {
				cnn[x]->s4d[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
				cnn[x]->s4y[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5输出
	for (j = 0;j<cnn[x]->ooutputNum;j++) {
		cnn[x]->od[j] = (float)0.0;
		cnn[x]->ov[j] = (float)0.0;
		cnn[x]->oy[j] = (float)0.0;
	}
	// 第一层输出数据
	nSize mapSize = { cnn[x]->c1mapSize,cnn[x]->c1mapSize };//卷积核大小,5X5
	nSize inSize = { cnn[x]->c1inputWidth,cnn[x]->c1inputHeight };//输入图像大小,28X28
	nSize outSize = { cnn[x]->s2inputWidth,cnn[x]->s2inputHeight };//输出图像大小,24X24
	float mapout[24][24];//临时保存卷积结果用的数组
	float tempconv[5][5];//临时用卷积核,旋转之后的
	for (i = 0; i<(cnn[x]->c1outChannels); i++) {//对C1层的每一个输出MAP,这里为6
		for (j = 0; j<(cnn[x]->c1inChannels); j++) {//对C1层的每一个输入MAP,这里为1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c1mapData[j * cnn[x]->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout[t][k] += tempconv[r][c] * IData[x*inSize.r*inSize.c + (t + r) * inSize.r + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//相加然后返回给res
				}
			}
		}
		//当一个输出MAP卷积完所有的输入图像之后,就可以进行sigmoid函数的计算了,下面两行用来把得到的输出MAP的每一个值计算sigmoid,如C3层就是把8X8大小的矩阵用sigmoid函数计算,得到8X8大小的最终输出MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//截止到这一步，没有任何逻辑问题，但是，S2出现了问题。
	//继续查找S2错误


	// 第二层的输出传播S2，采样层

	outSize.c = cnn[x]->c3inputWidth;//输出图像大小,12X12
	outSize.r = cnn[x]->c3inputHeight;//输出图像大小,12X12
	inSize.c = cnn[x]->s2inputWidth;//输入图像大小,24X24
	inSize.r = cnn[x]->s2inputHeight;//输入图像大小,24X24
	int mSize = 2;//以2为大小池化
	for (i = 0; i < (cnn[x]->s2outChannels); i++) {//对6幅输出图像,每一副都由C1层进行池化
												   //下采样池化
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn[x]->c1y[i *inSize.c *inSize.r + m*inSize.c + n];
						//cout << m << " " << n <<" "<< cnn[x]->c1y[i *inSize.c *inSize.r + m*inSize.c + n]<<endl;
					}
				}
				cnn[x]->s2y[i*outSize.c*outSize.r + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i << " " << t << " " << j << " :" << cnn[x]->s2y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}
	// 第三层输出传播,这里是全连接
	outSize.c = cnn[x]->s4inputWidth;//输出图像大小,8X8
	outSize.r = cnn[x]->s4inputHeight;//输出图像大小,8X8
	inSize.c = cnn[x]->c3inputWidth;//输入图像大小,12X12
	inSize.r = cnn[x]->c3inputHeight;//输入图像大小,12X12
	mapSize.c = cnn[x]->c3mapSize;//卷积核大小,5X5
	mapSize.r = cnn[x]->c3mapSize;//卷积核大小,5X5
	float mapout2[8][8];//临时保存卷积结果用的数组
	for (i = 0; i<(cnn[x]->c3outChannels); i++) {//对C3层的每一个输出MAP,这里为12
		for (j = 0; j<(cnn[x]->c3inChannels); j++) {//对C3层的每一个输入MAP,这里为6
													//初始化卷积用数组
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c3mapData[j *cnn[x]->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn[x]->s2y[j * is * is + (t + r)* is + k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn[x]->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//相加然后返回给res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c3basicData[i]);//得到C3层最后的输出MAP
																																										  //cout << i << " " << r << " " << c << " " << cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// 第四层的输出传播
	inSize.c = cnn[x]->s4inputWidth;//输入图像大小,8X8
	inSize.r = cnn[x]->s4inputHeight;//输入图像大小,8X8
	outSize.c = inSize.c / cnn[x]->s4mapSize;//输出图像大小,4X4
	outSize.r = inSize.r / cnn[x]->s4mapSize;//输出图像大小,4X4
	for (i = 0; i<(cnn[x]->s4outChannels); i++) {
		for (t = 0; t < outSize.c; t++)
		{
			for (j = 0; j < outSize.r; j++)
			{
				float sum = 0.0;
				for (m = t * mSize; m < t * mSize + mSize; m++) {
					for (n = j * mSize; n < j * mSize + mSize; n++) {
						sum = sum + cnn[x]->c3y[i *inSize.c *inSize.r + m*inSize.r + n];
						//
					}
				}
				cnn[x]->s4y[i*outSize.c*outSize.r + t*outSize.r + j] = sum / (float)(mSize*mSize);
				//cout << i <<" "<< t << " " << j << " " << cnn[x]->s4y[i*outSize.c*outSize.r + t*outSize.r + j] << endl;
			}
		}
	}

	// 输出层O5的处理
	// 首先需要将前面的多维输出展开成一维向量
	float O5inData[192]; //分配长度为192个数组来把S4层的输出矩阵导入
	for (i = 0; i < (cnn[x]->s4outChannels); i++) {//S4层的12个输出矩阵
		for (r = 0; r < outSize.r; r++) {//对每一个4X4的MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn[x]->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//输入数据是一个长度为192的一维矩阵,其中S4层第i个输出MAP的第r行第c列的数据的存储位置为i*outSize.r*outSize.c+r*outSize.c+c,这里是行优先存储,注意
																														 //cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn[x]->oinputNum,cnn[x]->ooutputNum };//定义一个矩阵大小为10(高度,行数)X192(宽度,列数)
															//nnSize.c=192,nnSize.r=10,代表192X10的全连接网络
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn[x]->owData[i*nnSize.c + j];//向量相乘之后相加,然后返回结果
		cnn[x]->ov[i] = o;
	}
	for (i = 0; i<cnn[x]->ooutputNum; i++)//最后用sigmoid函数
		cnn[x]->oy[i] = activation_Sigma(cnn[x]->ov[i], cnn[x]->obasicData[i]);//计算sigmoid函数,即输出层的输出值
	for (i = 0; i<cnn[x]->ooutputNum; i++)
		cnn[x]->e[i] = cnn[x]->oy[i] - LData[py*10+i];//误差是实际输出减去真正正确的输出,对应公式为ai-yi=-(yi-ai),注意这里的y[i]是ai,而yi是outputData[i]
																 // 输出层O5的灵敏度
	for (i = 0; i<cnn[x]->ooutputNum; i++)
		cnn[x]->od[i] = cnn[x]->e[i] * sigma_derivation(cnn[x]->oy[i]);//对10个神经元来说,每个神经元的输出层的灵敏度公式为-(yi-ai)(ai*(1-ai)),注意这里的y[i]是ai,而yi是outputData[i]
																			 // S4层，传递到S4层的误差
																			 // 这里没有激活函数
	outSize.r = cnn[x]->s4inputWidth / cnn[x]->s4mapSize;
	outSize.c = cnn[x]->s4inputHeight / cnn[x]->s4mapSize;//S4层的输出矩阵大小,这里是4X4
	for (i = 0; i < cnn[x]->s4outChannels; i++) {//对每一个输出矩阵,都有一个和输出矩阵一样大小的敏感度矩阵与之对应
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn[x]->ooutputNum; j++) {//这里对应公式是普通神经网络非输出层的残差计算公式,详解见MATLAB版本各变量说明那篇文章fvd变量的说明
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt用来定位权值,S4层第i个输出MAP第r行第c列与第j个神经元的权值为[j][i*outSize.c*outSize.r + r*outSize.c + c],因为他是二维行优先存储矩阵,第一维代表了他链接的输出层的第j个神经元,第二维代表的是那条边上的权值
					cnn[x]->s4d[i*outSize.r*outSize.r + r* outSize.r + c] = cnn[x]->s4d[i*outSize.r*outSize.r + r* outSize.r + c] + cnn[x]->od[j] * cnn[x]->owData[j * cnn[x]->ooutputNum +wInt];
				}
			}
		}
	}
	int mapdata = cnn[x]->s4mapSize;//这里需要进行上采样操作,因此需要扩充mapSize大小的上采样,这里是2X2
	nSize S4dSize = { cnn[x]->s4inputWidth / cnn[x]->s4mapSize,cnn[x]->s4inputHeight / cnn[x]->s4mapSize };//S4层的敏感度矩阵大小,这里是4X4,也就是S4层输出矩阵大小
	float C3e[8][8];
	for (i = 0; i<cnn[x]->c3outChannels; i++) {//C3层每一个输出MAP都对应一个敏感度矩阵
												 //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn[x]->s4mapSize; j = j + cnn[x]->s4mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S4dSize.c*cnn[x]->s4mapSize; t = t + cnn[x]->s4mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn[x]->s4mapSize; m++) {//每次对连续的upc个元素赋值
					C3e[j][t + m] = cnn[x]->s4d[i * S4dSize.r*cnn[x]->s4mapSize + j * S4dSize.r / cnn[x]->s4mapSize + t / cnn[x]->s4mapSize];//填充行
				}
			}
			for (n = 1; n < cnn[x]->s4mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S4dSize.c*cnn[x]->s4mapSize; t++) {//列方向切换
					C3e[j + n][t] = C3e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn[x]->s4inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是8
			for (c = 0; c<cnn[x]->s4inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是8
				cnn[x]->c3d[i * cnn[x]->s4inputHeight * cnn[x]->s4inputWidth +r*cnn[x]->s4inputWidth + c] = C3e[r][c] * sigma_derivation(cnn[x]->c3y[i * cnn[x]->s4inputHeight * cnn[x]->s4inputWidth + r*cnn[x]->s4inputWidth + c]) / (float)(cnn[x]->s4mapSize*cnn[x]->s4mapSize);//注意这里需要除以(float)(cnn[x]->s4mapSize*cnn[x]->s4mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C3层的敏感度矩阵
	}
	// S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
	// 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
	outSize.c = cnn[x]->c3inputWidth;//S2层敏感度矩阵大小为12X12
	outSize.r = cnn[x]->c3inputHeight;//S2层敏感度矩阵大小为12X12
	inSize.r = cnn[x]->s4inputWidth;
	inSize.c = cnn[x]->s4inputHeight;//C3层敏感度矩阵的大小
	mapSize.r = cnn[x]->c3mapSize;
	mapSize.c = cnn[x]->c3mapSize;//C3层卷积核大小5X5
	float corr[12][12];//存储相关计算结果
	float exData[16][16];//存储full之后的临时变量
	int addr, addc;

	addr = addc = mapSize.r - 1;//要扩展的边长
	for (i = 0; i<cnn[x]->s2outChannels; i++) {//对于S2层每一个输出MAP,6
		for (j = 0; j<cnn[x]->c3outChannels; j++) {//对于C3层每一个输出MAP,由于这里是全连接结构,因此S2层的每一副图像与C3层的每一副图像都有关,12
													 //float** corr = correlation(cnn[x]->c3mapData[i][j], mapSize, cnn[x]->c3d[j], inSize, full);//这里本来要把C3层对应的卷积核在先旋转180度然后在进行卷积操作,而实际上卷积操作又把卷积核旋转了180度,因此这里直接就不旋转卷积核,而是直接和卷积核相乘,full类型相乘
			int outSizeW = inSize.c + (mapSize.c - 1); // 这里的输出扩大一部分,完全卷积得到的卷积MAP的宽度/列数,12
			int outSizeH = inSize.r + (mapSize.r - 1);// 这里的输出扩大一部分,完全卷积得到的卷积MAP的高度/行数,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData大小,16
												   //扩展矩阵
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//如果是在新扩充的边缘处,设置为0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn[x]->c3d[j * (inSize.r + 2 * addr) * (inSize.r + 2 * addr) + (t - addr)*(inSize.r + 2 * addr) + k - addc]; // 不然,复制原向量的数据
				}
			}
			//卷积操作
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<outSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<mapSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<mapSize.c; c++) {//对于卷积核的每一列
							corr[t][k] = corr[t][k] + cnn[x]->c3mapData[i*cnn[x]->c3outChannels *mapSize.r * mapSize.c  + j *mapSize.r * mapSize.c + r * mapSize.r + c] * exData[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->s2d[i*outSize.r*outSize.r + t*outSize.r  + k] = cnn[x]->s2d[i*outSize.r*outSize.r + t*outSize.r + k] + corr[t][k];//相加然后返回给res
				}
			}
		}
	}
	// C1层，卷积层
	mapdata = cnn[x]->s2mapSize;//C1层灵敏度map的大小,24X24
	nSize S2dSize = { cnn[x]->s2inputWidth / cnn[x]->s2mapSize,cnn[x]->s2inputHeight / cnn[x]->s2mapSize };//S2层灵敏度MAP的大小,12X12里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
	float C1e[24][24];
	for (i = 0; i<cnn[x]->c1outChannels; i++) {//C1层每一个输出MAP都对应一个敏感度矩阵
		for (j = 0; j<S2dSize.r*cnn[x]->s2mapSize; j = j + cnn[x]->s2mapSize) {//在行方向上,每次填充upr个相同的值,注意这里是高度,这里一个j就是原始map一行的数据,一次for循环执行完,整个一行的数据就扩充完了
			for (t = 0; t<S2dSize.c*cnn[x]->s2mapSize; t = t + cnn[x]->s2mapSize) {// 宽的扩充,即x方向上每隔upc个值改变一次赋值
				for (m = 0; m<cnn[x]->s2mapSize; m++) {//每次对连续的upc个元素赋值
					C1e[j][t + m] = cnn[x]->s2d[i * S2dSize.r*S2dSize.r  + j*S2dSize.r / cnn[x]->s2mapSize+t / cnn[x]->s2mapSize];//填充行
				}
			}
			for (n = 1; n < cnn[x]->s2mapSize; n++) {     //  高的扩充,第二行到最后一行
				for (t = 0; t < S2dSize.c*cnn[x]->s2mapSize; t++) {//列方向切换
					C1e[j + n][t] = C1e[j][t];//填充刚才第一行的结果
				}
			}
		}
		for (r = 0; r<cnn[x]->s2inputHeight; r++)//对每一个敏感度矩阵的行,注意这里大小是24
			for (c = 0; c<cnn[x]->s2inputWidth; c++)//对每一个敏感度矩阵的列,注意这里大小是24
				cnn[x]->c1d[i*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth+r*cnn[x]->s2inputWidth+c] = C1e[r][c] * sigma_derivation(cnn[x]->c1y[i*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c]) / (float)(cnn[x]->s2mapSize*cnn[x]->s2mapSize);//注意这里需要除以(float)(cnn[x]->s2mapSize*cnn[x]->s2mapSize),即除以4,以便把原来的敏感度矩阵平均分配给C1层的敏感度矩阵
	}

	//apply
	// C1层的权重更新
	nSize dSize = { cnn[x]->s2inputHeight,cnn[x]->s2inputWidth };//C1层灵敏度矩阵大小,24X24
	nSize ySize = { cnn[x]->c1inputHeight,cnn[x]->c1inputWidth };//C1层输入矩阵大小,28X28
	mapSize.r = cnn[x]->c1mapSize;
	mapSize.c = cnn[x]->c1mapSize;//C1层卷积核大小
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnn[x]->c1outChannels; i++) {//对于每一副输出MAP,这里是6,大小24X24
		for (j = 0; j<cnn[x]->c1inChannels; j++) {//对于每一副输入图像,这里是1,大小28X28
													//首先,一维转二维计算,旋转180度似乎不对
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = IData[x*inSize.r*inSize.c + (t + r) * inSize.r*ySize.c + c];
				}
			}
			//这里之所以会出错,是数组交换最简单的问题,a=b,b=a不能直接写,要用C做中转!!!!
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//旋转180度,一目了然
																	   //cout << tin[r][c] << " ";
				}
				//cout << endl;
			}
			//system("pause");
			//旋转卷积核
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov[r][c] = cnn[x]->c1d[i*ySize.r * ySize.c +ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//旋转180度,一目了然
				}
			}

			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r+k] = cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] + cov[r][c] * tin[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] = cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] * -1 * 1.0;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnn[x]->c1d[i*dSize.r*dSize.c + t*dSize.r +j];
		cnn[x]->c1dbasicData[i] = -1 * 1.0*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// C3层的权重更新
	dSize.c = cnn[x]->s4inputWidth;//C3层灵敏度矩阵大小,8X8
	dSize.r = cnn[x]->s4inputHeight;//C3层灵敏度矩阵大小,8X8
	ySize.c = cnn[x]->c3inputWidth;//C3层输入矩阵大小,12X12
	ySize.r = cnn[x]->c3inputHeight;//C3层输入矩阵大小,12X12
	mapSize.c = cnn[x]->c3mapSize;//C3层卷积核大小,5X5
	mapSize.r = cnn[x]->c3mapSize;//C3层卷积核大小,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnn[x]->c3outChannels; i++) {//对于每一副输出MAP,这里是12,大小8X8
		for (j = 0; j<cnn[x]->c3inChannels; j++) {//对于每一副输入图像,这里是8,大小12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnn[x]->s2y[j*ySize.r * ySize.c + ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//旋转卷积核
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnn[x]->c3d[i*ySize.r * ySize.c + ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//旋转180度,一目了然
				}
			}
			//计算卷积
			for (t = 0; t<mapSize.r; t++) {//对于输出MAP的每一行
				for (k = 0; k<mapSize.c; k++) {//对于输出MAP的每一列
					for (r = 0; r<dSize.r; r++) {//对于卷积核的每一行
						for (c = 0; c<dSize.c; c++) {//对于卷积核的每一列
							cnn[x]->c3dmapData[j*cnn[x]->c3outChannels*mapSize.r*mapSize.r +i*mapSize.r*mapSize.r+t*mapSize.r+k] = cnn[x]->c3dmapData[j*cnn[x]->c3outChannels*mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData的第j行第i列的值,等于卷积核第r行第c列的值乘以扩充之后原始图像的第j+r行第i+c列的结果的总和,即完成了卷积操作
						}
					}
				}
			}
			for (t = 0; t<mapSize.r; t++)
				for (k = 0; k<mapSize.c; k++)
					cnn[x]->c3dmapData[j * cnn[x]->c3outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] = cnn[x]->c3dmapData[j * cnn[x]->c3outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] * -1 * 1.0;
		}
		float sum = 0.0;
		for (t = 0; t<dSize.r; t++)
			for (j = 0; j<dSize.c; j++)
				sum = sum + cnn[x]->c3d[i*dSize.r * dSize.c + t*dSize.c+j];
		cnn[x]->c3dbasicData[i] = -1 * 1.0*sum;//更新偏置b的梯度,偏置b的梯度就是每一副输出MAP[i]对应敏感度矩阵的各元素之和
	}
	// 输出层
	// 首先需要将前面的多维输出展开成一维向量
	for (j = 0; j<cnn[x]->ooutputNum; j++) {//对于输出层每一个输出神经元,即10个神经元
		for (i = 0; i<cnn[x]->oinputNum; i++)//对192个输入更新梯度
			cnn[x]->odwData[j * 10+i] = -1 * 1.0*cnn[x]->od[j] * O5inData[i];//对W的梯度求法,即aj*delta,然后乘学习率以更新梯度
		cnn[x]->odbasicData[j] = -1 * 1.0*cnn[x]->od[j];//对b更新梯度,b的梯度就是敏感度delta
	}
}