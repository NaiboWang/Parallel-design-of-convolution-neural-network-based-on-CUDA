#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include "cnn.h"
#include<iostream>
using namespace std;
clock_t start, finish;//����ʱ���õ�
double  duration;

void cnnsetup(CNN** cnn, nSize inputSize, int outputSize, FILE *fp)
{
	start = clock();//��ʼ��ʱ
	for (int i = 0;i < BATCHSIZE;i++)//��ʼ��BATCHSIZE��cnn[i]����
	{
		cnn[i]->layerNum = 5;//����cnn[i][i]����Ϊ5
		nSize inSize;//����ͼ���С
		int mapSize = 5;//�������˴�СΪ5
		inSize.c = inputSize.c;//����ͼ���СΪ28X28
		inSize.r = inputSize.r;//����ͼ���СΪ28X28
		initCovLayer(cnn[i], inSize.c, inSize.r, 5, 1, 6);//������ͼ���СΪ28X28,����˴�СΪ5X5,����ͼ����Ϊ1,���MAP��Ϊ6��ʼ��C1��,�����ʼ�����̼�initCovLayer��������
		inSize.c = inSize.c - mapSize + 1;//S2�������MAP�Ĵ�СΪ28-5+1=24,��24X24
		inSize.r = inSize.r - mapSize + 1;//S2�������MAP�Ĵ�СΪ28-5+1=24,��24X24
		initPoolLayer(cnn[i], inSize.c, inSize.r, 2, 6, 6, AvePool); //������ͼ���СΪ24X24, �ػ���СΪ2X2, ����ͼ����Ϊ6, ���MAP��Ϊ6,�ػ�����Ϊƽ���ػ���ʼ��S2��, �����ʼ�����̼�initPoolLayer��������
		inSize.c = inSize.c / 2;//C3�������ͼ���СΪ24/2=12,��12X12
		inSize.r = inSize.r / 2;//C3�������ͼ���СΪ24/2=12,��12X12
		initCovLayer2(cnn[i], inSize.c, inSize.r, 5, 6, 12);//������ͼ���СΪ12X12,����˴�СΪ5X5,����ͼ����Ϊ6,���MAP��Ϊ12��ʼ��C3��,�����ʼ�����̼�initCovLayer��������
		inSize.c = inSize.c - mapSize + 1;//S4������ͼ���СΪ12-5+1=8,��8X8
		inSize.r = inSize.r - mapSize + 1;//S4������ͼ���СΪ12-5+1=8,��8X8
		initPoolLayer2(cnn[i], inSize.c, inSize.r, 2, 12, 12, AvePool);//������ͼ���СΪ8X8, �ػ���СΪ2X2, ����ͼ����Ϊ12, ���MAP��Ϊ12,�ػ�����Ϊƽ���ػ���ʼ��S4��, �����ʼ�����̼�initPoolLayer��������
		inSize.c = inSize.c / 2;//ȫ�������������ͼ���СΪ8/2=4,��4X4
		inSize.r = inSize.r / 2;//ȫ�������������ͼ���СΪ8/2=4,��4X4
		initOutLayer(cnn[i], inSize.c*inSize.r * 12, outputSize);//������ͼ���СΪ4*4*12=192,���ͼ��Ϊ10��ʼ�������,�����ʼ�����̼�initOutLayer��������
		cudaMallocManaged(&cnn[i]->e, cnn[i]->ooutputNum*sizeof(float));
		for (int j = 0;j < cnn[i]->ooutputNum;j++)
			cnn[i]->e[i] = 0.0;
	}
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("setuptime:%f seconds\n", duration);
	fprintf(fp, "setuptime:%f seconds\n", duration);
}
//��ʼ�������,����Ϊ����ͼ��Ĵ�СinputWidth,inputHeight,����˴�СmapSize,����ͼ�����inChannels,���ͼ�����outChannels
void initCovLayer(CNN* cnn,int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{

	cnn->c1inputHeight = inputHeight;//����ͼ��߶�ΪinputHeight
	cnn->c1inputWidth = inputWidth;//����ͼ����ΪinputWidth
	cnn->c1mapSize = mapSize;//����˴�СΪmapSize

	cnn->c1inChannels = inChannels;//����ͼ�����
	cnn->c1outChannels = outChannels;//���MAP����

	cnn->c1isFullConnect = true; // Ĭ��Ϊȫ����

								// Ȩ�ؿռ�ĳ�ʼ�����������е��ã�[r][c]
	int i, j, c, r;
	srand((unsigned)time(NULL));//�������ʼ�������Ա�ÿ�γ�ʼ���õ��������������ͬ
	cudaMallocManaged(&cnn->c1mapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	cudaMallocManaged(&cnn->c1dmapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	for (i = 0;i<inChannels;i++) {//��һ������ͼ��
		for (j = 0;j<outChannels;j++) {//��һ�����ͼ��
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
	int outW = inputWidth - mapSize + 1;//���MAP��С�Ŀ��
	int outH = inputHeight - mapSize + 1;//���MAP��С�ĸ߶�
	cudaMallocManaged(&cnn->c1d, outChannels *outH*outW* sizeof(float));
	cudaMallocManaged(&cnn->c1v, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c1y, outChannels *outH*outW * sizeof(float));
}
void initCovLayer2(CNN* cnn, int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{

	cnn->c3inputHeight = inputHeight;//����ͼ��߶�ΪinputHeight
	cnn->c3inputWidth = inputWidth;//����ͼ����ΪinputWidth
	cnn->c3mapSize = mapSize;//����˴�СΪmapSize

	cnn->c3inChannels = inChannels;//����ͼ�����
	cnn->c3outChannels = outChannels;//���MAP����

	cnn->c3isFullConnect = true; // Ĭ��Ϊȫ����

								 // Ȩ�ؿռ�ĳ�ʼ�����������е��ã�[r][c]
	int i, j, c, r;
	srand((unsigned)time(NULL));//�������ʼ�������Ա�ÿ�γ�ʼ���õ��������������ͬ
	cudaMallocManaged(&cnn->c3mapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	cudaMallocManaged(&cnn->c3dmapData, inChannels*outChannels*mapSize*mapSize * sizeof(float));
	for (i = 0;i<inChannels;i++) {//��һ������ͼ��
		for (j = 0;j<outChannels;j++) {//��һ�����ͼ��
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
	int outW = inputWidth - mapSize + 1;//���MAP��С�Ŀ��
	int outH = inputHeight - mapSize + 1;//���MAP��С�ĸ߶�
	cudaMallocManaged(&cnn->c3d, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c3v, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->c3y, outChannels *outH*outW * sizeof(float));
}
//��ʼ���ػ���,����Ϊ����ͼ��Ĵ�СinputWidth,inputHeight,�ػ���СmapSize,����ͼ�����inChannels,���ͼ�����outChannels,�ػ�����poolType
void initPoolLayer(CNN* cnn,int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
	cnn->s2inputHeight = inputHeight;
	cnn->s2inputWidth = inputWidth;
	cnn->s2mapSize = mapSize;
	cnn->s2inChannels = inChannels;
	cnn->s2outChannels = outChannels;
	cnn->s2poolType = poolType;
	cudaMallocManaged(&cnn->s2basicData, outChannels * sizeof(float));
																  //���涨�����ͼ���С,��S2��Ϊ24/2=12
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
	//���涨�����ͼ���С,��S2��Ϊ24/2=12
	int outW = inputWidth / mapSize;
	int outH = inputHeight / mapSize;

	int j, r;
	cudaMallocManaged(&cnn->s4d, outChannels *outH*outW * sizeof(float));
	cudaMallocManaged(&cnn->s4y, outChannels *outH*outW * sizeof(float));
}
//��ʼ�����һ��Ĳ���,������Ϊ��ͨ������,����Ϊ����ڵ���inputNum������ڵ���outputNum
void initOutLayer(CNN* cnn,int inputNum, int outputNum)
{
	cnn->oinputNum = inputNum;
	cnn->ooutputNum = outputNum;

	cudaMallocManaged(&cnn->obasicData, outputNum* sizeof(float));
	cudaMallocManaged(&cnn->odbasicData, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->od, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->ov, outputNum * sizeof(float));
	cudaMallocManaged(&cnn->oy, outputNum * sizeof(float));

	// Ȩ�صĳ�ʼ��
	cudaMallocManaged(&cnn->owData, outputNum *inputNum * sizeof(float));
	cudaMallocManaged(&cnn->odwData, outputNum *inputNum * sizeof(float));
	int i, j;
	//���³�ʼ��Ȩֵ����
	srand((unsigned)time(NULL));
	for (i = 0;i<outputNum;i++) {
		for (j = 0;j<inputNum;j++) {
			float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2; // ����һ��-1��1�������
			cnn->owData[i*outputNum + j] = randnum*sqrt((float)6.0 / (float)(inputNum + outputNum));//��ʽ��Ȼ�����Ϊʲô��������

		}
	}
	cnn->isFullConnect = true;//����Ϊȫ����
}
//������,MATLAB�汾cnnsetup�����ݾ�ȫ������

// ������������������,ע������������,������һά����
int vecmaxIndex(float* vec, int veclength)
{
	//�������һ����򵥵���veclength�����������������Ԫ�ص��㷨,��󷵻ص��������Ǹ����
	//������������������Ƚϲ��Խ������ȷ�ı�ǩ����ǲ���ͬһ��ֵ,Ȼ��������
	int i;
	float maxnum = -1.0;
	int maxIndex = 0;
	for (i = 0;i<veclength;i++) {
		if (maxnum<vec[i]) {
			maxnum = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;//�������ƶ������Ǹ���Ԫ�����
}

// ����cnn����,����Ĳ���Ϊ:cnn����ѵ���õ�cnn����,inputDataΪ���Լ���ԭʼͼ������,outputDataΪ���Լ�ʵ�ʵ���ȷ���,testNumΪ���Լ�����,����Ϊ10000
float cnntest(CNN** cnn, ImgArr inputData, LabelArr outputDat, int testNum, FILE *fp)
{
	Time t;
	//t.start();
	int n = 0, i;
	int incorrectnum = 0;  //����Ԥ�����Ŀ
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
	return (float)*p / (float)testNum;//���ش�����
}

// ����cnn
void savecnn(CNN* cnn, const char* filename)//������CNN������ÿһ���Ȩֵ(�����)��ƫ�ô洢���ļ���
{
	FILE  *fp = NULL;
	fp = fopen(filename, "wb");
	if (fp == NULL)
		printf("write file failed\n");
	int i, j, r;
	// C1������
	fwrite(cnn->c1mapData, sizeof(float), cnn->c1inChannels * cnn->c1outChannels *cnn->c1mapSize * cnn->c1mapSize, fp);
	fwrite(cnn->c1basicData, sizeof(float), cnn->c1outChannels, fp);
	// C3����
	fwrite(cnn->c3mapData, sizeof(float), cnn->c3inChannels * cnn->c3outChannels *cnn->c3mapSize * cnn->c3mapSize, fp);
	fwrite(cnn->c3basicData, sizeof(float), cnn->c3outChannels, fp);
	// O5�����
	fwrite(cnn->owData, sizeof(float), cnn->ooutputNum * cnn->oinputNum, fp);
	fwrite(cnn->obasicData, sizeof(float), cnn->ooutputNum, fp);
	fclose(fp);
}
// ����cnn������
void importcnn(CNN* cnn, const char* filename)//�������ļ��е���ÿһ���Ȩֵ(�����)��ƫ�õ�CNN����
{
	FILE  *fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("write file failed\n");

	int i, j, c, r;
	// C1������
	for (i = 0;i<cnn->c1inChannels;i++)
		for (j = 0;j<cnn->c1outChannels;j++)
			for (r = 0;r<cnn->c1mapSize;r++)
				for (c = 0;c<cnn->c1mapSize;c++) {
					float* in = (float*)malloc(sizeof(float));//����һ������Ϊ1������?Ϊʲô������
					fread(in, sizeof(float), 1, fp);
					cnn->c1mapData[i * cnn->c1outChannels * cnn->c1mapSize * cnn->c1mapSize +j * cnn->c1mapSize*cnn->c1mapSize+r*cnn->c1mapSize+c] = *in;
				}

	for (i = 0;i<cnn->c1outChannels;i++)
		fread(&cnn->c1basicData[i], sizeof(float), 1, fp);//��ȡƫ��ֵ,һ��6��

															// C3����
	for (i = 0;i<cnn->c3inChannels;i++)
		for (j = 0;j<cnn->c3outChannels;j++)
			for (r = 0;r<cnn->c3mapSize;r++)
				for (c = 0;c<cnn->c3mapSize;c++)
					fread(&cnn->c3mapData[i*cnn->c3outChannels*cnn->c3mapSize*cnn->c3mapSize+j*cnn->c3mapSize*cnn->c3mapSize+r*cnn->c3mapSize+c], sizeof(float), 1, fp);//ͬ��,��ȡ����ֵ

	for (i = 0;i<cnn->c3outChannels;i++)
		fread(&cnn->c3basicData[i], sizeof(float), 1, fp);//��ȡƫ��ֵ,һ��12��

	// O5�����
	for (i = 0;i<cnn->ooutputNum;i++)
		for (j = 0;j<cnn->oinputNum;j++)
			fread(&cnn->owData[i*cnn->oinputNum + j], sizeof(float), 1, fp);//��ȡ������Ȩֵ����

	for (i = 0;i<cnn->ooutputNum;i++)
		fread(&cnn->obasicData[i], sizeof(float), 1, fp);//��ȡ������ƫ��ֵ,һ��10��

	fclose(fp);
}
//����ѵ��CNN������,���ݴ����ԭʼͼ��inputData,ͼ�����ȷֵ(��ǩ)outputData,ѵ���Ĳ���opts�Լ�ѵ����������trainNum��ѵ������,����trainNumΪ55000,inputDataΪ60000��ԭʼͼ��,outputDataΪ60000����ǩ
void cnntrain(CNN** cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum, FILE *fp, ImgArr inputData1, LabelArr outputData1, int testNum)
{
	int testtime = 0;
	//���ﲢû�д���ԭʼ����,����˳��ѵ����,��������Ϊ���ҵĳɱ�̫��
	// ѧϰѵ���������,����Ϊ55000��
	cnn[0]->L = (float*)malloc(trainNum * sizeof(float));//��һ��cnn������ѧϰ���
	int e;
	if (trainNum % BATCHSIZE != 0)
	{
		cout << "�Բ���,�����������ܱ�ȫ������������,���ܽ���ѵ��!" << endl;
		exit(-1);
	}
	for (e = 0;e<opts.numepochs;e++) {//ѵ������
		float incorrectRatio = 0.0;//������,Ĭ��Ϊ0
		string t;
		int train = trainNum / BATCHSIZE;//��ѵ������
		for (int n = 0;n<train;n++) {//��ѵ��
			cnncpy(cnn);//�ѵ�һ��CNN����Ϣ���Ƹ�����BATCHSIZE-1��
			int bs = n*BATCHSIZE;
			//cout << bs << endl;
			cnntrains << <BATCHSIZE, 1 >> > (cnn, inputData->ImgPtr, outputData->LabelPtr, bs);
			cudaDeviceSynchronize();
			cnnupdategrad(cnn);//������������������ݶ�
			float l = 0.0;
			int i;
			for (i = 0; i<cnn[0]->ooutputNum; i++)
				l = l + cnn[0]->e[i] * cnn[0]->e[i];//����������e[i]^2,�������2���������ľ������E,e[i] = t[i] - y[i],��cnnbp����
			if (n == 0)
				cnn[0]->L[n] = l / (float)2.0;//��һ�������ֵΪl(L)/2
			else
				cnn[0]->L[n] = cnn[0]->L[n - 1] * 0.99 + 0.01*l / (float)2.0;//�ڶ��ο�ʼ�����ֵ�����������
			if (n % 20 == 0)
			{
				char* filedir = "E:\\CNNData\\";//�Ȱ�cnnԭ����Ȩֵ���浽���Ŀ¼��
				const char* filename = combine_strings(filedir, combine_strings(intTochar(testtime++), ".cnn"));//�ļ�������n.cnn
				savecnn(cnn[0], filename);//�Ѿ�������籣������
				incorrectRatio = cnntest(cnn, inputData1, outputData1, testNum, fp);//����CNN����,���������,�õ��ǵ�һ��CNN����,����ĺ͵�һ����һ����
				cout << "test" << "error:" << incorrectRatio << endl;
				fprintf(fp, "testerror:%f\n", incorrectRatio);
				cout << "test" << e << "error:" << incorrectRatio << endl;
			}
		}
	}
}

// ����InputData��ͼ�����ݣ�inputData[r][c],r��c�У��������Ȩ��ģ����һ�µ�
//ע��������õ�������ѧϰ,Ҳ����һ��ͼ��һ��ͼ���ѧϰ,ÿ��ͼ�񶼻�����һ��Ȩֵ,Ȼ�����ϸ���
void cnnff(CNN* cnn, float* inputData)
{
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn->s2inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn->s2inputHeight;//�����һ������MAP����Ĵ�С,������24X24
										// ��һ��Ĵ���
	int i, j, r, c, t, k, m, n;
	// ��һ���������
	nSize mapSize = { cnn->c1mapSize,cnn->c1mapSize };//����˴�С,5X5
	nSize inSize = { cnn->c1inputWidth,cnn->c1inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnn->s2inputWidth,cnn->s2inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnn->c1outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnn->c1inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn->c1mapData[j * cnn->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r+(mapSize.r - 1 - r)*mapSize.r +mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] += tempconv[r][c] * inputData[(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//��ֹ����һ����û���κ��߼����⣬���ǣ�S2���������⡣
	//��������S2����


	// �ڶ�����������S2��������

	outSize.c = cnn->c3inputWidth;//���ͼ���С,12X12
	outSize.r = cnn->c3inputHeight;//���ͼ���С,12X12
	inSize.c = cnn->s2inputWidth;//����ͼ���С,24X24
	inSize.r = cnn->s2inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i < (cnn->s2outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												  //�²����ػ�
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
	// �������������,������ȫ����
	outSize.c = cnn->s4inputWidth;//���ͼ���С,8X8
	outSize.r = cnn->s4inputHeight;//���ͼ���С,8X8
	inSize.c = cnn->c3inputWidth;//����ͼ���С,12X12
	inSize.r = cnn->c3inputHeight;//����ͼ���С,12X12
	mapSize.c = cnn->c3mapSize;//����˴�С,5X5
	mapSize.r = cnn->c3mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnn->c3outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnn->c3inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
												   //��ʼ�����������
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn->c3mapData[j *cnn->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c+(mapSize.r - 1 - r)*mapSize.r+mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn->s2y[j * is * is + (t + r)* is + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn->c3basicData[i]);//�õ�C3���������MAP
				//cout << i << " " << r << " " << c << " " << cnn->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnn->s4inputWidth;//����ͼ���С,8X8
	inSize.r = cnn->s4inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnn->s4mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnn->s4mapSize;//���ͼ���С,4X4
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

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn->s4outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
				//cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn->oinputNum,cnn->ooutputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
															//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn->owData[i*nnSize.c+j];//�������֮�����,Ȼ�󷵻ؽ��
		cnn->ov[i] = o;
	}
	for (i = 0; i<cnn->ooutputNum; i++)//�����sigmoid����
		cnn->oy[i] = activation_Sigma(cnn->ov[i], cnn->obasicData[i]);//����sigmoid����,�����������ֵ

}

// sigmoid����� input�����ݣ�inputNum˵��������Ŀ��bas����ƫ��
__host__ __device__ float activation_Sigma(float input, float bas) // sigma�����
{
	float temp = input + bas;
	return (float)1.0 / ((float)(1.0 + exp(-temp)));
}
//��һ�����ƽ��ֵ�ĺ���,����S��ػ�,����:output������ĳػ�����,outputSize������ػ�����Ĵ�С.input���������,inputsize����������С,mapSize�ǳػ�����Ĵ�С
//��S2���������һ��24X24��С�ľ���,Ȼ����2X2��СΪһ��������ƽ��ֵ,������12X12��С�ľ���
void avgPooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize) // ��ƽ��ֵ
{
	int outputW = inputSize.c / mapSize;//������
	int outputH = inputSize.r / mapSize;//����߶�
	if (outputSize.c != outputW || outputSize.r != outputH)//��������������С�͸����������С����ͬ��ʱ��,����
		printf("ERROR: output size is wrong!!");

	int i, j, m, n;
	//���¼���ƽ��ֵ,��������ƽ��,�ܼ򵥲�������,ע���int���͵�mapsizeת����float������
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

// ����ȫ�����������ǰ�򴫲�
// ���������,����λ�ö�ӦԪ�����Ȼ�����,ע������ĳ��ǵ�˲���,���Ǿ�����˲���
__host__ __device__ float vecMulti(float* vec1, float* vec2, int vecL)
{
	int i;
	float m = 0;
	for (i = 0;i<vecL;i++)
		m = m + vec1[i] * vec2[i];//�������֮�����,Ȼ�󷵻ؽ��
	return m;
}
//�˺�������������ͨ�������ǰ�򴫲�����,�����һ������map�ļ��㷽��,����˵��:��input�����ÿһ�����ݺ�wdata�����ÿһ�����ݵ��Ȼ�����,���õ��Ľ������output������,nnSize��������˾���Ĵ�С,Ҫ��������˾���Ĵ�С��ͬ,��ΪnnSize
void nnff(float* output, float* input, float** wdata, nSize nnSize)
{
	int w = nnSize.c;//���,������,192
	int h = nnSize.r;//�߶�,������,10

	int i;
	for (i = 0;i<h;i++)//��ÿһ������,vecMulti��������������Ӧλ�õ�192��Ԫ�طֱ����Ȼ�����,�����ھ�������,�ټ���һ��ƫ��b�͵õ���һ����Ԫ������z,����һ����10����Ԫ
		output[i] = vecMulti(input, wdata[i], w);
}

__host__ __device__ float sigma_derivation(float y) { // Logic��������Ա���΢��,��sigmoid�����ĵ���
	return y*(1 - y); // ����y��ָ��������������ֵ���������Ա���
}
// ����ĺ��򴫲�
void cnnbp(CNN* cnn, float* outputData)
{
	//nSize outSize,inSize,mapSize;
	//int i, j, c, r,t,k,m,n; // �����浽������
	//for (i = 0; i<cnn->ooutputNum; i++)
	//	cnn->e[i] = cnn->oy[i] - outputData[i];//�����ʵ�������ȥ������ȷ�����,��Ӧ��ʽΪai-yi=-(yi-ai),ע�������y[i]��ai,��yi��outputData[i]
	//											 // �����O5��������
	//for (i = 0; i<cnn->ooutputNum; i++)
	//	cnn->od[i] = cnn->e[i] * sigma_derivation(cnn->oy[i]);//��10����Ԫ��˵,ÿ����Ԫ�������������ȹ�ʽΪ-(yi-ai)(ai*(1-ai)),ע�������y[i]��ai,��yi��outputData[i]
	//																// S4�㣬���ݵ�S4������
	//																// ����û�м����
	//outSize.r = cnn->s4inputWidth / cnn->s4mapSize;
	//outSize.c = cnn->s4inputHeight / cnn->s4mapSize;//S4�����������С,������4X4
	//for (i = 0; i < cnn->s4outChannels; i++) {//��ÿһ���������,����һ�����������һ����С�����жȾ�����֮��Ӧ
	//	for (r = 0; r < outSize.r; r++) {
	//		for (c = 0; c < outSize.c; c++) {
	//			for (j = 0; j < cnn->ooutputNum; j++) {//�����Ӧ��ʽ����ͨ������������Ĳв���㹫ʽ,����MATLAB�汾������˵����ƪ����fvd������˵��
	//				int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt������λȨֵ,S4���i�����MAP��r�е�c�����j����Ԫ��ȨֵΪ[j][i*outSize.c*outSize.r + r*outSize.c + c],��Ϊ���Ƕ�ά�����ȴ洢����,��һά�����������ӵ������ĵ�j����Ԫ,�ڶ�ά��������������ϵ�Ȩֵ
	//				cnn->s4d[i][r][c] = cnn->s4d[i][r][c] + cnn->od[j] * cnn->owData[j][wInt];
	//			}
	//		}
	//	}
	//}
	//int mapdata = cnn->s4mapSize;//������Ҫ�����ϲ�������,�����Ҫ����mapSize��С���ϲ���,������2X2
	//nSize S4dSize = { cnn->s4inputWidth / cnn->s4mapSize,cnn->s4inputHeight / cnn->s4mapSize };//S4������жȾ����С,������4X4,Ҳ����S4����������С
	//float C3e[8][8];
	//for (i = 0; i<cnn->c3outChannels; i++) {//C3��ÿһ�����MAP����Ӧһ�����жȾ���
	//										  //S4dSize12 mapSize2
	//	for (j = 0; j<S4dSize.r*cnn->s4mapSize; j = j + cnn->s4mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
	//		for (t = 0; t<S4dSize.c*cnn->s4mapSize; t = t + cnn->s4mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
	//			for (m = 0; m<cnn->s4mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
	//				C3e[j][t + m] = cnn->s4d[i][j / cnn->s4mapSize][t / cnn->s4mapSize];//�����
	//			}
	//		}
	//		for (n = 1; n < cnn->s4mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
	//			for (t = 0; t < S4dSize.c*cnn->s4mapSize; t++) {//�з����л�
	//				C3e[j + n][t] = C3e[j][t];//���ղŵ�һ�еĽ��
	//			}
	//		}
	//	}
	//	for (r = 0; r<cnn->s4inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��8
	//		for (c = 0; c<cnn->s4inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��8
	//			cnn->c3d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->c3y[i][r][c]) / (float)(cnn->s4mapSize*cnn->s4mapSize);//ע��������Ҫ����(float)(cnn->s4mapSize*cnn->s4mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C3������жȾ���
	//}
	//// S2�㣬S2��û�м����������ֻ�о�����м��������
	//// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��
	//outSize.c = cnn->c3inputWidth;//S2�����жȾ����СΪ12X12
	//outSize.r = cnn->c3inputHeight;//S2�����жȾ����СΪ12X12
	//inSize.r = cnn->s4inputWidth;
	//inSize.c = cnn->s4inputHeight;//C3�����жȾ���Ĵ�С
	//mapSize.r = cnn->c3mapSize;
	//mapSize.c = cnn->c3mapSize;//C3�����˴�С5X5
	//float corr[12][12];//�洢��ؼ�����
	//float exData[16][16];//�洢full֮�����ʱ����
	//int addr, addc;

	//addr = addc = mapSize.r - 1;//Ҫ��չ�ı߳�
	//for (i = 0; i<cnn->s2outChannels; i++) {//����S2��ÿһ�����MAP,6
	//	for (j = 0; j<cnn->c3outChannels; j++) {//����C3��ÿһ�����MAP,����������ȫ���ӽṹ,���S2���ÿһ��ͼ����C3���ÿһ��ͼ���й�,12
	//											  //float** corr = correlation(cnn->c3mapData[i][j], mapSize, cnn->c3d[j], inSize, full);//���ﱾ��Ҫ��C3���Ӧ�ľ����������ת180��Ȼ���ڽ��о������,��ʵ���Ͼ�������ְѾ������ת��180��,�������ֱ�ӾͲ���ת�����,����ֱ�Ӻ;�������,full�������
	//		int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����,12
	//		int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����,12
	//		int newSize = outSizeW - 1 + mapSize.c;//exInputData��С,16
	//											   //��չ����
	//		for (t = 0; t<inSize.r + 2 * addr; t++) {
	//			for (k = 0; k<inSize.c + 2 * addc; k++) {
	//				if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//�������������ı�Ե��,����Ϊ0
	//					exData[t][k] = (float)0.0;
	//				else
	//					exData[t][k] = cnn->c3d[j][t - addr][k - addc]; // ��Ȼ,����ԭ����������
	//			}
	//		}
	//		//�������
	//		for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
	//			for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
	//				corr[t][k] = 0.0;
	//			}
	//		}
	//		for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
	//			for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
	//				for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
	//					for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
	//						corr[t][k] = corr[t][k] + cnn->c3mapData[i][j][r][c] * exData[t + r][k + c];
	//						//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
	//					}
	//				}
	//			}
	//		}
	//		for (t = 0; t<outSize.r; t++) {
	//			for (k = 0; k<outSize.c; k++) {
	//				cnn->s2d[i][t][k] = cnn->s2d[i][t][k] + corr[t][k];//���Ȼ�󷵻ظ�res
	//			}
	//		}
	//	}
	//}
	//// C1�㣬�����
	//mapdata = cnn->s2mapSize;//C1��������map�Ĵ�С,24X24
	//nSize S2dSize = { cnn->s2inputWidth / cnn->s2mapSize,cnn->s2inputHeight / cnn->s2mapSize };//S2��������MAP�Ĵ�С,12X12���Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	//float C1e[24][24];
	//for (i = 0; i<cnn->c1outChannels; i++) {//C1��ÿһ�����MAP����Ӧһ�����жȾ���
	//	for (j = 0; j<S2dSize.r*cnn->s2mapSize; j = j + cnn->s2mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
	//		for (t = 0; t<S2dSize.c*cnn->s2mapSize; t = t + cnn->s2mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
	//			for (m = 0; m<cnn->s2mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
	//				C1e[j][t + m] = cnn->s2d[i][j / cnn->s2mapSize][t / cnn->s2mapSize];//�����
	//			}
	//		}
	//		for (n = 1; n < cnn->s2mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
	//			for (t = 0; t < S2dSize.c*cnn->s2mapSize; t++) {//�з����л�
	//				C1e[j + n][t] = C1e[j][t];//���ղŵ�һ�еĽ��
	//			}
	//		}
	//	}
	//	for (r = 0; r<cnn->s2inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��24
	//		for (c = 0; c<cnn->s2inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��24
	//			cnn->c1d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->c1y[i][r][c]) / (float)(cnn->s2mapSize*cnn->s2mapSize);//ע��������Ҫ����(float)(cnn->s2mapSize*cnn->s2mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C1������жȾ���
	//}
}

// ����Ȩ��
void cnnapplygrads(CNN* cnn, CNNOpts opts, float* inputData)
{
	//// �������Ȩ�ص���Ҫ�Ǿ����������
	//// �����������ط���Ȩ�ؾͿ�����
	//int i, j, r, c,t,k;
	//nSize mapSize;
	//nSize dSize = { cnn->s2inputHeight,cnn->s2inputWidth };//C1�������Ⱦ����С,24X24
	//nSize ySize = { cnn->c1inputHeight,cnn->c1inputWidth };//C1����������С,28X28
	//mapSize.r = cnn->c1mapSize;
	//mapSize.c = cnn->c1mapSize;//C1�����˴�С
	//float cov[24][24];
	////float cmout[5][5];
	//float tins[28][28];
	//float tin[28][28];
	//for (i = 0; i<cnn->c1outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
	//	for (j = 0; j<cnn->c1inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
	//											 //����,һάת��ά����,��ת180���ƺ�����
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tins[r][c] = inputData[r*ySize.c + c];
	//			}
	//		}
	//		//����֮���Ի����,�����齻����򵥵�����,a=b,b=a����ֱ��д,Ҫ��C����ת!!!!
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
	//																   //cout << tin[r][c] << " ";
	//			}
	//			//cout << endl;
	//		}
	//		//system("pause");
	//		//��ת�����
	//		for (r = 0; r<dSize.r; r++) {
	//			for (c = 0; c<dSize.c; c++) {
	//				cov[r][c] = cnn->c1d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
	//			}
	//		}

	//		//������
	//		for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
	//			for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
	//				for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
	//					for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
	//						cnn->c1dmapData[j][i][t][k] = cnn->c1dmapData[j][i][t][k] + cov[r][c] * tin[t + r][k + c];
	//						//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
	//	cnn->c1dbasicData[i] = -1 * 1.0*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	//}
	//// C3���Ȩ�ظ���
	//dSize.c = cnn->s4inputWidth;//C3�������Ⱦ����С,8X8
	//dSize.r = cnn->s4inputHeight;//C3�������Ⱦ����С,8X8
	//ySize.c = cnn->c3inputWidth;//C3����������С,12X12
	//ySize.r = cnn->c3inputHeight;//C3����������С,12X12
	//mapSize.c = cnn->c3mapSize;//C3�����˴�С,5X5
	//mapSize.r = cnn->c3mapSize;//C3�����˴�С,5X5
	//float cov2[8][8];
	//float tin2[12][12];
	//for (i = 0; i<cnn->c3outChannels; i++) {//����ÿһ�����MAP,������12,��С8X8
	//	for (j = 0; j<cnn->c3inChannels; j++) {//����ÿһ������ͼ��,������8,��С12X12
	//		for (r = 0; r<ySize.r; r++) {
	//			for (c = 0; c<ySize.c; c++) {
	//				tin2[r][c] = cnn->s2y[j][ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
	//			}
	//		}
	//		//��ת�����
	//		for (r = 0; r<dSize.r; r++) {
	//			for (c = 0; c<dSize.c; c++) {
	//				cov2[r][c] = cnn->c3d[i][dSize.r - 1 - r][dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
	//			}
	//		}
	//		//������
	//		for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
	//			for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
	//				for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
	//					for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
	//						cnn->c3dmapData[j][i][t][k] = cnn->c3dmapData[j][i][t][k] + cov2[r][c] * tin2[t + r][k + c];
	//						//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
	//	cnn->c3dbasicData[i] = -1 * 1.0*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	//}
	//float O5inData[192]; //���䳤��Ϊ192����������S4������������
	//for (i = 0; i < (cnn->s4outChannels); i++) {//S4���12���������
	//	for (r = 0; r < 4; r++) {//��ÿһ��4X4��MAP
	//		for (c = 0; c < 4; c++) {
	//			O5inData[i*4*4 + r*4 + c] = cnn->s4y[i][r][c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
	//		}
	//	}
	//}
	//// �����
	//// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	//for (j = 0; j<cnn->ooutputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
	//	for (i = 0; i<cnn->oinputNum; i++)//��192����������ݶ�
	//		cnn->odwData[j][i] = -1 * 1.0*cnn->od[j] * O5inData[i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
	//	cnn->odbasicData[j] = -1 * 1.0*cnn->od[j];//��b�����ݶ�,b���ݶȾ������ж�delta
	//}
}

void cnnclear(CNN* cnn)
{
	// ����Ԫ�Ĳ����������,��Ҫ��������м䱣�����v,ÿһ������y�Լ��������ֵd,�����ЩֵΪ0.0
	int j, c, r;
	// C1����
	for (j = 0;j<cnn->c1outChannels;j++) {
		for (r = 0;r<cnn->s2inputHeight;r++) {
			for (c = 0;c<cnn->s2inputWidth;c++) {
				cnn->c1d[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
				cnn->c1v[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
				cnn->c1y[j*cnn->s2inputHeight*cnn->s2inputWidth + r*cnn->s2inputWidth + c] = (float)0.0;
			}
		}
	}
	// S2����
	for (j = 0;j<cnn->s2outChannels;j++) {
		for (r = 0;r<cnn->c3inputHeight;r++) {
			for (c = 0;c<cnn->c3inputWidth;c++) {
				cnn->s2d[j*cnn->c3inputHeight*cnn->c3inputWidth+r*cnn->c3inputWidth+c] = (float)0.0;
				cnn->s2y[j*cnn->c3inputHeight*cnn->c3inputWidth + r*cnn->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3����
	for (j = 0;j<cnn->c3outChannels;j++) {
		for (r = 0;r<cnn->s4inputHeight;r++) {
			for (c = 0;c<cnn->s4inputWidth;c++) {
				cnn->c3d[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth +c] = (float)0.0;
				cnn->c3v[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth + c] = (float)0.0;
				cnn->c3y[j*cnn->s4inputHeight*cnn->s4inputWidth + r * cnn->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4����
	for (j = 0;j<cnn->s4outChannels;j++) {
		for (r = 0;r<cnn->s4inputHeight / cnn->s4mapSize;r++) {
			for (c = 0;c<cnn->s4inputWidth / cnn->s4mapSize;c++) {
				cnn->s4d[j * (cnn->s4inputHeight / cnn->s4mapSize) * (cnn->s4inputWidth / cnn->s4mapSize) + r* (cnn->s4inputWidth / cnn->s4mapSize)+c] = (float)0.0;
				cnn->s4y[j * (cnn->s4inputHeight / cnn->s4mapSize) * (cnn->s4inputWidth / cnn->s4mapSize) + r* (cnn->s4inputWidth / cnn->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5���
	for (j = 0;j<cnn->ooutputNum;j++) {
		cnn->od[j] = (float)0.0;
		cnn->ov[j] = (float)0.0;
		cnn->oy[j] = (float)0.0;
	}
}

// �������ڲ��Եĺ���,�����Զ����Ƶķ�ʽ��ѵ���õ�CNN������������ݱ��浽�ļ���
void savecnndata(CNN* cnn, const char* filename, float** inputdata) // ����CNN�����е��������
{
	
}
void cnnupdategrad(CNN** cnnarray)
{
	//int i, j;
	//nSize mapSize = { cnnarray[0]->c1mapSize,cnnarray[0]->c1mapSize };//C1�����˴�С
	//for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//	cnnarray[0]->e[i] *= cnnarray[0]->e[i];//�����������ƽ�������
	//for (int s = 1; s < BATCHSIZE; s++)
	//{
	//	//�ۼ����
	//	for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//		cnnarray[0]->e[i] += cnnarray[s]->e[i] * cnnarray[s]->e[i];
	//	//C1���ݶ��ۼ�
	//	for (i = 0; i < cnnarray[0]->c1outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
	//		for (j = 0; j < cnnarray[0]->c1inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
	//			addmat(cnnarray[0]->c1dmapData[j][i], cnnarray[0]->c1dmapData[j][i], mapSize, cnnarray[s]->c1dmapData[j][i], mapSize);//�ۼӾ�����ݶ�
	//		}
	//	}
	//	for (int j = 0; j < cnnarray[0]->c1outChannels; j++) {//����ÿһ�����MAP,�ۼ�ƫ���ݶ�������6,��С24X24
	//		cnnarray[0]->c1dbasicData[j] += cnnarray[s]->c1dbasicData[j];
	//	}
	//	//C3���ݶ��ۼ�
	//	for (i = 0; i < cnnarray[0]->c3outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
	//		for (j = 0; j < cnnarray[0]->c3inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
	//			addmat(cnnarray[0]->c3dmapData[j][i], cnnarray[0]->c3dmapData[j][i], mapSize, cnnarray[s]->c3dmapData[j][i], mapSize);//�ۼӾ�����ݶ�
	//		}
	//	}
	//	for (int j = 0; j < cnnarray[0]->c3outChannels; j++) {//����ÿһ�����MAP,�ۼ�ƫ���ݶ�������6,��С24X24
	//		cnnarray[0]->c3dbasicData[j] += cnnarray[s]->c3dbasicData[j];
	//	}
	//	//������ݶ��ۼ�
	//	for (j = 0; j<cnnarray[0]->ooutputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
	//		for (i = 0; i<cnnarray[0]->oinputNum; i++)//��192����������ݶ�
	//			cnnarray[0]->odwData[j][i] += cnnarray[s]->odwData[j][i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
	//		cnnarray[0]->odbasicData[j] += cnnarray[s]->odbasicData[j];//��b�����ݶ�,b���ݶȾ������ж�delta
	//	}
	//}
	////������Ȩ��ƽ��������Ȩ��
	//for (i = 0; i < cnnarray[0]->ooutputNum; i++)
	//	cnnarray[0]->e[i] /= (float)BATCHSIZE;//����������ƽ��ֵ
	//for (i = 0; i < cnnarray[0]->c1outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
	//	for (j = 0; j < cnnarray[0]->c1inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
	//		multifactor(cnnarray[0]->c1dmapData[j][i], cnnarray[0]->c1dmapData[j][i], mapSize, 1.0 / BATCHSIZE);//������ݶ���ƽ��
	//		addmat(cnnarray[0]->c1mapData[j][i], cnnarray[0]->c1mapData[j][i], mapSize, cnnarray[0]->c1dmapData[j][i], mapSize);//�����ݶ�
	//	}
	//}
	//for (int j = 0; j < cnnarray[0]->c1outChannels; j++) {
	//	cnnarray[0]->c1dbasicData[j] /= (float)BATCHSIZE;//ƫ����ƽ��
	//	cnnarray[0]->c1basicData[j] += cnnarray[0]->c1dbasicData[j];
	//}
	////C3���ݶ���ƽ��
	//for (i = 0; i < cnnarray[0]->c3outChannels; i++) {//����ÿһ�����MAP
	//	for (j = 0; j < cnnarray[0]->c3inChannels; j++) {//����ÿһ������ͼ��
	//		multifactor(cnnarray[0]->c3dmapData[j][i], cnnarray[0]->c3dmapData[j][i], mapSize, 1.0 / (float)BATCHSIZE);//������ݶ���ƽ��
	//		addmat(cnnarray[0]->c3mapData[j][i], cnnarray[0]->c3mapData[j][i], mapSize, cnnarray[0]->c3dmapData[j][i], mapSize);//�����ݶ�
	//	}
	//}
	//for (int j = 0; j < cnnarray[0]->c3outChannels; j++) {
	//	cnnarray[0]->c3dbasicData[j] /= (float)BATCHSIZE;
	//	cnnarray[0]->c3basicData[j] += cnnarray[0]->c3dbasicData[j];
	//}
	////�������ƽ���ݶ�
	//for (j = 0; j<cnnarray[0]->ooutputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
	//	for (i = 0; i < cnnarray[0]->oinputNum; i++)//��192����������ݶ�
	//	{
	//		cnnarray[0]->odwData[j][i] /= (float)BATCHSIZE;//��ƽ��
	//		cnnarray[0]->owData[j][i] += cnnarray[0]->odwData[j][i];//�����ݶ�
	//	}
	//	cnnarray[0]->odbasicData[j] /= (float)BATCHSIZE;//��ƽ��
	//	cnnarray[0]->obasicData[j] += cnnarray[0]->odbasicData[j];//�����ݶ�
	//}

}
void cnncpy(CNN** cnnarray)
{
	start = clock();//��ʼ��ʱ
	for (int k = 1; k < BATCHSIZE; k++)
	{
		int i, j, r, s;
		int t1 = cnnarray[0]->c1inChannels *cnnarray[0]->c1outChannels*cnnarray[0]->c1mapSize*cnnarray[0]->c1mapSize;
		int t2 = cnnarray[0]->c3inChannels *cnnarray[0]->c3outChannels*cnnarray[0]->c3mapSize*cnnarray[0]->c3mapSize;
		// ����C1������inChannels*outChannels*mapSize*mapSize
		for (i = 0; i < t1; i++)
			cnnarray[k]->c1mapData[i] = cnnarray[0]->c1mapData[i];
		for (i = 0; i < cnnarray[0]->c1outChannels; i++)
			cnnarray[k]->c1basicData[i] = cnnarray[0]->c1basicData[i];
		//C3����Ϣ����
		for (i = 0; i < t2; i++)
				cnnarray[k]->c3mapData[i] = cnnarray[0]->c3mapData[i];
		for (i = 0; i < cnnarray[0]->c3outChannels; i++)
			cnnarray[k]->c3basicData[i] = cnnarray[0]->c3basicData[i];
		//�������Ϣ����
		for (i = 0; i<cnnarray[0]->ooutputNum; i++)
			for (j = 0; j < cnnarray[0]->oinputNum; j++)
				cnnarray[k]->owData[i*cnnarray[0]->oinputNum + j] = cnnarray[0]->owData[i*cnnarray[0]->oinputNum + j];
		for (i = 0; i < cnnarray[0]->ooutputNum; i++)
			cnnarray[k]->obasicData[i] = cnnarray[0]->obasicData[i];
	}
	finish = clock();//������ʱ,��λ����
	duration = (double)(finish - start) / CLOCKS_PER_SEC;//��λ������
	printf("copytime:%f seconds\n", duration);
}
__global__ void testcnn(CNN** cnn, float* inputData,float* LabelData,int* wrongnum)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	wrongnum[x] = 0;
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn[x]->s2inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn[x]->s2inputHeight;//�����һ������MAP����Ĵ�С,������24X24
									  // ��һ��Ĵ���
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
	// S2����
	for (j = 0;j<cnn[x]->s2outChannels;j++) {
		for (r = 0;r<cnn[x]->c3inputHeight;r++) {
			for (c = 0;c<cnn[x]->c3inputWidth;c++) {
				cnn[x]->s2d[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
				cnn[x]->s2y[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3����
	for (j = 0;j<cnn[x]->c3outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth;c++) {
				cnn[x]->c3d[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3v[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3y[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4����
	for (j = 0;j<cnn[x]->s4outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight / cnn[x]->s4mapSize;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth / cnn[x]->s4mapSize;c++) {
				cnn[x]->s4d[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
				cnn[x]->s4y[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5���
	for (j = 0;j<cnn[x]->ooutputNum;j++) {
		cnn[x]->od[j] = (float)0.0;
		cnn[x]->ov[j] = (float)0.0;
		cnn[x]->oy[j] = (float)0.0;
	}
	// ��һ���������
	nSize mapSize = { cnn[x]->c1mapSize,cnn[x]->c1mapSize };//����˴�С,5X5
	nSize inSize = { cnn[x]->c1inputWidth,cnn[x]->c1inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnn[x]->s2inputWidth,cnn[x]->s2inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnn[x]->c1outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnn[x]->c1inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c1mapData[j * cnn[x]->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] += tempconv[r][c] * inputData[x*inSize.r*inSize.c+(t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//��ֹ����һ����û���κ��߼����⣬���ǣ�S2���������⡣
	//��������S2����


	// �ڶ�����������S2��������

	outSize.c = cnn[x]->c3inputWidth;//���ͼ���С,12X12
	outSize.r = cnn[x]->c3inputHeight;//���ͼ���С,12X12
	inSize.c = cnn[x]->s2inputWidth;//����ͼ���С,24X24
	inSize.r = cnn[x]->s2inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i < (cnn[x]->s2outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												//�²����ػ�
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
	// �������������,������ȫ����
	outSize.c = cnn[x]->s4inputWidth;//���ͼ���С,8X8
	outSize.r = cnn[x]->s4inputHeight;//���ͼ���С,8X8
	inSize.c = cnn[x]->c3inputWidth;//����ͼ���С,12X12
	inSize.r = cnn[x]->c3inputHeight;//����ͼ���С,12X12
	mapSize.c = cnn[x]->c3mapSize;//����˴�С,5X5
	mapSize.r = cnn[x]->c3mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnn[x]->c3outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnn[x]->c3inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
												 //��ʼ�����������
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c3mapData[j *cnn[x]->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn[x]->s2y[j * is * is + (t + r)* is + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn[x]->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c3basicData[i]);//�õ�C3���������MAP
																																								 //cout << i << " " << r << " " << c << " " << cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnn[x]->s4inputWidth;//����ͼ���С,8X8
	inSize.r = cnn[x]->s4inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnn[x]->s4mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnn[x]->s4mapSize;//���ͼ���С,4X4
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

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn[x]->s4outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn[x]->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
																													  //cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn[x]->oinputNum,cnn[x]->ooutputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
													  //nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn[x]->owData[i*nnSize.c + j];//�������֮�����,Ȼ�󷵻ؽ��
		cnn[x]->ov[i] = o;
	}
	for (i = 0; i<cnn[x]->ooutputNum; i++)//�����sigmoid����
		cnn[x]->oy[i] = activation_Sigma(cnn[x]->ov[i], cnn[x]->obasicData[i]);//����sigmoid����,�����������ֵ
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
	//���ڽṹ����û�ж��嵱ǰ�����MAP�Ĵ�С,��˻�õ�ǰ�����MAP�Ĵ�Сֻ��ͨ����һ������MAP�Ĵ�С�����
	int outSizeW = cnn[x]->s2inputWidth;//�����һ������MAP����Ĵ�С,������24X24
	int outSizeH = cnn[x]->s2inputHeight;//�����һ������MAP����Ĵ�С,������24X24
										 // ��һ��Ĵ���
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
	// S2����
	for (j = 0;j<cnn[x]->s2outChannels;j++) {
		for (r = 0;r<cnn[x]->c3inputHeight;r++) {
			for (c = 0;c<cnn[x]->c3inputWidth;c++) {
				cnn[x]->s2d[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
				cnn[x]->s2y[j*cnn[x]->c3inputHeight*cnn[x]->c3inputWidth + r*cnn[x]->c3inputWidth + c] = (float)0.0;
			}
		}
	}
	// C3����
	for (j = 0;j<cnn[x]->c3outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth;c++) {
				cnn[x]->c3d[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3v[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
				cnn[x]->c3y[j*cnn[x]->s4inputHeight*cnn[x]->s4inputWidth + r * cnn[x]->s4inputWidth + c] = (float)0.0;
			}
		}
	}
	// S4����
	for (j = 0;j<cnn[x]->s4outChannels;j++) {
		for (r = 0;r<cnn[x]->s4inputHeight / cnn[x]->s4mapSize;r++) {
			for (c = 0;c<cnn[x]->s4inputWidth / cnn[x]->s4mapSize;c++) {
				cnn[x]->s4d[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
				cnn[x]->s4y[j * (cnn[x]->s4inputHeight / cnn[x]->s4mapSize) * (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + r* (cnn[x]->s4inputWidth / cnn[x]->s4mapSize) + c] = (float)0.0;
			}
		}
	}
	// O5���
	for (j = 0;j<cnn[x]->ooutputNum;j++) {
		cnn[x]->od[j] = (float)0.0;
		cnn[x]->ov[j] = (float)0.0;
		cnn[x]->oy[j] = (float)0.0;
	}
	// ��һ���������
	nSize mapSize = { cnn[x]->c1mapSize,cnn[x]->c1mapSize };//����˴�С,5X5
	nSize inSize = { cnn[x]->c1inputWidth,cnn[x]->c1inputHeight };//����ͼ���С,28X28
	nSize outSize = { cnn[x]->s2inputWidth,cnn[x]->s2inputHeight };//���ͼ���С,24X24
	float mapout[24][24];//��ʱ����������õ�����
	float tempconv[5][5];//��ʱ�þ����,��ת֮���
	for (i = 0; i<(cnn[x]->c1outChannels); i++) {//��C1���ÿһ�����MAP,����Ϊ6
		for (j = 0; j<(cnn[x]->c1inChannels); j++) {//��C1���ÿһ������MAP,����Ϊ1
			for (t = 0; t <outSize.r; t++)
			{
				for (k = 0; k < outSize.c; k++)
				{
					mapout[t][k] = 0.0;
				}
			}
			for (r = 0; r<mapSize.r; r++) {
				for (c = 0; c<mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c1mapData[j * cnn[x]->c1outChannels *mapSize.r*mapSize.r + i * mapSize.r*mapSize.r + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout[t][k] += tempconv[r][c] * IData[x*inSize.r*inSize.c + (t + r) * inSize.r + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->c1v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		//��һ�����MAP��������е�����ͼ��֮��,�Ϳ��Խ���sigmoid�����ļ�����,�������������ѵõ������MAP��ÿһ��ֵ����sigmoid,��C3����ǰ�8X8��С�ľ�����sigmoid��������,�õ�8X8��С���������MAP
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c1v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c1basicData[i]);
				//cout <<i<<" "<<r<<" "<<c<<" :"<< cnn[x]->c1y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}
	//��ֹ����һ����û���κ��߼����⣬���ǣ�S2���������⡣
	//��������S2����


	// �ڶ�����������S2��������

	outSize.c = cnn[x]->c3inputWidth;//���ͼ���С,12X12
	outSize.r = cnn[x]->c3inputHeight;//���ͼ���С,12X12
	inSize.c = cnn[x]->s2inputWidth;//����ͼ���С,24X24
	inSize.r = cnn[x]->s2inputHeight;//����ͼ���С,24X24
	int mSize = 2;//��2Ϊ��С�ػ�
	for (i = 0; i < (cnn[x]->s2outChannels); i++) {//��6�����ͼ��,ÿһ������C1����гػ�
												   //�²����ػ�
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
	// �������������,������ȫ����
	outSize.c = cnn[x]->s4inputWidth;//���ͼ���С,8X8
	outSize.r = cnn[x]->s4inputHeight;//���ͼ���С,8X8
	inSize.c = cnn[x]->c3inputWidth;//����ͼ���С,12X12
	inSize.r = cnn[x]->c3inputHeight;//����ͼ���С,12X12
	mapSize.c = cnn[x]->c3mapSize;//����˴�С,5X5
	mapSize.r = cnn[x]->c3mapSize;//����˴�С,5X5
	float mapout2[8][8];//��ʱ����������õ�����
	for (i = 0; i<(cnn[x]->c3outChannels); i++) {//��C3���ÿһ�����MAP,����Ϊ12
		for (j = 0; j<(cnn[x]->c3inChannels); j++) {//��C3���ÿһ������MAP,����Ϊ6
													//��ʼ�����������
			for (t = 0; t < 8; t++)
			{
				for (k = 0; k < 8; k++)
				{
					mapout2[t][k] = 0.0;
				}
			}
			for (r = 0; r < mapSize.r; r++) {
				for (c = 0; c < mapSize.c; c++) {
					tempconv[r][c] = cnn[x]->c3mapData[j *cnn[x]->c3outChannels *mapSize.r*mapSize.c + i*mapSize.r*mapSize.c + (mapSize.r - 1 - r)*mapSize.r + mapSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			int is = outSize.r + mapSize.r - 1;
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							mapout2[t][k] = mapout2[t][k] + tempconv[r][c] * cnn[x]->s2y[j * is * is + (t + r)* is + k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t < outSize.r; t++) {
				for (k = 0; k < outSize.c; k++) {
					cnn[x]->c3v[i*outSize.r*outSize.c + t * outSize.c + k] += mapout2[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] = activation_Sigma(cnn[x]->c3v[i*outSize.r*outSize.c + r * outSize.c + c], cnn[x]->c3basicData[i]);//�õ�C3���������MAP
																																										  //cout << i << " " << r << " " << c << " " << cnn[x]->c3y[i*outSize.r*outSize.c + r * outSize.c + c] << endl;
			}
		}
	}

	// ���Ĳ���������
	inSize.c = cnn[x]->s4inputWidth;//����ͼ���С,8X8
	inSize.r = cnn[x]->s4inputHeight;//����ͼ���С,8X8
	outSize.c = inSize.c / cnn[x]->s4mapSize;//���ͼ���С,4X4
	outSize.r = inSize.r / cnn[x]->s4mapSize;//���ͼ���С,4X4
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

	// �����O5�Ĵ���
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	float O5inData[192]; //���䳤��Ϊ192����������S4������������
	for (i = 0; i < (cnn[x]->s4outChannels); i++) {//S4���12���������
		for (r = 0; r < outSize.r; r++) {//��ÿһ��4X4��MAP
			for (c = 0; c < outSize.c; c++) {
				O5inData[i*outSize.r*outSize.c + r*outSize.c + c] = cnn[x]->s4y[i*outSize.r*outSize.c + r*outSize.c + c];//����������һ������Ϊ192��һά����,����S4���i�����MAP�ĵ�r�е�c�е����ݵĴ洢λ��Ϊi*outSize.r*outSize.c+r*outSize.c+c,�����������ȴ洢,ע��
																														 //cout << O5inData[i*outSize.r*outSize.c + r*outSize.c + c] <<endl;
			}
		}
	}
	nSize nnSize = { cnn[x]->oinputNum,cnn[x]->ooutputNum };//����һ�������СΪ10(�߶�,����)X192(���,����)
															//nnSize.c=192,nnSize.r=10,����192X10��ȫ��������
	for (i = 0; i < nnSize.r; i++)
	{
		float o = 0;
		for (j = 0; j<nnSize.c; j++)
			o = o + O5inData[j] * cnn[x]->owData[i*nnSize.c + j];//�������֮�����,Ȼ�󷵻ؽ��
		cnn[x]->ov[i] = o;
	}
	for (i = 0; i<cnn[x]->ooutputNum; i++)//�����sigmoid����
		cnn[x]->oy[i] = activation_Sigma(cnn[x]->ov[i], cnn[x]->obasicData[i]);//����sigmoid����,�����������ֵ
	for (i = 0; i<cnn[x]->ooutputNum; i++)
		cnn[x]->e[i] = cnn[x]->oy[i] - LData[py*10+i];//�����ʵ�������ȥ������ȷ�����,��Ӧ��ʽΪai-yi=-(yi-ai),ע�������y[i]��ai,��yi��outputData[i]
																 // �����O5��������
	for (i = 0; i<cnn[x]->ooutputNum; i++)
		cnn[x]->od[i] = cnn[x]->e[i] * sigma_derivation(cnn[x]->oy[i]);//��10����Ԫ��˵,ÿ����Ԫ�������������ȹ�ʽΪ-(yi-ai)(ai*(1-ai)),ע�������y[i]��ai,��yi��outputData[i]
																			 // S4�㣬���ݵ�S4������
																			 // ����û�м����
	outSize.r = cnn[x]->s4inputWidth / cnn[x]->s4mapSize;
	outSize.c = cnn[x]->s4inputHeight / cnn[x]->s4mapSize;//S4�����������С,������4X4
	for (i = 0; i < cnn[x]->s4outChannels; i++) {//��ÿһ���������,����һ�����������һ����С�����жȾ�����֮��Ӧ
		for (r = 0; r < outSize.r; r++) {
			for (c = 0; c < outSize.c; c++) {
				for (j = 0; j < cnn[x]->ooutputNum; j++) {//�����Ӧ��ʽ����ͨ������������Ĳв���㹫ʽ,����MATLAB�汾������˵����ƪ����fvd������˵��
					int wInt = i*outSize.c*outSize.r + r*outSize.c + c;//wInt������λȨֵ,S4���i�����MAP��r�е�c�����j����Ԫ��ȨֵΪ[j][i*outSize.c*outSize.r + r*outSize.c + c],��Ϊ���Ƕ�ά�����ȴ洢����,��һά�����������ӵ������ĵ�j����Ԫ,�ڶ�ά��������������ϵ�Ȩֵ
					cnn[x]->s4d[i*outSize.r*outSize.r + r* outSize.r + c] = cnn[x]->s4d[i*outSize.r*outSize.r + r* outSize.r + c] + cnn[x]->od[j] * cnn[x]->owData[j * cnn[x]->ooutputNum +wInt];
				}
			}
		}
	}
	int mapdata = cnn[x]->s4mapSize;//������Ҫ�����ϲ�������,�����Ҫ����mapSize��С���ϲ���,������2X2
	nSize S4dSize = { cnn[x]->s4inputWidth / cnn[x]->s4mapSize,cnn[x]->s4inputHeight / cnn[x]->s4mapSize };//S4������жȾ����С,������4X4,Ҳ����S4����������С
	float C3e[8][8];
	for (i = 0; i<cnn[x]->c3outChannels; i++) {//C3��ÿһ�����MAP����Ӧһ�����жȾ���
												 //S4dSize12 mapSize2
		for (j = 0; j<S4dSize.r*cnn[x]->s4mapSize; j = j + cnn[x]->s4mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S4dSize.c*cnn[x]->s4mapSize; t = t + cnn[x]->s4mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn[x]->s4mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C3e[j][t + m] = cnn[x]->s4d[i * S4dSize.r*cnn[x]->s4mapSize + j * S4dSize.r / cnn[x]->s4mapSize + t / cnn[x]->s4mapSize];//�����
				}
			}
			for (n = 1; n < cnn[x]->s4mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S4dSize.c*cnn[x]->s4mapSize; t++) {//�з����л�
					C3e[j + n][t] = C3e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn[x]->s4inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��8
			for (c = 0; c<cnn[x]->s4inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��8
				cnn[x]->c3d[i * cnn[x]->s4inputHeight * cnn[x]->s4inputWidth +r*cnn[x]->s4inputWidth + c] = C3e[r][c] * sigma_derivation(cnn[x]->c3y[i * cnn[x]->s4inputHeight * cnn[x]->s4inputWidth + r*cnn[x]->s4inputWidth + c]) / (float)(cnn[x]->s4mapSize*cnn[x]->s4mapSize);//ע��������Ҫ����(float)(cnn[x]->s4mapSize*cnn[x]->s4mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C3������жȾ���
	}
	// S2�㣬S2��û�м����������ֻ�о�����м��������
	// �ɾ���㴫�ݸ������������ݶȣ��������㹲��6*12�����ģ��
	outSize.c = cnn[x]->c3inputWidth;//S2�����жȾ����СΪ12X12
	outSize.r = cnn[x]->c3inputHeight;//S2�����жȾ����СΪ12X12
	inSize.r = cnn[x]->s4inputWidth;
	inSize.c = cnn[x]->s4inputHeight;//C3�����жȾ���Ĵ�С
	mapSize.r = cnn[x]->c3mapSize;
	mapSize.c = cnn[x]->c3mapSize;//C3�����˴�С5X5
	float corr[12][12];//�洢��ؼ�����
	float exData[16][16];//�洢full֮�����ʱ����
	int addr, addc;

	addr = addc = mapSize.r - 1;//Ҫ��չ�ı߳�
	for (i = 0; i<cnn[x]->s2outChannels; i++) {//����S2��ÿһ�����MAP,6
		for (j = 0; j<cnn[x]->c3outChannels; j++) {//����C3��ÿһ�����MAP,����������ȫ���ӽṹ,���S2���ÿһ��ͼ����C3���ÿһ��ͼ���й�,12
													 //float** corr = correlation(cnn[x]->c3mapData[i][j], mapSize, cnn[x]->c3d[j], inSize, full);//���ﱾ��Ҫ��C3���Ӧ�ľ����������ת180��Ȼ���ڽ��о������,��ʵ���Ͼ�������ְѾ������ת��180��,�������ֱ�ӾͲ���ת�����,����ֱ�Ӻ;�������,full�������
			int outSizeW = inSize.c + (mapSize.c - 1); // ������������һ����,��ȫ����õ��ľ��MAP�Ŀ��/����,12
			int outSizeH = inSize.r + (mapSize.r - 1);// ������������һ����,��ȫ����õ��ľ��MAP�ĸ߶�/����,12
			int newSize = outSizeW - 1 + mapSize.c;//exInputData��С,16
												   //��չ����
			for (t = 0; t<inSize.r + 2 * addr; t++) {
				for (k = 0; k<inSize.c + 2 * addc; k++) {
					if (t<addr || k<addc || t >= (inSize.r + addr) || k >= (inSize.c + addc))//�������������ı�Ե��,����Ϊ0
						exData[t][k] = (float)0.0;
					else
						exData[t][k] = cnn[x]->c3d[j * (inSize.r + 2 * addr) * (inSize.r + 2 * addr) + (t - addr)*(inSize.r + 2 * addr) + k - addc]; // ��Ȼ,����ԭ����������
				}
			}
			//�������
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					corr[t][k] = 0.0;
				}
			}
			for (t = 0; t<outSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<outSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<mapSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<mapSize.c; c++) {//���ھ���˵�ÿһ��
							corr[t][k] = corr[t][k] + cnn[x]->c3mapData[i*cnn[x]->c3outChannels *mapSize.r * mapSize.c  + j *mapSize.r * mapSize.c + r * mapSize.r + c] * exData[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
						}
					}
				}
			}
			for (t = 0; t<outSize.r; t++) {
				for (k = 0; k<outSize.c; k++) {
					cnn[x]->s2d[i*outSize.r*outSize.r + t*outSize.r  + k] = cnn[x]->s2d[i*outSize.r*outSize.r + t*outSize.r + k] + corr[t][k];//���Ȼ�󷵻ظ�res
				}
			}
		}
	}
	// C1�㣬�����
	mapdata = cnn[x]->s2mapSize;//C1��������map�Ĵ�С,24X24
	nSize S2dSize = { cnn[x]->s2inputWidth / cnn[x]->s2mapSize,cnn[x]->s2inputHeight / cnn[x]->s2mapSize };//S2��������MAP�Ĵ�С,12X12���Pooling����ƽ�������Է��򴫵ݵ���һ��Ԫ������ݶ�û�б仯
	float C1e[24][24];
	for (i = 0; i<cnn[x]->c1outChannels; i++) {//C1��ÿһ�����MAP����Ӧһ�����жȾ���
		for (j = 0; j<S2dSize.r*cnn[x]->s2mapSize; j = j + cnn[x]->s2mapSize) {//���з�����,ÿ�����upr����ͬ��ֵ,ע�������Ǹ߶�,����һ��j����ԭʼmapһ�е�����,һ��forѭ��ִ����,����һ�е����ݾ���������
			for (t = 0; t<S2dSize.c*cnn[x]->s2mapSize; t = t + cnn[x]->s2mapSize) {// �������,��x������ÿ��upc��ֵ�ı�һ�θ�ֵ
				for (m = 0; m<cnn[x]->s2mapSize; m++) {//ÿ�ζ�������upc��Ԫ�ظ�ֵ
					C1e[j][t + m] = cnn[x]->s2d[i * S2dSize.r*S2dSize.r  + j*S2dSize.r / cnn[x]->s2mapSize+t / cnn[x]->s2mapSize];//�����
				}
			}
			for (n = 1; n < cnn[x]->s2mapSize; n++) {     //  �ߵ�����,�ڶ��е����һ��
				for (t = 0; t < S2dSize.c*cnn[x]->s2mapSize; t++) {//�з����л�
					C1e[j + n][t] = C1e[j][t];//���ղŵ�һ�еĽ��
				}
			}
		}
		for (r = 0; r<cnn[x]->s2inputHeight; r++)//��ÿһ�����жȾ������,ע�������С��24
			for (c = 0; c<cnn[x]->s2inputWidth; c++)//��ÿһ�����жȾ������,ע�������С��24
				cnn[x]->c1d[i*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth+r*cnn[x]->s2inputWidth+c] = C1e[r][c] * sigma_derivation(cnn[x]->c1y[i*cnn[x]->s2inputHeight*cnn[x]->s2inputWidth + r*cnn[x]->s2inputWidth + c]) / (float)(cnn[x]->s2mapSize*cnn[x]->s2mapSize);//ע��������Ҫ����(float)(cnn[x]->s2mapSize*cnn[x]->s2mapSize),������4,�Ա��ԭ�������жȾ���ƽ�������C1������жȾ���
	}

	//apply
	// C1���Ȩ�ظ���
	nSize dSize = { cnn[x]->s2inputHeight,cnn[x]->s2inputWidth };//C1�������Ⱦ����С,24X24
	nSize ySize = { cnn[x]->c1inputHeight,cnn[x]->c1inputWidth };//C1����������С,28X28
	mapSize.r = cnn[x]->c1mapSize;
	mapSize.c = cnn[x]->c1mapSize;//C1�����˴�С
	float cov[24][24];
	//float cmout[5][5];
	float tins[28][28];
	float tin[28][28];
	for (i = 0; i<cnn[x]->c1outChannels; i++) {//����ÿһ�����MAP,������6,��С24X24
		for (j = 0; j<cnn[x]->c1inChannels; j++) {//����ÿһ������ͼ��,������1,��С28X28
													//����,һάת��ά����,��ת180���ƺ�����
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tins[r][c] = IData[x*inSize.r*inSize.c + (t + r) * inSize.r*ySize.c + c];
				}
			}
			//����֮���Ի����,�����齻����򵥵�����,a=b,b=a����ֱ��д,Ҫ��C����ת!!!!
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin[r][c] = tins[ySize.r - 1 - r][ySize.c - 1 - c];//��ת180��,һĿ��Ȼ
																	   //cout << tin[r][c] << " ";
				}
				//cout << endl;
			}
			//system("pause");
			//��ת�����
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov[r][c] = cnn[x]->c1d[i*ySize.r * ySize.c +ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}

			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r+k] = cnn[x]->c1dmapData[j * cnn[x]->c1outChannels * mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] + cov[r][c] * tin[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
		cnn[x]->c1dbasicData[i] = -1 * 1.0*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// C3���Ȩ�ظ���
	dSize.c = cnn[x]->s4inputWidth;//C3�������Ⱦ����С,8X8
	dSize.r = cnn[x]->s4inputHeight;//C3�������Ⱦ����С,8X8
	ySize.c = cnn[x]->c3inputWidth;//C3����������С,12X12
	ySize.r = cnn[x]->c3inputHeight;//C3����������С,12X12
	mapSize.c = cnn[x]->c3mapSize;//C3�����˴�С,5X5
	mapSize.r = cnn[x]->c3mapSize;//C3�����˴�С,5X5
	float cov2[8][8];
	float tin2[12][12];
	for (i = 0; i<cnn[x]->c3outChannels; i++) {//����ÿһ�����MAP,������12,��С8X8
		for (j = 0; j<cnn[x]->c3inChannels; j++) {//����ÿһ������ͼ��,������8,��С12X12
			for (r = 0; r<ySize.r; r++) {
				for (c = 0; c<ySize.c; c++) {
					tin2[r][c] = cnn[x]->s2y[j*ySize.r * ySize.c + ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//��ת�����
			for (r = 0; r<dSize.r; r++) {
				for (c = 0; c<dSize.c; c++) {
					cov2[r][c] = cnn[x]->c3d[i*ySize.r * ySize.c + ySize.r*(dSize.r - 1 - r) + dSize.c - 1 - c];//��ת180��,һĿ��Ȼ
				}
			}
			//������
			for (t = 0; t<mapSize.r; t++) {//�������MAP��ÿһ��
				for (k = 0; k<mapSize.c; k++) {//�������MAP��ÿһ��
					for (r = 0; r<dSize.r; r++) {//���ھ���˵�ÿһ��
						for (c = 0; c<dSize.c; c++) {//���ھ���˵�ÿһ��
							cnn[x]->c3dmapData[j*cnn[x]->c3outChannels*mapSize.r*mapSize.r +i*mapSize.r*mapSize.r+t*mapSize.r+k] = cnn[x]->c3dmapData[j*cnn[x]->c3outChannels*mapSize.r*mapSize.r + i*mapSize.r*mapSize.r + t*mapSize.r + k] + cov2[r][c] * tin2[t + r][k + c];
							//outputData�ĵ�j�е�i�е�ֵ,���ھ���˵�r�е�c�е�ֵ��������֮��ԭʼͼ��ĵ�j+r�е�i+c�еĽ�����ܺ�,������˾������
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
		cnn[x]->c3dbasicData[i] = -1 * 1.0*sum;//����ƫ��b���ݶ�,ƫ��b���ݶȾ���ÿһ�����MAP[i]��Ӧ���жȾ���ĸ�Ԫ��֮��
	}
	// �����
	// ������Ҫ��ǰ��Ķ�ά���չ����һά����
	for (j = 0; j<cnn[x]->ooutputNum; j++) {//���������ÿһ�������Ԫ,��10����Ԫ
		for (i = 0; i<cnn[x]->oinputNum; i++)//��192����������ݶ�
			cnn[x]->odwData[j * 10+i] = -1 * 1.0*cnn[x]->od[j] * O5inData[i];//��W���ݶ���,��aj*delta,Ȼ���ѧϰ���Ը����ݶ�
		cnn[x]->odbasicData[j] = -1 * 1.0*cnn[x]->od[j];//��b�����ݶ�,b���ݶȾ������ж�delta
	}
}