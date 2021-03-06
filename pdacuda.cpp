#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/time.h>
#include "iostream"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/cudaarithm.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafilters.hpp"

using namespace cv;
using namespace std;
using namespace cv::cuda;

uchar* canal;
uchar* canal2;
IplImage* cinza;
IplImage* color;
IplImage* regiao;

__global__ void brilho(float4* imagem, int width, int height, float b)
{
	const int i = blockIdx.x * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;

	if(i < width * height)
	{
		float v = imagem[i].x + b;
                float u  = imagem[i].y + b;
                float z = imagem[i].z + b;
		imagem[i] = make_float4(v, u, z, 0);
	}
}

extern "C" void cuda_brilho(float* imagem, int width, int height, dim3 blocks, dim3 block_size, float bri)
{
	brilho <<< blocks, block_size >>> ((float4*)imagem, width, height, bri);
}

IplImage* brilhoCuda(  IplImage* input_image, float bri ){
    int width = input_image->width;
    int height = input_image->height;
    int bpp = input_image->nChannels;

    float* cpu_image = new float[width * height * 4];
    if (!cpu_image)
    {
        std::cout << "ERROR: Failed to allocate memory" << std::endl;

    }

    for (int i = 0; i < width * height; i++)
	{
		cpu_image[i * 4 + 0] = (unsigned char)input_image->imageData[i * bpp + 0] / 255.f;
		cpu_image[i * 4 + 1] = (unsigned char)input_image->imageData[i * bpp + 1] / 255.f;
		cpu_image[i * 4 + 2] = (unsigned char)input_image->imageData[i * bpp + 2] / 255.f;
	}

    float* gpu_image = NULL;
	cudaError_t cuda_err = cudaMalloc((void **)(&gpu_image), (width * height * 4) * sizeof(float));
    if (cuda_err != cudaSuccess)
    {
        std::cout << "ERROR: Failed cudaMalloc" << std::endl;

    }

	cuda_err = cudaMemcpy(gpu_image, cpu_image, (width * height * 4) * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess)
    {
        std::cout << "ERROR: Failed cudaMemcpy" << std::endl;
     }

	dim3 block(16, 16);
	dim3 grid((int)ceil(double((width * height) / 256.0)));

    cuda_brilho(gpu_image, width, height, grid, block,bri);
    cudaMemcpy(cpu_image, gpu_image, (width * height * 4) * sizeof(float), cudaMemcpyDeviceToHost);

   if (cuda_err != cudaSuccess)
    {
        std::cout << "ERROR: Failed cudaMemcpy" << std::endl;

    }

    cuda_err = cudaFree(gpu_image);
    if (cuda_err != cudaSuccess)
    {
        std::cout << "ERROR: Failed cudaFree" << std::endl;

    }

	char* buff = new char[width * height * bpp];
    if (!buff)
    {
        std::cout << "ERROR: Failed to allocate memory" << std::endl;
     }


	for (int i = 0; i < (width * height); i++)
	{
		buff[i * bpp + 0] = (char)floor(cpu_image[i * 4 + 0] * 255.f);
		buff[i * bpp + 1] = (char)floor(cpu_image[i * 4 + 1] * 255.f);
		buff[i * bpp + 2] = (char)floor(cpu_image[i * 4 + 2] * 255.f);
	}

    IplImage* out_image = cvCreateImage(cvSize(width, height), input_image->depth, bpp);
    if (!out_image)
    {
        std::cout << "ERROR: Failed cvCreateImage" << std::endl;
    }

    out_image->imageData = buff;
    return out_image;
}

int filtroMateus( cv::Mat src, String fileSaida){
  Mat imageFinal = src;
  Mat image = src;
  Mat saida = Mat::zeros(image.size(), image.type());
  cv::cuda::Stream out;
  struct timeval inicio, fim;
  gettimeofday(&inicio, NULL);
  IplImage* brilho = brilhoCuda(new IplImage(src),-10);
  image = cv::cvarrToMat(brilho);
  cv::cuda::GpuMat dstM, srcM, dstG;
  srcM.upload(src);
  cv::cuda::cvtColor(srcM, dstG, COLOR_BGR2GRAY);
  cv::Ptr<cv::cuda::Filter> blurFilter = cv::cuda::createBoxFilter(dstG.type(), -1, cv::Size(9,9));
  blurFilter->apply(dstG, dstM);
  cv::Mat resultMedia(dstM);
  src = resultMedia;
  cv::Mat dst, dst2 ;
  cv::cuda::GpuMat dstC, srcC;
  srcC.upload(src);
  dstC = dstG;
  cv::Mat result_host(dstC);
  cv::cuda::threshold(result_host, dst2, 90, 255.0, cv::THRESH_BINARY_INV);
  cv::Mat result_host2(dst2);
  Mat imagemProcessada =result_host2;

	vector<Point> data;

	//pecorrendo todos os pontos da imagem detectado a parte de cima!

	for (int col = imagemProcessada.cols-1;  col>0; col--){

		for( int lin =0; lin< imagemProcessada.rows; lin++){

			if ( imagemProcessada.at<char>(lin,col) == (char) 255){

				Point p ;

				p.x = col;

				p.y = lin;

				data.push_back(p);

				break;

			}

		}

	}

	int k = 0;
	gettimeofday(&fim, NULL);
	while ( k <data.size())

	{

		circle(	imageFinal, data.at(k),3,Scalar(255,0,0),1,8,0);
		k++;

	}
	imwrite( fileSaida,imageFinal);
	int tmili = 0;
	tmili = (int) (1000 * (fim.tv_sec - inicio.tv_sec)
			+ ((fim.tv_usec - inicio.tv_usec) / 1000));

	printf(" tempo decorrido em MS %d  \n", tmili);

}
string itos(int i) // convert int to string
{
    stringstream s;
    s << i;
    return s.str();
}


int main(void){

for (int i=1; i<=55; i++) {
	 std::string result = itos(i)+".jpg";
	 std::string resultout = "/home/lamp/saidasmateus/"+itos(i)+"out.jpg";
     std::cout << "" << result;
	 Mat imageFinal = imread(result, 1);
	 filtroMateus(imageFinal,resultout);
	}
return 0;
}

