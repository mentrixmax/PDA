#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

 
using namespace cv;
using namespace std;


int matriz[512][512], mat_regiao[512][512];
IplImage* cinza;
IplImage* color;
IplImage* regiao;
int cont_atual, cont_ant,x,y;
int matriz[512][512], mat_regiao[512][512];
uchar* canal;
uchar* canal2;


void quest21(void)
{	
	color = cvLoadImage("c:\\imagens\\l.jpg",1);
	Mat image = imread("c:\\imagens\\turtle.bmp",1);
	Mat saida = Mat::zeros( image.size(), image.type() );
	cv::threshold (image, saida, 70, 255, CV_THRESH_BINARY_INV);
	color = new  IplImage(saida);
	//waitKey(0);
	int width = 498;
	int heigth = 509;

	Mat imageGrayScale ;
	Mat imageBinary;
	cvtColor( image, imageGrayScale, CV_RGB2GRAY );
	cvWaitKey(1);
	cinza = cvCreateImage(cvGetSize(color), 8,1);
	regiao = cvCreateImage(cvGetSize(color), 8,1);
	canal = (uchar*)cinza->imageData;
	canal2 = (uchar*)regiao->imageData;
	cvCvtColor(color, cinza, CV_RGB2GRAY);

	for(y=0;y<width;y++)//Copiando a imagem para a matriz
	{
		canal = (uchar*)(cinza->imageData + y*(cinza->widthStep));
		for(x=0;x<heigth;x++)
		{
			mat_regiao[x][y] = 0;
			matriz[x][y] = canal[x];
		}
	}
	cvNamedWindow("entrada", 1);
	cvShowImage("entrada", cinza);

	cvSetMouseCallback("entrada", mouseEvent, 0);
	cvWaitKey(0);
	cvDestroyWindow("entrada");


	int i=300;
	while(i != 0)
	{
		cont_atual = 0;
		for(y=0;y<width;y++)
		{

			for(x=0;x<heigth;x++)
			{
				if(mat_regiao[x][y] == 255)
				{

					if(matriz[x-1][y-1] < 127)
					{
						mat_regiao[x-1][y-1] = 255;
					}

					if(matriz[x][y-1] < 127)
					{
						mat_regiao[x][y-1] = 255;
					}

					if(matriz[x+1][y-1] < 127)
					{
						mat_regiao[x+1][y-1] = 255;
					}

					if(matriz[x-1][y] < 127)
					{
						mat_regiao[x-1][y] = 255;
					}

					if(matriz[x+1][y] < 127)
					{
						mat_regiao[x+1][y] = 255;
					}

					if(matriz[x-1][y+1] < 127)
					{
						mat_regiao[x-1][y+1] = 255;
					}

					if(matriz[x][y+1] < 127)
					{
						mat_regiao[x][y+1] = 255;
					}

					if(matriz[x+1][y+1] < 127)
					{
						mat_regiao[x+1][y+1] = 255;
					}

				}

			}

		}

		i--;
		for(y=0;y<width;y++)
		{
			canal2 = (uchar*)(regiao->imageData + y*(regiao->widthStep));
			for(x=0;x<heigth;x++)
			{
				canal2[x] = mat_regiao[x][y];
			}
		}
		cvWaitKey(100);
		cvNamedWindow("Regiao",1);
		cvShowImage("Regiao", regiao);
	}

	cvNamedWindow("Regiao",1);
	cvShowImage("Regiao", regiao);
	cvWaitKey(0);
}

void calculoHistograma(Mat dst){
    int histSize = 64;
    Mat hist;
   int brightness =  100;

    double a, b, contrast = 100;
    if( contrast > 0 )
    {
        double delta = 127.*contrast/100;
        a = 255./(255. - delta*2);
        b = a*(brightness - delta);
    }
    else
    {
        double delta = -128.*contrast/100;
        a = (256.-delta*2)/255.;
        b = a*brightness + delta;
    }
    Mat imaOut  ; 
    dst.convertTo(imaOut, CV_8U, a, b);
    calcHist(&imaOut, 1, 0, Mat(), hist, 1, &histSize, 0);
    Mat histImage = Mat::ones(200, 320, CV_8U)*255;
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, CV_32F);

    histImage = Scalar::all(255);
    int binW = cvRound((double)histImage.cols/histSize);

    for( int i = 0; i < histSize; i++ )
        rectangle( histImage, Point(i*binW, histImage.rows),
                   Point((i+1)*binW, histImage.rows - cvRound(hist.at<float>(i))),
                   Scalar::all(0), -1, 8, 0 );
    imshow("histogram", histImage);

} 
int main( )
{
    Mat src1;
    src1 = imread("IMG_20151204_163737043.jpg", cv::IMREAD_COLOR);
    namedWindow( "Original image",WINDOW_NORMAL);
    imshow( "Original image", src1 ); 

    Mat grey;
    cvtColor(src1, grey, cv::COLOR_BGR2GRAY);
 //   calculoHistograma(src1);
    medianBlur(grey, grey, 9);
    Mat sobelx;
    Sobel(grey, sobelx, CV_32F, 1, 0);
 
    double minVal, maxVal;
    minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
    cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;
 
    Mat draw;
    sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
 
    namedWindow("image",WINDOW_NORMAL);
    imshow("image", draw);
    Mat img_bw;
    cv::threshold(grey, img_bw, 0, 255, 8); 	 
    imshow("image", img_bw);
    waitKey(0);                                        
    return 0;
}

 	
