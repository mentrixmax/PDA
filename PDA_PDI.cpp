#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/time.h>
#include "iostream"
#include <stdio.h>

using namespace cv;
using namespace std;


uchar* canal;
uchar* canal2;
IplImage* cinza;
IplImage* color;
IplImage* regiao;

int cont_atual, cont_ant,x,y;
int matriz[512][512], mat_regiao[512][512];

Mat brilhoNovo(Mat &mat, int valor){

	Mat new_image = Mat::zeros( mat.size(), mat.type() );
	double beta = valor;
	double alpha;
	alpha = 1;
	for( int y = 0; y < mat.rows; y++ )
	{ for( int x = 0; x < mat.cols; x++ )
	{ for( int c = 0; c < 3; c++ )
	{
		new_image.at<Vec3b>(y,x)[c] =
			saturate_cast<uchar>( alpha*( mat.at<Vec3b>(y,x)[c] ) + beta );
	}
	}
	}
	return new_image;
}



void filtroMateus(Mat &src)
{
	Mat imageFinal =  src; //imread("c:\\imagens\\IMG_20151204_163737043.jpg",1);
	Mat image =  src; //imread("c:\\imagens\\IMG_20151204_163737043.jpg",1);
	struct timeval inicio, fim;
        gettimeofday(&inicio,NULL);
	Mat saida = Mat::zeros( image.size(), image.type() );
	image = brilhoNovo(image, -10);
	medianBlur(image,saida,9);
	for (int i=0; i<10; i++)
        { medianBlur(saida,saida,9);
         }

	IplImage* imgPosMedia = new IplImage(saida);

	cv::threshold (saida, saida, 90, 255, THRESH_BINARY_INV);
	color = new  IplImage(saida);
	Mat imageGrayScale ;
	Mat imageBinary;
	cvtColor( image, imageGrayScale, COLOR_BGR2GRAY );
	cinza = cvCreateImage(cvGetSize(color), 8,1);
	regiao = cvCreateImage(cvGetSize(color), 8,1);
	canal = (uchar*)cinza->imageData;
	canal2 = (uchar*)regiao->imageData;
	cvCvtColor(color, cinza, COLOR_BGR2GRAY);


	cv::Mat m = cv::cvarrToMat(cinza);

	Mat imagemProcessada = Mat::zeros( m.size(), m.type() );;
	for (int i = 0 ; i<=m.size().height-1; i++)
	{ for(int j=0; j<=m.size().width-1; j++){
		if ( m.at<char>(i,j) < (char)253 ||m.at<char>(i,j) > (char)50 ){
			imagemProcessada.at<char>(i,j) = 255;
		}  
	}
	} 

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

	vector<Point> dataB;
	for (int colunas = 0; colunas < imagemProcessada.cols-1;colunas++){
		for (int linhas = imagemProcessada.rows -1 ; linhas > 0; linhas--){
			if ( imagemProcessada.at<char>(linhas,colunas) == (char) 255){
				Point p ;
				p.x = colunas;
				p.y = linhas;
				dataB.push_back(p);
				break;
			}
		}
	} 


	Mat pontosMaximo = Mat::zeros( imagemProcessada.size(), imagemProcessada.type() );
	int k = 0;

/*	while ( k <dataB.size())
	{
		circle(	imageFinal, dataB.at(k),3,Scalar(255,255,255),1,8,0);
		k++;
	}*/

	k = 0;
	while ( k <data.size())
	{	
		circle(	imageFinal, data.at(k),3,Scalar(255,255,255),1,8,0);
		k++;
	} 

	namedWindow("maximos", CV_NORMAL);
	imshow("maximos",  imageFinal);
	//waitKey(0);

/*	namedWindow("processada", CV_NORMAL);
	imshow("processada", new  imagemProcessada);
	waitKey(0);

	namedWindow("entrada", CV_NORMAL);
	imshow("entrada", cinza);
*/
    	gettimeofday(&fim,NULL);
        int tmili =  0;
	tmili = (int) ( 1000 * (fim.tv_sec - inicio.tv_sec) + ((fim.tv_usec - inicio.tv_usec) /1000));
	waitKey(0);
        printf(" tempo decorrido em MS \n %d ", tmili);
        waitKey(0); 
}
int main(int argc, char** argv){

/*	string filename = "VID_20151204_164043568.mp4";
	VideoCapture capture(filename);
	Mat frame;
	if( !capture.isOpened() )
		throw "Error when reading steam_avi";

	namedWindow( "w", 1);
	for( ; ; )
	{
		capture >> frame;
		//  if(!frame)
		//     break;
	    filtroMateus(frame);
	    imshow("w", frame);
		waitKey(80); // waits to display frame
	}
	waitKey(0);*/
	Mat imageFinal = imread("IMG_20151204_163737043.jpg",1);
	filtroMateus(imageFinal);
}
