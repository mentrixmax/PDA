#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/time.h>
#include "iostream"
#include <stdio.h>
#include <string.h>

using namespace cv;
using namespace std;

uchar* canal;
uchar* canal2;
IplImage* cinza;
IplImage* color;
IplImage* regiao;

int cont_atual, cont_ant, x, y;
int matriz[512][512], mat_regiao[512][512];

Mat brilhoNovo(Mat &mat, int valor) {

	Mat new_image = Mat::zeros(mat.size(), mat.type());
	double beta = valor;
	double alpha;
	alpha = 1;
	for (int y = 0; y < mat.rows; y++) {
		for (int x = 0; x < mat.cols; x++) {
			for (int c = 0; c < 3; c++) {
				new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
						alpha * (mat.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
	return new_image;
}

void filtroMateus(Mat &src, string fileout) {
	Mat imageFinal = src;
	Mat image = src;
	struct timeval inicio, fim;
	gettimeofday(&inicio, NULL);
	Mat saida = Mat::zeros(image.size(), image.type());
	image = brilhoNovo(image, -10);
	cvtColor(image, image, COLOR_BGR2GRAY);
        
	blur(image, saida, cv::Size(9,9));
	cv::threshold(saida, saida, 90, 255, THRESH_BINARY);
        
	cv::Mat m = saida;

	Mat imagemProcessada = Mat::zeros(m.size(), m.type());
        imagemProcessada = saida;
	vector<Point> data;
	//pecorrendo todos os pontos da imagem detectado a parte de cima!
	for (int col = imagemProcessada.cols - 1; col > 0; col--) {
		for (int lin = 0; lin < imagemProcessada.rows; lin++) {
			if (imagemProcessada.at<char>(lin, col) == (char) 0) {
				Point p;
				p.x = col;
				p.y = lin;
				data.push_back(p);
				break;
			}
		}
	}
	
	gettimeofday(&fim, NULL);
	int k = 0;
	while (k < data.size()) {
		circle(imageFinal, data.at(k), 3, Scalar(255, 0, 0), 1, 8, 0);
		k++;
	}

	

	int tmili = 0;
	tmili = (int) (1000 * (fim.tv_sec - inicio.tv_sec)
			+ ((fim.tv_usec - inicio.tv_usec) / 1000));

	imwrite( fileout, imageFinal );
	printf(" tempo decorrido em MS %d  \n", tmili);

}

string itos(int i) // convert int to string
{
    stringstream s;
    s << i;
    return s.str();
}
int main(int argc, char** argv) {
	for (int i=1; i<=55; i++) {
	 std::string result = itos(i)+".jpg";
	 std::string resultout = "/home/lamp/mateussaidas/"+itos(i)+"out.jpg";
     std::cout << "" << result;
	 Mat imageFinal = imread(result, 1);
	 filtroMateus(imageFinal,resultout);
	}
	waitKey(0);
}
