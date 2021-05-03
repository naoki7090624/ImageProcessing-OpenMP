#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock
#include <valarray>
#include <math.h>
#include <algorithm>

#ifdef _DEBUG
#pragma comment(lib, "opencv_world451d.lib")
#else
#pragma comment(lib, "opencv_world451.lib")
#endif

using namespace std;
#define PI 3.14159265359f
typedef vector<double> Array;
typedef vector<Array> Matrix;

namespace convolution
{
	Matrix getGaussian(int ksize, double sigma)
	{
		double sum = 0.0;
		int i, j;
		int x, y;
		int k = ksize * 2 + 1;
		Matrix kernel(k, Array(k));

		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				x = i - ksize;
				y = j - ksize;
				kernel[i][j] = exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / sqrt(2.0 * PI * sigma * sigma);
				sum += kernel[i][j];
			}
		}

		for (i = 0; i < k; i++) {
			for (j = 0; j < k; j++) {
				kernel[i][j] /= sum;
			}
		}

		return kernel;
	}

	void convGaussian(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int ksize, double sigma) {

		Matrix kernel = getGaussian(ksize, sigma);
		int k = ksize * 2 + 1;
		//double kernel[3][3] = { { 0.0625, 0.125, 0.0625 }, { 0.125, 0.25, 0.125 }, { 0.0625, 0.125, 0.0625 } };

		dst.create(src.rows, src.cols);
		dst = cv::Vec3d(0.0, 0.0, 0.0);

		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i < ksize || i >= (src.rows - ksize) || j < ksize || j >= (src.cols - ksize)) {
					dst(i, j)[0] = src(i, j)[0];
					dst(i, j)[1] = src(i, j)[1];
					dst(i, j)[2] = src(i, j)[2];
				}
				else {
					for (int n = 0; n < k; n++) {
						for (int m = 0; m < k; m++) {
							dst(i, j)[0] += (int)(src(i + n - ksize, j + m - ksize)[0] * kernel[n][m]);
							dst(i, j)[1] += (int)(src(i + n - ksize, j + m - ksize)[1] * kernel[n][m]);
							dst(i, j)[2] += (int)(src(i + n - ksize, j + m - ksize)[2] * kernel[n][m]);
						}
					}
				}
			}
		}
	}
}

namespace separable {
	Array getGaussian(int ksize, double sigma)
	{
		double sum = 0.0;
		int x;
		int k = ksize * 2 + 1;
		Array kernel(k);

		for (int i = 0; i < k; i++) {
			x = i - ksize;
			kernel[i] = exp(-(x * x) / (2.0 * sigma * sigma)) / sqrt(2.0 * PI * sigma * sigma);
			sum += kernel[i];
		}

		for (int i = 0; i < k; i++) {
			kernel[i] /= sum;
		}

		return kernel;
	}
	void convGaussian(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int fsize, double sigma) {
		Array kernel = getGaussian(fsize, sigma);
		int k = fsize * 2 + 1;

		cv::Mat_<cv::Vec3d> temp(src.rows, src.cols);
		temp = cv::Vec3d(0.0, 0.0, 0.0);

		dst.create(src.rows, src.cols);
		dst = cv::Vec3d(0.0, 0.0, 0.0);

		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i < fsize || i >= (src.rows - fsize) || j < fsize || j >= (src.cols - fsize)) {
					temp(i, j)[0] = src(i, j)[0];
					temp(i, j)[1] = src(i, j)[1];
					temp(i, j)[2] = src(i, j)[2];
				}
				else {
					for (int n = 0; n < k; n++) {
						temp(i, j)[0] += (src(i + n - fsize, j)[0] * kernel[n]);
						temp(i, j)[1] += (src(i + n - fsize, j)[1] * kernel[n]);
						temp(i, j)[2] += (src(i + n - fsize, j)[2] * kernel[n]);
					}
				}
			}
		}

#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i < fsize || i >= (src.rows - fsize) || j < fsize || j >= (src.cols - fsize)) {
					dst(i, j)[0] = src(i, j)[0];
					dst(i, j)[1] = src(i, j)[1];
					dst(i, j)[2] = src(i, j)[2];
				}
				else {
					for (int n = 0; n < k; n++) {
						dst(i, j)[0] += (int)(temp(i, j + n - fsize)[0] * kernel[n]);
						dst(i, j)[1] += (int)(temp(i, j + n - fsize)[1] * kernel[n]);
						dst(i, j)[2] += (int)(temp(i, j + n - fsize)[2] * kernel[n]);
					}
				}
			}
		}

	}
}



int main(int argc, char** argv)
{

	cv::Mat_<cv::Vec3b> source = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3b> destination1(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination2(source.rows, source.cols);

	int ksize = atoi(argv[2]);
	double sigma = atof(argv[3]);
	cout << "kernel size: " << ksize << endl;

	cv::imshow("Source Image", source);

	auto begin = chrono::high_resolution_clock::now();

	// basic gaussian conv
	convolution::convGaussian(source, destination1, ksize, sigma);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff1 = end1 - begin;

	// separable gaussian conv
	separable::convGaussian(source, destination2, ksize, sigma);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff2 = end2 - end1;

	cout << "2D conv time: " << diff1.count() << " s" << endl;
	cout << "separable time: " << diff2.count() << " s" << endl;

	cv::imshow("Gaussian Conv", destination1);
	cv::imshow("Separable Gaussian Conv", destination2);

	std::chrono::duration<double> diff = end2 - begin;
	cout << "Total time: " << diff.count() << " s" << endl;

	cv::waitKey();
	return 0;
}