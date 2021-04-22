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

	void convGaussian(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int fsize, double sigma) {

		Matrix kernel = getGaussian(fsize, sigma);
		int k = fsize * 2 + 1;
		//double kernel[3][3] = { { 0.0625, 0.125, 0.0625 }, { 0.125, 0.25, 0.125 }, { 0.0625, 0.125, 0.0625 } };

		dst.create(src.rows, src.cols);
		dst = cv::Vec3d(0.0, 0.0, 0.0);
		
		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i < fsize || i >= (src.rows - fsize) || j < fsize || j >= (src.cols - fsize)) {
					dst(i, j)[0] = 0;
					dst(i, j)[1] = 0;
					dst(i, j)[2] = 0;
				}
				else {
					for (int n = 0; n < k; n++) {
						for (int m = 0; m < k; m++) {
							dst(i, j)[0] += (int)(src(i + n - fsize, j + m - fsize)[0] * kernel[n][m]);
							dst(i, j)[1] += (int)(src(i + n - fsize, j + m - fsize)[1] * kernel[n][m]);
							dst(i, j)[2] += (int)(src(i + n - fsize, j + m - fsize)[2] * kernel[n][m]);
						}
					}
				}
			}
		}
	}

	void convLaplacian(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int fsize) {

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
					temp(i, j)[0] = 0;
					temp(i, j)[1] = 0;
					temp(i, j)[2] = 0;
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
					dst(i, j)[0] = 0;
					dst(i, j)[1] = 0;
					dst(i, j)[2] = 0;
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

namespace denoising {
	void Median(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int fsize, int N) {
		int k = (fsize * 2 + 1);
		int msize = k * k;
		double num = N * 2.0 + 1.0;

		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {

				if (i < fsize || i >= (src.rows - fsize) || j < fsize || j >= (src.cols - fsize)) {
					dst(i, j)[0] = 0;
					dst(i, j)[1] = 0;
					dst(i, j)[2] = 0;
				}

				else {
					int mid;
					Array sum(3);
					Matrix mlist(msize, Array(3));

					for (int n = 0; n < k; n++) {
						for (int m = 0; m < k; m++) {
							mid = n * k + m;
							mlist[mid][0] = src(i + n - fsize, j + m - fsize)[0];
							mlist[mid][1] = src(i + n - fsize, j + m - fsize)[1];
							mlist[mid][2] = src(i + n - fsize, j + m - fsize)[2];
						}
					}

					sort(mlist.begin(), mlist.end(), [](const vector<double>& alpha, const vector<double>& beta) {return alpha[0] < beta[0]; });
					mid = floor(msize / 2);
					sum[0] = 0;
					sum[1] = 0;
					sum[2] = 0;
					for (int n = mid - N; n <= mid + N; n++) {
						sum[0] += (int)mlist[n][0];
						sum[1] += (int)mlist[n][1];
						sum[2] += (int)mlist[n][2];
					}
					dst(i, j)[0] = (int)(sum[0] / (num));
					dst(i, j)[1] = (int)(sum[1] / (num));
					dst(i, j)[2] = (int)(sum[2] / (num));
				}

			}
		}
	}
}

namespace color
{
	void rgb2lch(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3d>& dst, int fsize) {
		dst.create(src.rows, src.cols);
		dst = cv::Vec3d(0.0, 0.0, 0.0);
		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				double R = src(i, j)[0] / 255.0;
				double G = src(i, j)[1] / 255.0;
				double B = src(i, j)[2] / 255.0;
				R = R > 0.04045 ? (pow((R + 0.055) / 1.055, 2.4)) : (R / 12.92);
				G = G > 0.04045 ? (pow((G + 0.055) / 1.055, 2.4)) : (G / 12.92);
				B = B > 0.04045 ? (pow((B + 0.055) / 1.055, 2.4)) : (B / 12.92);

				double X = 0.41239080 * R + 0.35758434 * G + 0.18048079 * B;
				double Y = 0.21263901 * R + 0.71516868 * G + 0.07219232 * B;
				double Z = 0.01933082 * R + 0.11919478 * G + 0.95053215 * B;
				X = X * 0.950489;
				Y = Y;
				Z = Z * 1.088840;
				X = X > 0.008856 ? (pow(X, 1.0 / 3.0)) : ((7.787 * X) + (4.0 / 29.0));
				Y = Y > 0.008856 ? (pow(Y, 1.0 / 3.0)) : ((7.787 * Y) + (4.0 / 29.0));
				Z = Z > 0.008856 ? (pow(Z, 1.0 / 3.0)) : ((7.787 * Z) + (4.0 / 29.0));

				double L = 116.0 * Y - 16.0;
				double a = 500.0 * (X - Y);
				double b = 200.0 * (Y - Z);

				double C = sqrt(a * a + b * b);
				double H = atan2(b, a);
				if (H < 0.0)
					H = H + 2.0 * 3.14;

				dst(i, j)[0] = L;
				dst(i, j)[1] = C;
				dst(i, j)[2] = H;
			}
		}
	}

	void lch2rgb(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<cv::Vec3b>& dst, int fsize) {
		dst.create(src.rows, src.cols);
		dst = cv::Vec3b(0, 0, 0);
		#pragma omp parallel for
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				double L = src(i, j)[0];
				double C = src(i, j)[1];
				double H = src(i, j)[2];

				double a = C * cos(H);
				double b = C * sin(H);

				double X = (L + 16.0) / 116.0 + a / 500.0;
				double Y = (L + 16.0) / 116.0;
				double Z = (L + 16.0) / 116.0 - b / 200.0;

				X = X > 0.20689655 ? (pow(X, 3.0)) : (0.12841855 * (X - 4.0 / 29.0));
				Y = Y > 0.20689655 ? (pow(Y, 3.0)) : (0.12841855 * (Y - 4.0 / 29.0));
				Z = Z > 0.20689655 ? (pow(Z, 3.0)) : (0.12841855 * (Z - 4.0 / 29.0));

				X = X * 0.950489;
				Y = Y;
				Z = Z * 1.088840;

				X = X > 0.0031308 ? (1.055 * pow(X, 1.0 / 2.4) - 0.055) : (12.92 * X);
				Y = Y > 0.0031308 ? (1.055 * pow(Y, 1.0 / 2.4) - 0.055) : (12.92 * Y);
				Z = Z > 0.0031308 ? (1.055 * pow(Z, 1.0 / 2.4) - 0.055) : (12.92 * Z);

				double R = 3.24096994 * X - 1.53738318 * Y - 0.49861076 * Z;
				double G = -0.96924364 * X + 1.8759675 * Y + 0.04155506 * Z;
				double B = 0.05563008 * X - 0.20397696 * Y + 1.05697151 * Z;
				dst(i, j)[0] = int(R * 255.0);
				dst(i, j)[1] = int(G * 255.0);
				dst(i, j)[2] = int(B * 255.0);
			}
		}
	}
}

int main(int argc, char** argv)
{

	cv::Mat_<cv::Vec3b> source = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3d> destination(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination2(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination3(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination4(source.rows, source.cols);

	cv::imshow("Source Image", source);

	//color::rgb2lch(source, destination, 1);
	//color::lch2rgb(destination, destination2, 1);

	auto begin = chrono::high_resolution_clock::now();

	// basic gaussian conv
	convolution::convGaussian(source, destination3, 1, 1.3);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff1 = end1 - begin;

	// separable gaussian conv
	//separable::convGaussian(source, destination4, 1, 1.3);
	denoising::Median(source, destination4, 1, 3);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff2 = end2 - end1;

	cout << "2D conv time: " << diff1.count() << " s" << endl;
	cout << "1D conv time: " << diff2.count() << " s" << endl;

	cv::imshow("Processed Image", destination3);
	cv::imshow("Processed Image", destination4);

	std::chrono::duration<double> diff = end2 - begin;
	cout << "Total time: " << diff.count() << " s" << endl;
	//cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
	//cout << "IPS: " << iter / diff.count() << endl;

	cv::waitKey();
	return 0;
}