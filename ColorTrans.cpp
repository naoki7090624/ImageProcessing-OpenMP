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


namespace color
{
	void rgb2lch(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3d>& dst, double angle) {
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
				X = X / 0.950489;
				Y = Y;
				Z = Z / 1.088840;
				X = X > 0.008856 ? (pow(X, 1.0 / 3.0)) : ((7.787 * X) + (4.0 / 29.0));
				Y = Y > 0.008856 ? (pow(Y, 1.0 / 3.0)) : ((7.787 * Y) + (4.0 / 29.0));
				Z = Z > 0.008856 ? (pow(Z, 1.0 / 3.0)) : ((7.787 * Z) + (4.0 / 29.0));

				double L = 116.0 * Y - 16.0;
				double a = 500.0 * (X - Y);
				double b = 200.0 * (Y - Z);

				double C = sqrt(a * a + b * b);
				double H = atan2(b, a);
				//if (H < 0.0)
				//	H = H + 2.0 * PI;
				//else if (H > 2.0 * PI)
				//	H = H - 2.0 * PI;

				dst(i, j)[0] = L;
				dst(i, j)[1] = C;
				dst(i, j)[2] = H;
			}
		}
	}

	void lch2rgb(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<cv::Vec3b>& dst) {
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

				double R = 3.24096994 * X - 1.53738318 * Y - 0.49861076 * Z;
				double G = -0.96924364 * X + 1.8759675 * Y + 0.04155506 * Z;
				double B = 0.05563008 * X - 0.20397696 * Y + 1.05697151 * Z;

				R = R > 0.0031308 ? (1.055 * pow(R, 1.0 / 2.4) - 0.055) : (12.92 * R);
				G = G > 0.0031308 ? (1.055 * pow(G, 1.0 / 2.4) - 0.055) : (12.92 * G);
				B = B > 0.0031308 ? (1.055 * pow(B, 1.0 / 2.4) - 0.055) : (12.92 * B);

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
	cv::Mat_<cv::Vec3d> destination1(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination2(source.rows, source.cols);

	double angle = atof(argv[2]);

	cv::imshow("Source Image", source);

	auto begin = chrono::high_resolution_clock::now();
	// rgb to lch
	color::rgb2lch(source, destination1, angle);
	color::lch2rgb(destination1, destination2);
	auto end = std::chrono::high_resolution_clock::now();

	cv::imshow("Processed Image1", destination2);

	std::chrono::duration<double> diff = end - begin;
	cout << "Total time: " << diff.count() << " s" << endl;

	cv::waitKey();
	return 0;
}