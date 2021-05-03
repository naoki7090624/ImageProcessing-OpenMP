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


namespace arithmetic {
	void plus(const cv::Mat_<cv::Vec3d>& src1, const cv::Mat_<cv::Vec3d>& src2, cv::Mat_<cv::Vec3b>& dst) {
		dst.create(src1.rows, src1.cols);
		dst = cv::Vec3b(0, 0, 0);
#pragma omp parallel for
		for (int i = 0; i < src1.rows; i++) {
			for (int j = 0; j < src1.cols; j++) {
				dst(i, j)[0] = (int)((src1(i, j)[0] + src2(i, j)[0]) / 2.0);
				dst(i, j)[1] = (int)((src1(i, j)[1] + src2(i, j)[1]) / 2.0);
				dst(i, j)[2] = (int)((src1(i, j)[2] + src2(i, j)[2]) / 2.0);
			}
		}
	}
}

namespace transformation {
	void linear(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<cv::Vec3b>& dst, double scaleH, double scaleW) {
		int H = (int)(scaleH * src.rows);
		int W = (int)(scaleW * src.cols);
		dst.create(H, W);
		dst = cv::Vec3b(0, 0, 0);
#pragma omp parallel for
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				int x = (int)((double)j / scaleW + 0.5);
				int y = (int)((double)i / scaleH + 0.5);
				if (x == src.cols) x = src.cols - 1;
				if (y == src.rows) y = src.rows - 1;
				dst(i, j)[0] = src(y, x)[0];
				dst(i, j)[1] = src(y, x)[1];
				dst(i, j)[2] = src(y, x)[2];
			}
		}
	}
	void resize(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<cv::Vec3b>& dst, int H, int W) {
		double scaleH = (double)H / (double)src.rows;
		double scaleW = (double)W / (double)src.cols;
		dst.create(H, W);
		dst = cv::Vec3b(0, 0, 0);
#pragma omp parallel for
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				int x = (int)((double)j / scaleW + 0.5);
				int y = (int)((double)i / scaleH + 0.5);
				if (x == src.cols) x = src.cols - 1;
				if (y == src.rows) y = src.rows - 1;
				dst(i, j)[0] = src(y, x)[0];
				dst(i, j)[1] = src(y, x)[1];
				dst(i, j)[2] = src(y, x)[2];
			}
		}
	}
}

int main(int argc, char** argv)
{

	cv::Mat_<cv::Vec3b> source = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3b> source2 = cv::imread(argv[2]);
	cv::Mat_<cv::Vec3b> destination1(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination2(source.rows, source.cols);
	cv::Mat_<cv::Vec3b> destination3(source.rows, source.cols);

	int h = atoi(argv[3]);
	int w = atoi(argv[4]);


	cv::imshow("Source Image", source);

	// transformation
	auto begin1 = chrono::high_resolution_clock::now();
	transformation::resize(source, destination1, h, w);
	auto end1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff1 = end1 - begin1;
	cout << "Total time: " << diff1.count() << " s" << endl;

	transformation::resize(source2, destination2, h, w);

	// arithmetic
	auto begin2 = chrono::high_resolution_clock::now();
	arithmetic::plus(destination1, destination2, destination3);
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff2 = end2 - begin2;
	cout << "Total time: " << diff2.count() << " s" << endl;

	cv::imshow("Style Image", source2);
	cv::imshow("Processed Image", destination3);


	cv::waitKey();
	return 0;
}