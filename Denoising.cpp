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


int main(int argc, char** argv)
{

	cv::Mat_<cv::Vec3b> source = cv::imread(argv[1]);
	cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);

	int ksize = atoi(argv[2]);
	int N = atoi(argv[3]);

	cv::imshow("Source Image", source);

	auto begin = chrono::high_resolution_clock::now();

	// basic gaussian conv
	denoising::Median(source, destination, ksize, N);
	auto end = std::chrono::high_resolution_clock::now();

	cv::imshow("Processed Image1", destination);

	std::chrono::duration<double> diff = end - begin;
	cout << "Total time: " << diff.count() << " s" << endl;

	cv::waitKey();
	return 0;
}