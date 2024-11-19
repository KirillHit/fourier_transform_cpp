#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"

void test_kernel(const cv::Mat& src, cv::Mat& kernel)
{
    cv::Mat res;
    ft::to_complex(kernel, kernel);
    ft::convolve_dft(src, kernel, res);
    ft::to_real(res, res);
    cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
    cv::imshow("res", res);

    while (cv::waitKey() != (int)'q')
    {
    }
}

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/park.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("org", img);

    cv::Mat c_img;
    ft::to_complex(img, c_img);

    int blur_size = 11;
    cv::Mat blur_kernel =
        cv::Mat::ones(cv::Size(blur_size, blur_size), CV_32F) / (blur_size * blur_size);
    cv::Mat Sobel_h_kernel = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Sobel_v_kernel = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::Mat Laplas_kernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    
    test_kernel(c_img, blur_kernel);
    test_kernel(c_img, Sobel_h_kernel);
    test_kernel(c_img, Sobel_v_kernel);
    test_kernel(c_img, Laplas_kernel);

    cv::destroyAllWindows();
}
