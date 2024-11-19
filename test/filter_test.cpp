#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"

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
    ft::to_complex(blur_kernel, blur_kernel);

    cv::Mat Sobel_h_kernel = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    ft::to_complex(Sobel_h_kernel, Sobel_h_kernel);

    cv::Mat Sobel_v_kernel = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    ft::to_complex(Sobel_v_kernel, Sobel_v_kernel);

    cv::Mat Laplas_kernel = (cv::Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    ft::to_complex(Laplas_kernel, Laplas_kernel);

    cv::Mat blur_img;
    ft::convolve_dft(c_img, blur_kernel, blur_img);
    ft::to_real(blur_img, blur_img);
    cv::normalize(blur_img, blur_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("blur_img", blur_img);

    cv::Mat Sobel_h_img;
    ft::convolve_dft(c_img, Sobel_h_kernel, Sobel_h_img);
    ft::to_real(Sobel_h_img, Sobel_h_img);
    cv::normalize(Sobel_h_img, Sobel_h_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Sobel_h_img", Sobel_h_img);

    cv::Mat Sobel_v_img;
    ft::convolve_dft(c_img, Sobel_v_kernel, Sobel_v_img);
    ft::to_real(Sobel_v_img, Sobel_v_img);
    cv::normalize(Sobel_v_img, Sobel_v_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Sobel_v_img", Sobel_v_img);

    cv::Mat Laplas_img;
    ft::convolve_dft(c_img, Laplas_kernel, Laplas_img);
    ft::to_real(Laplas_img, Laplas_img);
    cv::normalize(Laplas_img, Laplas_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Laplas_img", Laplas_img);

    while (cv::waitKey() != (int)'q')
    {
    }
    cv::destroyAllWindows();
}
