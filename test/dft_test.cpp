
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/lena.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    ft::dft_optimal_size(img, img);

    cv::Mat custom_dft_img, cv_dft_img, c_img;
    cv::Mat custom_dft_mag, cv_dft_mag;
    ft::to_complex(img, c_img);

    auto start = std::chrono::steady_clock::now();
    ft::dft_2d(c_img, custom_dft_img);
    auto mid = std::chrono::steady_clock::now();
    cv::dft(c_img, cv_dft_img);
    auto end = std::chrono::steady_clock::now();

    std::cout << std::format(
                     "Custom dft time: {} ms, Opencv dft time: {} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count(),
                     std::chrono::duration_cast<std::chrono::milliseconds>(end - mid).count())
              << std::endl;

    ft::magnitude(custom_dft_img, custom_dft_mag);
    cv::imshow("custom_dft_mag", custom_dft_mag);
    ft::magnitude(cv_dft_img, cv_dft_mag);
    cv::imshow("cv_dft_mag", cv_dft_mag);

    cv::Mat custom_idft_img;
    ft::dft_2d(custom_dft_img, custom_idft_img, true);
    ft::to_real(custom_idft_img, custom_idft_img);
    cv::normalize(custom_idft_img, custom_idft_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("custom_idft_img", custom_idft_img);

    cv::imshow("org", img);

    while (cv::waitKey() != (int)'q')
    {
    }
    cv::destroyAllWindows();
}
