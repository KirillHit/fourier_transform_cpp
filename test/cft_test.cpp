#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/park.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::Size img_size = img.size();
    ft::dft_optimal_size(img, img);
    cv::imshow("org", img);

    cv::Mat custom_dft_img, custom_fft_img, cv_dft_img, c_img;
    cv::Mat custom_dft_mag, custom_fft_mag, cv_dft_mag;
    ft::to_complex(img, c_img);

    auto start = std::chrono::steady_clock::now();
    ft::dft_2d(c_img, custom_dft_img);
    auto dft_time = std::chrono::steady_clock::now();
    ft::fft_2d(c_img, custom_fft_img);
    auto fft_time = std::chrono::steady_clock::now();
    cv::dft(c_img, cv_dft_img);
    auto end = std::chrono::steady_clock::now();

    std::cout
        << std::format(
               "Custom dft time: {} ms, Custom fft time: {} ms, opencv dft time: {} ms",
               std::chrono::duration_cast<std::chrono::milliseconds>(dft_time - start).count(),
               std::chrono::duration_cast<std::chrono::milliseconds>(fft_time - dft_time).count(),
               std::chrono::duration_cast<std::chrono::milliseconds>(end - fft_time).count())
        << std::endl;

    ft::magnitude(custom_dft_img, custom_dft_mag);
    cv::imshow("custom_dft_mag", custom_dft_mag);
    ft::magnitude(custom_fft_img, custom_fft_mag);
    cv::imshow("custom_fft_mag", custom_fft_mag);
    ft::magnitude(cv_dft_img, cv_dft_mag);
    cv::imshow("cv_dft_mag", cv_dft_mag);

    cv::Mat custom_idft_img;
    ft::dft_2d(custom_dft_img, custom_idft_img, true);
    ft::to_real(custom_idft_img, custom_idft_img);
    cv::normalize(custom_idft_img, custom_idft_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("custom_idft_img", custom_idft_img);

    cv::Mat custom_ifft_img;
    ft::fft_2d(custom_fft_img, custom_ifft_img, true);
    ft::to_real(custom_ifft_img, custom_ifft_img);
    cv::normalize(custom_ifft_img, custom_ifft_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("custom_ifft_img", custom_ifft_img(cv::Rect(0, 0, img_size.width, img_size.height)));

    while (cv::waitKey() != (int)'q')
    {
    }
    cv::destroyAllWindows();
}
