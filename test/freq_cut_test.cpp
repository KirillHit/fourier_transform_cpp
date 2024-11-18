#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"


int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/lena.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    ft::dft_optimal_size(img, img);
    cv::imshow("org", img);

    cv::Mat custom_fft_img, c_img;
    cv::Mat custom_fft_mag;
    ft::to_complex(img, c_img);

    ft::fft_2d(c_img, custom_fft_img);
    ft::magnitude(custom_fft_img, custom_fft_mag);
    cv::imshow("custom_fft_mag", custom_fft_mag);

    cv::Mat cut_fft_img, cut_fft_mag, cut_ifft_img;
    ft::cut_frequencies(custom_fft_img, cut_fft_img, 40);
    ft::magnitude(cut_fft_img, cut_fft_mag);
    cv::imshow("cut_fft_mag", cut_fft_mag);

    ft::fft_2d(cut_fft_img, cut_ifft_img, true);
    ft::to_real(cut_ifft_img, cut_ifft_img);
    cv::normalize(cut_ifft_img, cut_ifft_img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("ifft_img", cut_ifft_img);

    cv::Mat cut_fft_img2, cut_fft_mag2, cut_ifft_img2;
    ft::cut_frequencies(custom_fft_img, cut_fft_img2, 40, true);
    ft::magnitude(cut_fft_img2, cut_fft_mag2);
    cv::imshow("cut_fft_mag2", cut_fft_mag2);

    ft::fft_2d(cut_fft_img2, cut_ifft_img2, true);
    ft::to_real(cut_ifft_img2, cut_ifft_img2);
    cv::normalize(cut_ifft_img2, cut_ifft_img2, 0, 1, cv::NORM_MINMAX);
    cv::imshow("ifft_img2", cut_ifft_img2);


    while (cv::waitKey() != (int)'q')
    {
    }
    cv::destroyAllWindows();
}
