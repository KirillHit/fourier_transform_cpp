
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>
#include "ft_cpp/ft_cpp.hpp"

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/lena.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    ft::dft_optimal_size(img, img);
    
    cv::Mat custom_dft_img, cv_dft_img, c_img;
    cv::Mat custom_dft_mag, cv_dft_mag;
    ft::to_complex(img, c_img);

    ft::dft_2d(c_img, custom_dft_img);
    ft::magnitude(custom_dft_img, custom_dft_mag);    
    cv::imshow("custom_dft_mag", custom_dft_mag);

    cv::dft(c_img, cv_dft_img);
    ft::magnitude(cv_dft_img, cv_dft_mag);    
    cv::imshow("cv_dft_mag", cv_dft_mag);

    cv::imshow("org", img);

    while (cv::waitKey() != (int)'q')
    {
    }
    cv::destroyAllWindows();
}
