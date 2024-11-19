#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include "ft_cpp/ft_cpp.hpp"

void find_template(const cv::Mat& src, cv::Mat& templ)
{
    cv::imshow("template", templ);
    cv::Mat img_t, tsh_img;
    ft::to_complex(templ, templ);
    ft::template_matching(src, templ, img_t);
    ft::to_real(img_t, img_t);
    cv::normalize(img_t, img_t, 0, 1, cv::NORM_MINMAX);
    cv::threshold(img_t, tsh_img, 0.99, 1, cv::THRESH_BINARY);
    cv::imshow("img_t", img_t);
    cv::imshow("tsh_img", tsh_img);

    while (cv::waitKey() != (int)'q')
    {
    }
}

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/signs.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("org", img);

    cv::Mat c_img;
    ft::to_complex(img, c_img);

    cv::Mat template1 = img(cv::Rect(28, 12, 40, 70));
    cv::Mat template2 = img(cv::Rect(75, 180, 40, 72));
    cv::Mat template3 = img(cv::Rect(196, 195, 80, 56));
    find_template(c_img, template1);
    find_template(c_img, template2);
    find_template(c_img, template3);

    cv::destroyAllWindows();
}
