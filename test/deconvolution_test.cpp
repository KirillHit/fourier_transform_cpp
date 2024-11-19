#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <format>
#include <tuple>
#include "ft_cpp/ft_cpp.hpp"

void apply_kernel(const cv::Mat& src, const cv::Mat& kernel, cv::Mat& dst)
{
    cv::Mat res;
    ft::convolve_dft(src, kernel, res);
    cv::Mat img;
    ft::to_real(res, img);
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Normal kernel", img);
    res.copyTo(dst);
}

void prepare_conv(cv::Mat& src, cv::Mat& conv)
{
    cv::Mat temp1;
    cv::Size src_size = src.size();
    cv::Size conv_size = conv.size();

    cv::Mat temp2(src_size, src.type(), cv::Scalar::all(0));
    cv::Mat roi2(temp2, cv::Rect(0, 0, conv.cols, conv.rows));
    conv.copyTo(roi2);

    ft::fft_2d(src, temp1);
    ft::fft_2d(temp2, temp2);

    temp1.copyTo(src);
    temp2.copyTo(conv);
}

void get_gaussian_kernel(cv::Mat& dst, int size)
{
    cv::Mat res = cv::Mat::zeros(cv::Size(size, size), CV_64F);
    int center = size / 2;
    res.at<double>(center, center) = 1;
    cv::GaussianBlur(res, res, cv::Size(size, size), 0);
    res.copyTo(dst);
}

int main(void)
{
    cv::Mat img = cv::imread(std::string(SOURCE_DIR) + "/images/lena.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imshow("org", img);
    cv::Size src_size = img.size();

    cv::Mat c_img;
    ft::to_complex(img, c_img);

    cv::Mat Gaussian_kernel;
    get_gaussian_kernel(Gaussian_kernel, 11);
    ft::to_complex(Gaussian_kernel, Gaussian_kernel);

    cv::Mat blur_img;
    apply_kernel(c_img, Gaussian_kernel, blur_img);

    prepare_conv(blur_img, Gaussian_kernel);

    typedef std::tuple<const cv::Mat&, const cv::Mat&, const cv::Size&> tup_t;
    tup_t tup(blur_img, Gaussian_kernel, src_size);
    cv::TrackbarCallback cv_cb = [](int pos, void* userdata) {
        tup_t& tup = *(tup_t*)userdata;
        const cv::Mat& src = std::get<0>(tup);
        const cv::Mat& kernel = std::get<1>(tup);
        const cv::Size& src_size = std::get<2>(tup);
        cv::Mat ikernel, res;
        ft::inverse_img(kernel, ikernel, (static_cast<double>(pos)) / 100000);
        cv::mulSpectrums(src, ikernel, res, 0);
        cv::dft(res, res, cv::DFT_INVERSE);
        // It works but it's too slow
        // ft::fft_2d(res, res, true);
        ft::to_real(res, res);
        res = res(cv::Rect(0, 0, src_size.width, src_size.height));
        cv::normalize(res, res, 0, 1, cv::NORM_MINMAX);
        cv::imshow("Deconvolution", res);
    };

    int noise = 2000;
    cv_cb(noise, &tup);
    cv::createTrackbar("Noise", "Deconvolution", &noise, 20000, cv_cb, &tup);

    while (cv::waitKey() != (int)'q')
    {
    }

    cv::destroyAllWindows();
}
