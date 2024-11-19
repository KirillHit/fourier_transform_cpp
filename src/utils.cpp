#include "ft_cpp/utils.hpp"
#include "ft_cpp/fft.hpp"

namespace ft {

void to_complex(const cv::Mat1d& src, cv::Mat& dst)
{
    cv::Mat planes[] = {cv::Mat_<double>(src), cv::Mat::zeros(src.size(), CV_64F)};
    cv::Mat res;
    cv::merge(planes, 2, res);
    dst = res;
}

void to_real(const CMat& src, cv::Mat& dst)
{
    cv::extractChannel(src, dst, 0);
}

void to_center(cv::Mat& src)
{
    int cx = src.cols / 2;
    int cy = src.rows / 2;
    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void magnitude(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat planes[2];
    cv::split(src, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag_img = planes[0];
    mag_img += cv::Scalar::all(1);
    cv::log(mag_img, mag_img);
    mag_img = mag_img(cv::Rect(0, 0, mag_img.cols & -2, mag_img.rows & -2));
    to_center(mag_img);
    normalize(mag_img, mag_img, 0, 1, cv::NORM_MINMAX);
    dst = mag_img;
}

void dft_optimal_size(const cv::Mat& src, cv::Mat& dst)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    cv::copyMakeBorder(
        src,
        padded,
        0,
        m - src.rows,
        0,
        n - src.cols,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));
    dst = padded;
}

void cut_frequencies(const cv::Mat& src, cv::Mat& dst, const unsigned int& radius, bool inverse)
{
    cv::Size src_size = src.size();
    cv::Mat mask = cv::Mat::zeros(src_size, CV_8U);
    cv::circle(mask, cv::Point(0, 0), radius, 255, -1);
    cv::circle(mask, cv::Point(src_size.width, 0), radius, 255, -1);
    cv::circle(mask, cv::Point(0, src_size.height), radius, 255, -1);
    cv::circle(mask, cv::Point(src_size.width, src_size.height), radius, 255, -1);
    if (inverse)
    {
        cv::bitwise_not(mask, mask);
    }
    cv::bitwise_and(src, src, dst, mask = mask);
}

void convolve_dft(const cv::Mat& src, const cv::Mat& conv, cv::Mat& dst)
{
    cv::Mat temp1;
    cv::Size src_size = src.size();
    cv::Size conv_size = conv.size();
    cv::Size border_size(conv_size.width / 2, conv_size.height / 2);
    cv::copyMakeBorder(
        src,
        temp1,
        border_size.height,
        border_size.height,
        border_size.width,
        border_size.width,
        cv::BORDER_REFLECT);
    cv::Size temp1_size = temp1.size();

    cv::Mat temp2(temp1_size, src.type(), cv::Scalar::all(0));
    cv::Mat roi2(temp2, cv::Rect(0, 0, conv.cols, conv.rows));
    conv.copyTo(roi2);

    fft_2d(temp1, temp1);
    fft_2d(temp2, temp2);

    cv::mulSpectrums(temp1, temp2, temp1, 0);

    fft_2d(temp1, temp1, true);
    temp1(cv::Rect(border_size.width, border_size.height, src_size.width, src_size.height))
        .copyTo(dst);
}

void template_matching(const cv::Mat& src, const cv::Mat& conv, cv::Mat& dst)
{
    cv::Size src_size = src.size();
    cv::Mat temp2(src_size, src.type(), cv::Scalar::all(0));
    cv::Mat roi2(temp2, cv::Rect(0, 0, conv.cols, conv.rows));
    conv.copyTo(roi2);

    cv::Mat temp1;
    fft_2d(src, temp1);
    fft_2d(temp2, temp2);

    cv::Mat res;
    cv::mulSpectrums(temp1, temp2, res, 0, true);

    fft_2d(res, res, true);
    res(cv::Rect(0, 0, src_size.width, src_size.height)).copyTo(dst);
}

}  // namespace ft
