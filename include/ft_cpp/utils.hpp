#ifndef FT_UTILS_HPP_
#define FT_UTILS_HPP_

#include <cmath>
#include <complex>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "types.hpp"

namespace ft {

void to_complex(const cv::Mat1d& src, cv::Mat& dst);
void to_real(const CMat& src, cv::Mat& dst);
void to_center(cv::Mat& src);
void magnitude(const cv::Mat& src, cv::Mat& dst);
void dft_optimal_size(const cv::Mat& src, cv::Mat& dst);
void cut_frequencies(const cv::Mat& src,
                     cv::Mat& dst,
                     const unsigned int& radius,
                     bool inverse = false);
void convolve_dft(const cv::Mat& src, const cv::Mat& conv, cv::Mat& dst);

}  // namespace ft

#endif  // UTILS_HPP_
