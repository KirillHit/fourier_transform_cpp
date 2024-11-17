#ifndef FT_FFT_HPP_
#define FT_FFT_HPP_

#include <cmath>
#include <complex>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "types.hpp"

namespace ft {

void fft_1d(const CMat& src, CMat& dst, bool inverse = false);
void fft_2d(const cv::Mat& src, cv::Mat& dst, bool inverse = false);

}  // namespace ft

#endif  // FT_FFT_HPP_
