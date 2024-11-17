#ifndef FT_DFT_HPP_
#define FT_DFT_HPP_

#include <cmath>
#include <complex>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "types.hpp"

namespace ft {

CMat dft_1d(const CMat& src);
void dft_2d(const cv::Mat& src, cv::Mat& dst);

}  // namespace ft

#endif  // FT_DFT_HPP_
