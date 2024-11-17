#ifndef FT_TYPES_HPP_
#define FT_TYPES_HPP_

#include <complex>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace ft {

typedef std::complex<double> CDouble;
typedef cv::Mat_<CDouble> CMat;

}  // namespace ft

#endif  // FT_TYPES_HPP_
