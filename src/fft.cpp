#include "ft_cpp/fft.hpp"
#include "ft_cpp/utils.hpp"

namespace ft {

/**
 * @brief Splits a list into even and odd indices
 *
 * @param src Expected size: hight = 1, wight = M. M is even.
 * @param even
 * @param odd
 */
inline void segregate(const CMat& src, CMat& even, CMat& odd)
{
    cv::Size src_size = src.size();
    cv::Size dst_size = cv::Size(src_size.width / 2, src_size.height);
    CMat dst_odd(dst_size);
    CMat dst_even(dst_size);
    const ft::CDouble* ptr = src.ptr<CDouble>();
    ft::CDouble* ptr_even = dst_even.ptr<CDouble>();
    ft::CDouble* ptr_odd = dst_odd.ptr<CDouble>();
    for (size_t idx = 0; idx < src_size.width - 1; idx += 2, ptr_even++, ptr_odd++)
    {
        *ptr_even = ptr[idx];
        *ptr_odd = ptr[idx + 1];
    }
    even = dst_even;
    odd = dst_odd;
}

void fft_1d(const CMat& src, CMat& dst, bool inverse)
{
    cv::Size src_size = src.size();
    const size_t M = src_size.width;

    if (M == 1)
    {
        dst = src;
        return;
    }

    CMat even, odd;
    segregate(src, even, odd);
    fft_1d(even, even, inverse);
    fft_1d(odd, odd, inverse);

    double sign = -1.0;
    if (inverse)
    {
        double sign = 1.0;
    }
    using namespace std::complex_literals;
    const CDouble W = std::exp((sign * 2.0 * 1i * std::numbers::pi) / static_cast<double>(M));
    CMat w_prod(cv::Size(M / 2, 1));
    CDouble step_w = 1.0;
    ft::CDouble* w_ptr = w_prod.ptr<CDouble>();
    ft::CDouble* odd_ptr = odd.ptr<CDouble>();
    for (size_t idx = 0; idx < M / 2; ++idx, w_ptr++, odd_ptr++, step_w *= W)
    {
        *w_ptr = step_w * (*odd_ptr);
    }

    CMat res(src_size);
    res(cv::Rect(0, 0, M / 2, 1)) = even + w_prod;
    res(cv::Rect(M / 2, 0, M / 2, 1)) = even - w_prod;

    res.copyTo(dst);
}


void prepared_image_for_fft(const cv::Mat& src, cv::Mat& dst)
{
    auto power_two_size = [](int x) {
        return static_cast<int>(std::pow(2, std::ceil(std::log2(static_cast<double>(x)))));
    };
    cv::Size src_size = src.size();
    cv::Mat padded;
    cv::copyMakeBorder(
        src,
        padded,
        0,
        power_two_size(src_size.height) - src.rows,
        0,
        power_two_size(src_size.width) - src.cols,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));
    dst = padded;
}

void fft_2d(const cv::Mat& src, cv::Mat& dst, bool inverse)
{
    cv::Size src_size = src.size();
    CMat res = src.clone();
    prepared_image_for_fft(res, res);
    cv::Size res_size = res.size();
    for (int v_idx = 0; v_idx < res_size.width; ++v_idx)
    {
        CMat col_step = res.col(v_idx);
        CMat dst;
        fft_1d(col_step.t(), dst, inverse);
        col_step = dst.t();
    }
    for (int h_idx = 0; h_idx < res_size.height; ++h_idx)
    {
        CMat row_step = res.row(h_idx);
        fft_1d(row_step, row_step, inverse);
    }
    if (inverse)
    {
        res = res / (res_size.height * res_size.width);
        cv::flip(res, res, -1);
    }
    dst = res;
}

}  // namespace ft
