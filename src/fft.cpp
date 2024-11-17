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

    CMat res(src_size);
    if (M == 2)
    {
        ft::CDouble* dst_ptr = res.ptr<CDouble>();
        const ft::CDouble* src_ptr = src.ptr<CDouble>();
        dst_ptr[0] = src_ptr[0] + src_ptr[1];
        dst_ptr[1] = src_ptr[0] - src_ptr[1];
        res.copyTo(dst);
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

    res(cv::Rect(0, 0, M / 2, 1)) = even + w_prod;
    res(cv::Rect(M / 2, 0, M / 2, 1)) = even - w_prod;

    res.copyTo(dst);
}


void fft_2d(const cv::Mat& src, cv::Mat& dst, bool inverse)
{
    cv::Size src_size = src.size();
    bool t = src_size.height & 1;
    if ((src_size.height & 1) || (src_size.width & 1))
    {
        throw std::runtime_error("Wrong input size");
    }
    CMat res = src.clone();
    for (int v_idx = 0; v_idx < src_size.height; ++v_idx)
    {
        CMat col_step = res.col(v_idx);
        CMat dst;
        fft_1d(col_step.t(), dst, inverse);
        col_step = dst.t();
    }
    for (int h_idx = 0; h_idx < src_size.width; ++h_idx)
    {
        CMat row_step = res.row(h_idx);
        fft_1d(row_step, row_step, inverse);
    }
    if (inverse)
    {
        res = res / (src_size.height * src_size.width);
        cv::flip(res, res, -1);
    }
    dst = res;
}

}  // namespace ft
