#include "ft_cpp/dft.hpp"
#include "ft_cpp/utils.hpp"
#include <iostream>

namespace ft {

CMat dft_1d(const CMat& src)
{
    cv::Size src_size = src.size();
    if ((src_size.width != 1) && (src_size.height != 1))
    {
        throw std::runtime_error("Wrong input size");
    }
    const size_t M = std::max(src_size.height, src_size.width);

    using namespace std::complex_literals;
    const CDouble W = std::exp((-2.0 * 1i * std::numbers::pi) / static_cast<double>(M));

    CMat w_prod(cv::Size(M, M));
    CMat w_prep(cv::Size((M - 1) * (M - 1) + 1, 1));
    CDouble step_w = 1.0;
    ft::CDouble* prep_ptr = w_prep.ptr<CDouble>();
    for (size_t idx = 0; idx <= (M - 1) * (M - 1); ++idx, prep_ptr++, step_w *= W)
    {
        *prep_ptr = step_w;
    }

    prep_ptr = w_prep.ptr<CDouble>();
    for (size_t h_idx = 0; h_idx < M; h_idx++)
    {
        CMat first_row = w_prod.row(h_idx);
        CDouble* dst_ptr = first_row.ptr<CDouble>();
        for (size_t w_idx = 0; w_idx < M; ++w_idx, dst_ptr++, step_w *= W)
        {
            *dst_ptr = prep_ptr[h_idx * w_idx];
        }
    }


    CMat res;
    if (src_size.width == 1)
    {
        res = w_prod * src;
    }
    else
    {
        CMat temp = src.clone();
        res = (w_prod * temp.t()).t();
    }

    return res;
}

void dft_2d(const cv::Mat& src, cv::Mat& dst)
{
    cv::Size src_size = src.size();
    CMat res = src.clone();
    for (int v_idx = 0; v_idx < src_size.height; ++v_idx)
    {
        CMat col_step = res.col(v_idx);
        dft_1d(col_step).copyTo(col_step);
    }
    for (int h_idx = 0; h_idx < src_size.width; ++h_idx)
    {
        CMat row_step = res.row(h_idx);
        dft_1d(row_step).copyTo(row_step);
        // row_step.row(h_idx) = dft_1d(row_step);
    }
    dst = res;
}

}  // namespace ft
