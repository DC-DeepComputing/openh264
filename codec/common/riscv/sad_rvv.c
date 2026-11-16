/*!
 * \copy
 *     Copyright (c)  2026, Cisco Systems
 *     All rights reserved.
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions
 *     are met:
 *
 *        Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *     A PARTICULAR PURPOSE ARE DISCLAIMED.
 */

/*
 * RISC-V Vector (RVV 1.0) SAD kernels for openh264.
 *
 * 形态选择：SAD 是「逐 lane 绝对差 + 跨行累加 + 末尾一次归约」，
 * 没有 gather、没有跨 lane 洗牌，是 RVV 上最赚的一类。
 *   |a-b| = vmaxu - vminu - vsub   （GCC 15 未暴露 vabd intrinsic）
 *   累加：vwaddu.wv 把 u8 加宽累加到 u16（每 lane ≤ 255*h ≤ 4080，安全）
 *   归约：vredsum 只做一次（循环内归约会被延迟吃掉全部收益）
 */

#include <stdint.h>
#include "typedefs.h"
#include "sad_common.h"

#if defined(HAVE_RVV)
#include <riscv_vector.h>

static inline int32_t sad_core_rvv (const uint8_t* pSrc, int32_t iSrcStride,
                                    const uint8_t* pRef, int32_t iRefStride,
                                    int32_t iWidth, int32_t iHeight) {
  const size_t vl = __riscv_vsetvl_e8m1 ((size_t)iWidth);
  vuint16m2_t acc = __riscv_vmv_v_x_u16m2 (0, vl);

  for (int32_t y = 0; y < iHeight; y++) {
    vuint8m1_t va = __riscv_vle8_v_u8m1 (pSrc, vl);
    vuint8m1_t vb = __riscv_vle8_v_u8m1 (pRef, vl);
    vuint8m1_t mx = __riscv_vmaxu_vv_u8m1 (va, vb, vl);
    vuint8m1_t mn = __riscv_vminu_vv_u8m1 (va, vb, vl);
    vuint8m1_t d  = __riscv_vsub_vv_u8m1 (mx, mn, vl);
    acc = __riscv_vwaddu_wv_u16m2 (acc, d, vl);
    pSrc += iSrcStride;
    pRef += iRefStride;
  }

  vuint32m1_t sum = __riscv_vwredsumu_vs_u16m2_u32m1 (
                      acc, __riscv_vmv_v_x_u32m1 (0, 1), vl);
  return (int32_t) __riscv_vmv_x_s_u32m1_u32 (sum);
}

#define SAD_FUNC(name, w, h)                                                   \
int32_t WelsSampleSad##name##_rvv (uint8_t* pSrc, int32_t iSrcStride,          \
                                   uint8_t* pRef, int32_t iRefStride) {        \
  return sad_core_rvv (pSrc, iSrcStride, pRef, iRefStride, w, h);              \
}

SAD_FUNC (16x16, 16, 16)
SAD_FUNC (16x8,  16,  8)
SAD_FUNC (8x16,   8, 16)
SAD_FUNC (8x8,    8,  8)
SAD_FUNC (8x4,    8,  4)
SAD_FUNC (4x8,    4,  8)
SAD_FUNC (4x4,    4,  4)

/* SadFour batched: 4 directions share one src row load.
 * Original did 4 separate sad_core_rvv calls => src read 4 times.
 * Here each row loads src ONCE and 4 ref rows (up/down/left/right),
 * accumulating into 4 independent vectors in lockstep.
 * Order pSad[0]=-stride,[1]=+stride,[2]=-1,[3]=+1 preserved exactly. */
#define SAD_ROW_ACC(acc, s, r)                                                 \
  do {                                                                         \
    vuint8m1_t v_  = __riscv_vle8_v_u8m1 ((r), vl);                            \
    vuint8m1_t mx_ = __riscv_vmaxu_vv_u8m1 ((s), v_, vl);                      \
    vuint8m1_t mn_ = __riscv_vminu_vv_u8m1 ((s), v_, vl);                      \
    vuint8m1_t d_  = __riscv_vsub_vv_u8m1 (mx_, mn_, vl);                      \
    (acc) = __riscv_vwaddu_wv_u16m2 ((acc), d_, vl);                           \
  } while (0)

#define SAD_FOUR_FUNC(name, w, h)                                              \
void WelsSampleSadFour##name##_rvv (uint8_t* pSrc, int32_t iSrcStride,         \
                                    uint8_t* pRef, int32_t iRefStride,         \
                                    int32_t* pSad) {                           \
  const size_t vl = __riscv_vsetvl_e8m1 ((size_t)(w));                         \
  vuint16m2_t a0 = __riscv_vmv_v_x_u16m2 (0, vl);                              \
  vuint16m2_t a1 = __riscv_vmv_v_x_u16m2 (0, vl);                              \
  vuint16m2_t a2 = __riscv_vmv_v_x_u16m2 (0, vl);                              \
  vuint16m2_t a3 = __riscv_vmv_v_x_u16m2 (0, vl);                              \
  for (int32_t y = 0; y < (h); y++) {                                          \
    vuint8m1_t s = __riscv_vle8_v_u8m1 (pSrc, vl);                             \
    SAD_ROW_ACC (a0, s, pRef - iRefStride);                                    \
    SAD_ROW_ACC (a1, s, pRef + iRefStride);                                    \
    SAD_ROW_ACC (a2, s, pRef - 1);                                             \
    SAD_ROW_ACC (a3, s, pRef + 1);                                             \
    pSrc += iSrcStride;                                                        \
    pRef += iRefStride;                                                        \
  }                                                                            \
  vuint32m1_t seed = __riscv_vmv_v_x_u32m1 (0, 1);                             \
  pSad[0] = (int32_t) __riscv_vmv_x_s_u32m1_u32 (                              \
              __riscv_vwredsumu_vs_u16m2_u32m1 (a0, seed, vl));                \
  pSad[1] = (int32_t) __riscv_vmv_x_s_u32m1_u32 (                              \
              __riscv_vwredsumu_vs_u16m2_u32m1 (a1, seed, vl));                \
  pSad[2] = (int32_t) __riscv_vmv_x_s_u32m1_u32 (                              \
              __riscv_vwredsumu_vs_u16m2_u32m1 (a2, seed, vl));                \
  pSad[3] = (int32_t) __riscv_vmv_x_s_u32m1_u32 (                              \
              __riscv_vwredsumu_vs_u16m2_u32m1 (a3, seed, vl));                \
}

SAD_FOUR_FUNC (16x16, 16, 16)
SAD_FOUR_FUNC (16x8,  16,  8)
SAD_FOUR_FUNC (8x16,   8, 16)
SAD_FOUR_FUNC (8x8,    8,  8)
SAD_FOUR_FUNC (8x4,    8,  4)
SAD_FOUR_FUNC (4x8,    4,  8)
SAD_FOUR_FUNC (4x4,    4,  4)

#endif /* HAVE_RVV */
