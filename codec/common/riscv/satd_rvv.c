/*
 * RVV 4x4 SATD kernel for openh264 encoder (RD cost). gprof: 占编码时间 32.1%。
 *
 * 参考 C 语义（codec/encoder/core/src/sample.cpp WelsSampleSatd4x4_c）：
 *   diff -> 行方向 4 点 Hadamard -> 列方向 4 点 Hadamard -> sum(abs) -> (sum+1)>>1
 *
 * 设计要点：
 *  1) C 版用 int32_t pSampleMix[4][4] 中转，栈访存开销大；向量版整个 4x4 块
 *     放在一个 i16 向量里（VLEN=256 刚好 16 lane）。
 *  2) C 版蝶形先 (0,2)(1,3) 再 (0,1)(2,3)，等价于标准 H4 的行置换；
 *     最后只取 abs 之和，行置换不改变结果 → 可用最简单的邻位蝶形。
 *  3) 行变换 = 组内 dist1/dist2；列变换 = 跳行 dist4/dist8；
 *     index 用 vid^dist 现算（不走内存），蝶形 = gather+add+sub+merge 四条。
 *  4) 中间值上界：|diff|<=255 → stage1<=510 → stage2<=1020 → 两趟<=4080，i16 安全。
 */

#include <stdint.h>
#include "typedefs.h"

#if defined(HAVE_RVV)
#include <riscv_vector.h>

/* 本文件与 encoder 的 sample.h 不在同一 include 路径下，局部声明即可 */
int32_t WelsSampleSatd4x4_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd8x8_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd16x16_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd16x8_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd8x16_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd8x4_rvv (uint8_t*, int32_t, uint8_t*, int32_t);
int32_t WelsSampleSatd4x8_rvv (uint8_t*, int32_t, uint8_t*, int32_t);

#define SATD_VL 16

/* 一个 stage：偶数 lane 取和、奇数 lane 取差（以 dist 为配对距离） */
static inline vint16m2_t satd_stage (vint16m2_t v, vuint16m2_t idx,
                                     vbool8_t even) {
  vint16m2_t r = __riscv_vrgather_vv_i16m2 (v, idx, SATD_VL);
  vint16m2_t s = __riscv_vadd_vv_i16m2 (v, r, SATD_VL);   /* even: a+b */
  vint16m2_t d = __riscv_vsub_vv_i16m2 (r, v, SATD_VL);   /* odd : a-b */
  return __riscv_vmerge_vvm_i16m2 (d, s, even, SATD_VL);
}

int32_t WelsSampleSatd4x4_rvv (uint8_t* pSample1, int32_t iStride1,
                               uint8_t* pSample2, int32_t iStride2) {
  /* --- 1. 4 行 x 4 字节 -> 一个 16 lane 向量（行主序）--- */
  vuint8m1_t a = __riscv_vle8_v_u8m1 (pSample1, 4);
  a = __riscv_vslideup_vx_u8m1 (a, __riscv_vle8_v_u8m1 (pSample1 + iStride1, 4), 4, SATD_VL);
  a = __riscv_vslideup_vx_u8m1 (a, __riscv_vle8_v_u8m1 (pSample1 + 2 * iStride1, 4), 8, SATD_VL);
  a = __riscv_vslideup_vx_u8m1 (a, __riscv_vle8_v_u8m1 (pSample1 + 3 * iStride1, 4), 12, SATD_VL);

  vuint8m1_t b = __riscv_vle8_v_u8m1 (pSample2, 4);
  b = __riscv_vslideup_vx_u8m1 (b, __riscv_vle8_v_u8m1 (pSample2 + iStride2, 4), 4, SATD_VL);
  b = __riscv_vslideup_vx_u8m1 (b, __riscv_vle8_v_u8m1 (pSample2 + 2 * iStride2, 4), 8, SATD_VL);
  b = __riscv_vslideup_vx_u8m1 (b, __riscv_vle8_v_u8m1 (pSample2 + 3 * iStride2, 4), 12, SATD_VL);

  vint16m2_t va = __riscv_vreinterpret_v_u16m2_i16m2 (
                    __riscv_vzext_vf2_u16m2 (a, SATD_VL));
  vint16m2_t vb = __riscv_vreinterpret_v_u16m2_i16m2 (
                    __riscv_vzext_vf2_u16m2 (b, SATD_VL));
  vint16m2_t v = __riscv_vsub_vv_i16m2 (va, vb, SATD_VL);

  /* --- 2. index 与 even mask：全部用 vid 现算，不读常量表 --- */
  vuint16m2_t iota = __riscv_vid_v_u16m2 (SATD_VL);
  vuint16m2_t i1 = __riscv_vxor_vx_u16m2 (iota, 1, SATD_VL);
  vuint16m2_t i2 = __riscv_vxor_vx_u16m2 (iota, 2, SATD_VL);
  vuint16m2_t i4 = __riscv_vxor_vx_u16m2 (iota, 4, SATD_VL);
  vuint16m2_t i8 = __riscv_vxor_vx_u16m2 (iota, 8, SATD_VL);
  /* 蝶形的“和”落在组内低半区：(lane & stride) == 0。
   * 第 1 级 stride=1（偶数 lane）、第 2 级 stride=2（lane 0,1）、
   * 第 3 级 stride=4、第 4 级 stride=8 —— 不是统一用偶数 lane mask。*/
  vbool8_t m1 = __riscv_vmseq_vx_u16m2_b8 (__riscv_vand_vx_u16m2 (iota, 1, SATD_VL), 0, SATD_VL);
  vbool8_t m2 = __riscv_vmseq_vx_u16m2_b8 (__riscv_vand_vx_u16m2 (iota, 2, SATD_VL), 0, SATD_VL);
  vbool8_t m4 = __riscv_vmseq_vx_u16m2_b8 (__riscv_vand_vx_u16m2 (iota, 4, SATD_VL), 0, SATD_VL);
  vbool8_t m8 = __riscv_vmseq_vx_u16m2_b8 (__riscv_vand_vx_u16m2 (iota, 8, SATD_VL), 0, SATD_VL);

  /* --- 3. 行变换（组内 dist1 -> dist2）--- */
  v = satd_stage (v, i1, m1);
  v = satd_stage (v, i2, m2);
  /* --- 4. 列变换（跳行 dist4 -> dist8）--- */
  v = satd_stage (v, i4, m4);
  v = satd_stage (v, i8, m8);

  /* --- 5. sum|v| -> (sum + 1) >> 1 --- */
  vint16m2_t av = __riscv_vmax_vv_i16m2 (v, __riscv_vneg_v_i16m2 (v, SATD_VL), SATD_VL);
  vuint32m1_t sum = __riscv_vwredsumu_vs_u16m2_u32m1 (
                      __riscv_vreinterpret_v_i16m2_u16m2 (av),
                      __riscv_vmv_v_x_u32m1 (0, 1), SATD_VL);
  int32_t s = (int32_t) __riscv_vmv_x_s_u32m1_u32 (sum);
  return (s + 1) >> 1;
}

/* 其它尺寸在 C 里本就是 4x4 的拼接，这里复用 RVV 4x4 */
int32_t WelsSampleSatd8x8_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  int32_t sum = 0;
  sum += WelsSampleSatd4x4_rvv (p1, s1, p2, s2);
  sum += WelsSampleSatd4x4_rvv (p1 + 4, s1, p2 + 4, s2);
  sum += WelsSampleSatd4x4_rvv (p1 + (s1 << 2), s1, p2 + (s2 << 2), s2);
  sum += WelsSampleSatd4x4_rvv (p1 + (s1 << 2) + 4, s1, p2 + (s2 << 2) + 4, s2);
  return sum;
}

int32_t WelsSampleSatd16x16_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  int32_t sum = 0;
  for (int y = 0; y < 16; y += 8)
    for (int x = 0; x < 16; x += 8)
      sum += WelsSampleSatd8x8_rvv (p1 + y * s1 + x, s1, p2 + y * s2 + x, s2);
  return sum;
}

int32_t WelsSampleSatd16x8_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  return WelsSampleSatd8x8_rvv (p1, s1, p2, s2)
       + WelsSampleSatd8x8_rvv (p1 + 8, s1, p2 + 8, s2);
}

int32_t WelsSampleSatd8x16_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  return WelsSampleSatd8x8_rvv (p1, s1, p2, s2)
       + WelsSampleSatd8x8_rvv (p1 + (s1 << 3), s1, p2 + (s2 << 3), s2);
}

int32_t WelsSampleSatd8x4_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  return WelsSampleSatd4x4_rvv (p1, s1, p2, s2)
       + WelsSampleSatd4x4_rvv (p1 + 4, s1, p2 + 4, s2);
}

int32_t WelsSampleSatd4x8_rvv (uint8_t* p1, int32_t s1, uint8_t* p2, int32_t s2) {
  return WelsSampleSatd4x4_rvv (p1, s1, p2, s2)
       + WelsSampleSatd4x4_rvv (p1 + (s1 << 2), s1, p2 + (s2 << 2), s2);
}

#endif /* HAVE_RVV */
