/*
 * RVV 1.0 vectorized intra prediction for openh264 (high-value probe).
 * Parameter estimation stays scalar (8 weighted sums; 16-element law).
 * Plane 16x16 fill vectorized:
 *   pred[i][j] = clip((a + b*(j-7) + c*(i-7) + 16) >> 5)
 *   vid supplies (j-7) for all 16 columns; row step adds c.
 * Bit-exact to WelsI16x16LumaPredPlane_c (same rounding + WelsClip1 clamp).
 */
#include <stdint.h>
#include "typedefs.h"

#if defined(HAVE_RVV)
#include <riscv_vector.h>

void WelsI16x16LumaPredPlane_rvv (uint8_t* pPred, uint8_t* pRef, int32_t stride) {
  int32_t iTopSum = 0, iLeftSum = 0;
  uint8_t* pTop = &pRef[-stride];
  uint8_t* pLeft = &pRef[-1];
  for (int i = 0; i < 8; i++) {
    iTopSum  += (i + 1) * (pTop[8 + i] - pTop[6 - i]);
    iLeftSum += (i + 1) * (pLeft[(8 + i) * stride] - pLeft[(6 - i) * stride]);
  }
  int32_t a = (pLeft[15 * stride] + pTop[15]) << 4;
  int32_t b = (5 * iTopSum + 32) >> 6;
  int32_t c = (5 * iLeftSum + 32) >> 6;

  const size_t vl = 16;
  vint16m2_t idx = __riscv_vreinterpret_v_u16m2_i16m2 (
                     __riscv_vsub_vx_u16m2 (__riscv_vid_v_u16m2 (vl), 7, vl));
  vint16m2_t jterm = __riscv_vmul_vx_i16m2 (idx, (int16_t)b, vl);
  vint16m2_t rowbase = __riscv_vmv_v_x_i16m2 ((int16_t)(a - 7 * c), vl);
  vint16m2_t zero = __riscv_vmv_v_x_i16m2 (0, vl);
  vint16m2_t hi = __riscv_vmv_v_x_i16m2 (255, vl);

  for (int i = 0; i < 16; i++) {
    vint16m2_t val = __riscv_vadd_vv_i16m2 (rowbase, jterm, vl);
    vint16m2_t plus16 = __riscv_vadd_vx_i16m2 (val, 16, vl);
    vint16m2_t shifted = __riscv_vsra_vx_i16m2 (plus16, 5, vl);
    vint16m2_t cl = __riscv_vmax_vv_i16m2 (shifted, zero, vl);
    cl = __riscv_vmin_vv_i16m2 (cl, hi, vl);
    vuint8m1_t out = __riscv_vncvt_x_x_w_u8m1 (
                       __riscv_vreinterpret_v_i16m2_u16m2 (cl), vl);
    __riscv_vse8_v_u8m1 (pPred + i * 16, out, vl);
    rowbase = __riscv_vadd_vx_i16m2 (rowbase, (int16_t)c, vl);
  }
}

#endif /* HAVE_RVV */
