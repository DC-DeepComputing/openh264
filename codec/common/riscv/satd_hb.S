/*  Hand-written RVV SATD family for openh264 (iteration 4: boundary squeeze).
 *  dav1d-style batched full-width kernels; this round adds:
 *    - merged rodata tables (idx + masks): one la + 4 vle16 replace
 *      2 la + 2 vle + vid + 2x(li+vand+vmseq)  -> -7 ops per kernel
 *    - abs-fold identity |x+y|+|x-y| = 2*max(|x|,|y|) applied at the LAST
 *      linear stage (vertical stage 2): maxes are computed directly from the
 *      c-pair vectors, the four out vectors are never materialised,
 *      and since sum|out| = 2*sum(pairmax), (sum+1)>>1 == sum(pairmax):
 *      the addi/srai tail vanishes entirely.
 *    - 8x8 reordered to horizontal-first so the fold lands on the free
 *      vertical stage (cross-register), saving ~10 ops vs folding at
 *      horizontal stage 2; vredsum result is returned directly (no e32
 *      switch, no rounding shift).
 *
 *  Horizontal stage result per row matches WelsSampleSatd4x4_c lane-for-lane
 *  (stage 1 pairs (0,2),(1,3); stage 2 pairs (0,1),(2,3); vmerge picks
 *  sums on mask-1 lanes, diffs on mask-0 lanes).
 *  Vertical produces {c0+c1, c0-c1, c2+c3, c2-c3} — same set as _c; the fold
 *  takes max(|c0|,|c1|) + max(|c2|,|c3|) per lane which equals (sum|out|)/2.
 */

    .section .rodata
    .align 4
/* NOTE: v0 used as a vmerge mask is read BIT-PER-ELEMENT (bit i of the raw
 * register gates element i). Tables loaded with vle16 must therefore pack
 * the mask bits into element 0, NOT as one 0/1 value per lane. */
.Lsatd_tab:              /* 4-lane tables for the single 4x4 kernel */
    .half 2, 3, 0, 1     /* s1 idx: pairs (0,2),(1,3) */
    .half 1, 0, 3, 2     /* s2 idx: pairs (0,1),(2,3) */
    .half 0x0003, 0, 0, 0 /* m1: bits [1,1,0,0] */
    .half 0x0005, 0, 0, 0 /* m2: bits [1,0,1,0] */
.Lsatd8_tab:             /* 16-lane tables for all batched kernels */
    .half 2, 3, 0, 1,  6, 7, 4, 5,  10, 11, 8, 9,  14, 15, 12, 13
    .half 1, 0, 3, 2,  5, 4, 7, 6,   9, 8, 11, 10,  13, 12, 15, 14
    .half 0x3333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /* m1: [1,1,0,0]x4 */
    .half 0x5555, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /* m2: [1,0,1,0]x4 */

    .text

/* int32_t WelsSampleSatd4x4_rvv_hb(uint8_t *s1, int s1stride,
 *                                  uint8_t *s2, int s2stride) */
    .globl WelsSampleSatd4x4_rvv_hb
    .type  WelsSampleSatd4x4_rvv_hb, @function
WelsSampleSatd4x4_rvv_hb:
    /* ---- load 4 rows of each block (stride-aware, 4 bytes per row) ---- */
    li      t1, 4
    vsetvli t1, t1, e8, m1, ta, ma
    add     t0, a0, a1           /* src row1 */
    slli    t1, a1, 1            /* 2*s1 */
    add     t2, a0, t1           /* src row2 */
    add     t3, t0, t1           /* src row3 */
    vle8.v  v0, (a0)
    vle8.v  v1, (t0)
    vle8.v  v2, (t2)
    vle8.v  v3, (t3)
    add     t0, a2, a3           /* ref row1 */
    slli    t1, a3, 1            /* 2*s2 */
    add     t2, a2, t1           /* ref row2 */
    add     t3, t0, t1           /* ref row3 */
    vle8.v  v4, (a2)
    vle8.v  v5, (t0)
    vle8.v  v6, (t2)
    vle8.v  v7, (t3)

    /* ---- residual rows, byte -> halfword widening subtract ---- */
    vwsubu.vv v16, v0, v4        /* row0 -> v16 (m2 pair v16:v17) */
    vwsubu.vv v18, v1, v5        /* row1 -> v18 */
    vwsubu.vv v20, v2, v6        /* row2 -> v20 */
    vwsubu.vv v22, v3, v7        /* row3 -> v22 */

    li      t1, 4                 /* reload vl (t1 was clobbered as scratch) */
    vsetvli t1, t1, e16, m1, ta, ma

    /* ---- merged tables: s1@0 s2@8 m1@16 m2@24 ---- */
    la      t0, .Lsatd_tab
    vle16.v v8, (t0)              /* s1 idx */
    addi    t0, t0, 8
    vle16.v v11, (t0)             /* s2 idx */
    addi    t0, t0, 8
    vle16.v v0, (t0)              /* stage-1 mask */

    /* ---- horizontal stage 1 ---- */
    vrgather.vv v14, v16, v8
    vadd.vv  v15, v16, v14       /* sums  [s02, s13, s02, s13] */
    vsub.vv  v16, v14, v16       /* diffs [d02, d13, d02, d13] pattern */
    vmerge.vvm v16, v16, v15, v0 /* row0 = [s02, s13, d02, d13] */
    vrgather.vv v14, v18, v8
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v8
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v8
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* ---- horizontal stage 2 ---- */
    addi    t0, t0, 8
    vle16.v v0, (t0)              /* stage-2 mask */
    vrgather.vv v14, v16, v11
    vadd.vv  v15, v16, v14
    vsub.vv  v16, v14, v16
    vmerge.vvm v16, v16, v15, v0 /* row0 = [h0, h2, h1, h3] (set == _c's) */
    vrgather.vv v14, v18, v11
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v11
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v11
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* ---- vertical stage 1 (register add/sub only) ----
     * c0 = r0+r2, c1 = r1+r3, c2 = r0-r2, c3 = r1-r3 */
    vadd.vv v14, v16, v20         /* c0 */
    vadd.vv v15, v18, v22        /* c1 */
    vsub.vv v12, v16, v20        /* c2 */
    vsub.vv v16, v18, v22        /* c3 (v16 free now) */

    /* ---- abs-fold at the last linear stage ----
     * sum|{c0+c1, c0-c1, c2+c3, c2-c3}| = 2*(max(|c0|,|c1|)+max(|c2|,|c3|))
     * result = sum|out| / 2 = sum(pairmax) -- no rounding shift needed */
    vneg.v  v18, v14
    vmax.vv v18, v14, v18         /* |c0| */
    vneg.v  v19, v15
    vmax.vv v19, v15, v19         /* |c1| */
    vmax.vv v18, v18, v19         /* M1 = max(|c0|,|c1|) */
    vneg.v  v19, v12
    vmax.vv v19, v12, v19         /* |c2| */
    vneg.v  v15, v16
    vmax.vv v15, v16, v15         /* |c3| */
    vmax.vv v19, v19, v15         /* M2 = max(|c2|,|c3|) */
    vadd.vv v18, v18, v19         /* acc = M1+M2 */

    vmv.v.i v14, 0
    vredsum.vs v14, v18, v14     /* vector = v18, seed = 0 */
    vmv.x.s a0, v14
    ret
    .size WelsSampleSatd4x4_rvv_hb, .-WelsSampleSatd4x4_rvv_hb

/* int32_t WelsSampleSatd8x8_rvv_hb(uint8_t *s1, int s1stride,
 *                                  uint8_t *s2, int s2stride)
 *
 * Batched full-width (dav1d style): one 16-lane register holds the same
 * relative row of all four 4x4 sub-blocks:
 *   F0 = [r0(8B) | r4(8B)], F1 = [r1|r5], F2 = [r2|r6], F3 = [r3|r7]
 * lane layout: [blk(0,0) | blk(0,4) | blk(4,0) | blk(4,4)] per 4-lane group.
 * HORIZONTAL FIRST (reordered this round) so the abs-fold lands on the
 * free vertical stage 2; vertical stage 1 is cross-register add/sub. */
    .globl WelsSampleSatd8x8_rvv_hb
    .type  WelsSampleSatd8x8_rvv_hb, @function
WelsSampleSatd8x8_rvv_hb:
    /* ---- load 8 rows of src and ref (8 bytes per row) ---- */
    li      a4, 8
    vsetvli a4, a4, e8, m1, ta, ma
    add     t0, a0, a1           /* src r1 */
    slli    t1, a1, 1            /* 2*s1 */
    add     t2, a0, t1           /* src r2 */
    add     t3, t0, t1           /* src r3 */
    slli    t4, t1, 1            /* 4*s1 */
    add     t5, a0, t4           /* src r4 */
    add     t6, t0, t4           /* src r5 */
    vle8.v  v0, (a0)
    vle8.v  v1, (t0)
    vle8.v  v2, (t2)
    vle8.v  v3, (t3)
    vle8.v  v4, (t5)
    vle8.v  v5, (t6)
    add     t2, t5, t1           /* src r6 */
    add     t3, t6, t1           /* src r7 */
    vle8.v  v6, (t2)
    vle8.v  v7, (t3)
    add     t0, a2, a3           /* ref r1 */
    slli    t1, a3, 1            /* 2*s2 */
    add     t2, a2, t1           /* ref r2 */
    add     t3, t0, t1           /* ref r3 */
    slli    t4, t1, 1            /* 4*s2 */
    add     t5, a2, t4           /* ref r4 */
    add     t6, t0, t4           /* ref r5 */
    vle8.v  v8, (a2)
    vle8.v  v9, (t0)
    vle8.v  v10, (t2)
    vle8.v  v11, (t3)
    vle8.v  v12, (t5)
    vle8.v  v13, (t6)
    add     t2, t5, t1           /* ref r6 */
    add     t3, t6, t1           /* ref r7 */
    vle8.v  v14, (t2)
    vle8.v  v15, (t3)

    /* ---- pack row pairs: v0 = [r0 | r4], ... (16 bytes, full width) ---- */
    li      a4, 16
    vsetvli a4, a4, e8, m1, ta, ma
    vslideup.vi v0, v4, 8        /* [r0|r4] */
    vslideup.vi v1, v5, 8        /* [r1|r5] */
    vslideup.vi v2, v6, 8        /* [r2|r6] */
    vslideup.vi v3, v7, 8        /* [r3|r7] */
    vslideup.vi v8, v12, 8
    vslideup.vi v9, v13, 8
    vslideup.vi v10, v14, 8
    vslideup.vi v11, v15, 8

    /* ---- residual, widened to 16 lanes e16 (full width) ---- */
    vwsubu.vv v16, v0, v8        /* F0 = [r0-r0' | r4-r4'] */
    vwsubu.vv v18, v1, v9        /* F1 = [r1|r5] */
    vwsubu.vv v20, v2, v10       /* F2 = [r2|r6] */
    vwsubu.vv v22, v3, v11       /* F3 = [r3|r7] */

    li      a4, 16
    vsetvli a4, a4, e16, m1, ta, ma

    /* ---- merged tables: s1@0 s2@32 m1@64 m2@96 ---- */
    la      t0, .Lsatd8_tab
    vle16.v v10, (t0)            /* stage-1 idx (per-block [2,3,0,1]) */
    addi    t0, t0, 32
    vle16.v v11, (t0)            /* stage-2 idx (per-block [1,0,3,2]) */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-1 mask */

    /* ---- horizontal stage 1 (per 4-lane block group) ---- */
    vrgather.vv v8, v16, v10
    vadd.vv  v9, v16, v8
    vsub.vv  v16, v8, v16
    vmerge.vvm v16, v16, v9, v0
    vrgather.vv v8, v18, v10
    vadd.vv  v9, v18, v8
    vsub.vv  v18, v8, v18
    vmerge.vvm v18, v18, v9, v0
    vrgather.vv v8, v20, v10
    vadd.vv  v9, v20, v8
    vsub.vv  v20, v8, v20
    vmerge.vvm v20, v20, v9, v0
    vrgather.vv v8, v22, v10
    vadd.vv  v9, v22, v8
    vsub.vv  v22, v8, v22
    vmerge.vvm v22, v22, v9, v0

    /* ---- horizontal stage 2 ---- */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-2 mask */
    vrgather.vv v8, v16, v11
    vadd.vv  v9, v16, v8
    vsub.vv  v16, v8, v16
    vmerge.vvm v16, v16, v9, v0
    vrgather.vv v8, v18, v11
    vadd.vv  v9, v18, v8
    vsub.vv  v18, v8, v18
    vmerge.vvm v18, v18, v9, v0
    vrgather.vv v8, v20, v11
    vadd.vv  v9, v20, v8
    vsub.vv  v20, v8, v20
    vmerge.vvm v20, v20, v9, v0
    vrgather.vv v8, v22, v11
    vadd.vv  v9, v22, v8
    vsub.vv  v22, v8, v22
    vmerge.vvm v22, v22, v9, v0

    /* ---- vertical stage 1 (free: cross-register add/sub) ----
     * per block: c0=r0+r2, c1=r1+r3, c2=r0-r2, c3=r1-r3 */
    vadd.vv v24, v16, v20        /* c0 */
    vadd.vv v26, v18, v22        /* c1 */
    vsub.vv v28, v16, v20        /* c2 */
    vsub.vv v30, v18, v22        /* c3 */

    /* ---- abs-fold at vertical stage 2 ----
     * result = sum over lanes of max(|c0|,|c1|)+max(|c2|,|c3|)
     * acc <= 4 blocks * 16320/... sum|out|/2 <= 32640, fits e16 */
    vneg.v  v8, v24
    vmax.vv v8, v24, v8          /* |c0| */
    vneg.v  v9, v26
    vmax.vv v9, v26, v9          /* |c1| */
    vmax.vv v8, v8, v9           /* M1 */
    vneg.v  v9, v28
    vmax.vv v9, v28, v9          /* |c2| */
    vneg.v  v16, v30
    vmax.vv v16, v30, v16        /* |c3| */
    vmax.vv v9, v9, v16          /* M2 */
    vadd.vv v8, v8, v9           /* acc = M1 + M2 */

    vmv.v.i v14, 0
    vredsum.vs v14, v8, v14     /* vector = v8, seed = 0 */
    vmv.x.s a0, v14
    ret
    .size WelsSampleSatd8x8_rvv_hb, .-WelsSampleSatd8x8_rvv_hb

/* ---- compound sizes: reuse the batched kernels ----
 * Common prologue/epilogue; s1 = src base, s2 = ref base, s0 = sum. */

.macro SATD_COMPOUND_PROLOGUE
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0, 16(sp)
    sd      s1, 8(sp)
    sd      s2, 0(sp)
    mv      s0, zero
    mv      s1, a0
    mv      s2, a2
.endm

.macro SATD_COMPOUND_EPILOGUE
    mv      t0, s0              /* sum out before restoring caller s0 */
    ld      ra, 24(sp)
    ld      s0, 16(sp)
    ld      s1, 8(sp)
    ld      s2, 0(sp)
    addi    sp, sp, 32
    mv      a0, t0
    ret
.endm

.macro SATD_CALL4 SUB, OFF
    .if \OFF == 0
    call    \SUB
    .else
    addi    a0, s1, \OFF
    addi    a2, s2, \OFF
    call    \SUB
    .endif
    add     s0, s0, a0
.endm

/* 8x4: batched. Two 4x4 blocks side-by-side; rows r0..r3 are 8 bytes each
 * (block A in lanes 0-3, block B in lanes 4-7 of each row register).
 * Horizontal = per-4-lane-group in-register gathers (vl=8);
 * vertical stage 1 + abs-fold are FREE cross-register max/add. */
    .globl WelsSampleSatd8x4_rvv_hb
    .type  WelsSampleSatd8x4_rvv_hb, @function
WelsSampleSatd8x4_rvv_hb:
    li      a4, 8
    vsetvli a4, a4, e8, m1, ta, ma
    add     t0, a0, a1           /* src r1 */
    slli    t1, a1, 1
    add     t2, a0, t1           /* src r2 */
    add     t3, t0, t1           /* src r3 */
    vle8.v  v0, (a0)
    vle8.v  v1, (t0)
    vle8.v  v2, (t2)
    vle8.v  v3, (t3)
    add     t0, a2, a3           /* ref r1 */
    slli    t1, a3, 1
    add     t2, a2, t1           /* ref r2 */
    add     t3, t0, t1           /* ref r3 */
    vle8.v  v8, (a2)
    vle8.v  v9, (t0)
    vle8.v  v10, (t2)
    vle8.v  v11, (t3)

    vwsubu.vv v16, v0, v8        /* row0 residual (8 halfword lanes) */
    vwsubu.vv v18, v1, v9
    vwsubu.vv v20, v2, v10
    vwsubu.vv v22, v3, v11

    li      a4, 8
    vsetvli a4, a4, e16, m1, ta, ma
    la      t0, .Lsatd8_tab      /* vl=8 uses front half of each table */
    vle16.v v8, (t0)             /* stage-1 idx */
    addi    t0, t0, 32
    vle16.v v10, (t0)            /* stage-2 idx */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-1 mask */

    /* stage 1 */
    vrgather.vv v14, v16, v8
    vadd.vv  v15, v16, v14
    vsub.vv  v16, v14, v16
    vmerge.vvm v16, v16, v15, v0
    vrgather.vv v14, v18, v8
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v8
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v8
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* stage 2 */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-2 mask */
    vrgather.vv v14, v16, v10
    vadd.vv  v15, v16, v14
    vsub.vv  v16, v14, v16
    vmerge.vvm v16, v16, v15, v0
    vrgather.vv v14, v18, v10
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v10
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v10
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* vertical stage 1: FREE cross-register butterflies */
    vadd.vv v24, v16, v20        /* c0 = r0+r2 */
    vadd.vv v26, v18, v22        /* c1 = r1+r3 */
    vsub.vv v28, v16, v20        /* c2 = r0-r2 */
    vsub.vv v30, v18, v22        /* c3 = r1-r3 */

    /* abs-fold at vertical stage 2: acc <= 2x16320/2 fits e16 */
    vneg.v  v8, v24
    vmax.vv v8, v24, v8          /* |c0| */
    vneg.v  v9, v26
    vmax.vv v9, v26, v9          /* |c1| */
    vmax.vv v8, v8, v9           /* M1 */
    vneg.v  v9, v28
    vmax.vv v9, v28, v9          /* |c2| */
    vneg.v  v16, v30
    vmax.vv v16, v30, v16        /* |c3| */
    vmax.vv v9, v9, v16          /* M2 */
    vadd.vv v8, v8, v9           /* acc */

    vmv.v.i v14, 0
    vredsum.vs v14, v8, v14
    vmv.x.s a0, v14
    ret
    .size WelsSampleSatd8x4_rvv_hb, .-WelsSampleSatd8x4_rvv_hb

/* 4x8: batched. Two 4x4 blocks stacked vertically; register Fi = [r_i | r_{i+4}]
 * (8 lanes: block A row i in lanes 0-3, block B row i in lanes 4-7).
 * Horizontal = per-group in-register gathers; vertical + fold FREE. */
    .globl WelsSampleSatd4x8_rvv_hb
    .type  WelsSampleSatd4x8_rvv_hb, @function
WelsSampleSatd4x8_rvv_hb:
    li      a4, 4
    vsetvli a4, a4, e8, m1, ta, ma
    add     t0, a0, a1           /* src r1 */
    slli    t1, a1, 1            /* 2*s1 */
    add     t2, a0, t1           /* src r2 */
    add     t3, t0, t1           /* src r3 */
    slli    t4, t1, 1            /* 4*s1 */
    add     t5, a0, t4           /* src r4 */
    add     t6, t0, t4           /* src r5 */
    vle8.v  v0, (a0)
    vle8.v  v1, (t0)
    vle8.v  v2, (t2)
    vle8.v  v3, (t3)
    vle8.v  v4, (t5)
    vle8.v  v5, (t6)
    add     t2, t5, t1           /* src r6 */
    add     t3, t6, t1           /* src r7 */
    vle8.v  v6, (t2)
    vle8.v  v7, (t3)
    add     t0, a2, a3           /* ref r1 */
    slli    t1, a3, 1
    add     t2, a2, t1           /* ref r2 */
    add     t3, t0, t1           /* ref r3 */
    slli    t4, t1, 1
    add     t5, a2, t4           /* ref r4 */
    add     t6, t0, t4           /* ref r5 */
    vle8.v  v8, (a2)
    vle8.v  v9, (t0)
    vle8.v  v10, (t2)
    vle8.v  v11, (t3)
    vle8.v  v12, (t5)
    vle8.v  v13, (t6)
    add     t2, t5, t1           /* ref r6 */
    add     t3, t6, t1           /* ref r7 */
    vle8.v  v14, (t2)
    vle8.v  v15, (t3)

    /* pack row pairs: v0=[r0|r4], v1=[r1|r5], v2=[r2|r6], v3=[r3|r7] */
    li      a4, 8
    vsetvli a4, a4, e8, m1, ta, ma
    vslideup.vi v0, v4, 4
    vslideup.vi v1, v5, 4
    vslideup.vi v2, v6, 4
    vslideup.vi v3, v7, 4
    vslideup.vi v8, v12, 4
    vslideup.vi v9, v13, 4
    vslideup.vi v10, v14, 4
    vslideup.vi v11, v15, 4

    vwsubu.vv v16, v0, v8        /* F0 = [r0-r0' | r4-r4'] */
    vwsubu.vv v18, v1, v9        /* F1 = [r1|r5] */
    vwsubu.vv v20, v2, v10       /* F2 = [r2|r6] */
    vwsubu.vv v22, v3, v11       /* F3 = [r3|r7] */

    li      a4, 8
    vsetvli a4, a4, e16, m1, ta, ma
    la      t0, .Lsatd8_tab
    vle16.v v8, (t0)             /* stage-1 idx */
    addi    t0, t0, 32
    vle16.v v10, (t0)            /* stage-2 idx */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-1 mask */

    /* stage 1 */
    vrgather.vv v14, v16, v8
    vadd.vv  v15, v16, v14
    vsub.vv  v16, v14, v16
    vmerge.vvm v16, v16, v15, v0
    vrgather.vv v14, v18, v8
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v8
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v8
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* stage 2 */
    addi    t0, t0, 32
    vle16.v v0, (t0)             /* stage-2 mask */
    vrgather.vv v14, v16, v10
    vadd.vv  v15, v16, v14
    vsub.vv  v16, v14, v16
    vmerge.vvm v16, v16, v15, v0
    vrgather.vv v14, v18, v10
    vadd.vv  v15, v18, v14
    vsub.vv  v18, v14, v18
    vmerge.vvm v18, v18, v15, v0
    vrgather.vv v14, v20, v10
    vadd.vv  v15, v20, v14
    vsub.vv  v20, v14, v20
    vmerge.vvm v20, v20, v15, v0
    vrgather.vv v14, v22, v10
    vadd.vv  v15, v22, v14
    vsub.vv  v22, v14, v22
    vmerge.vvm v22, v22, v15, v0

    /* vertical stage 1: FREE (F0/F2 = rows (0,2) of both blocks, etc.) */
    vadd.vv v24, v16, v20        /* c0 = r0+r2 (blkA) | r4+r6 (blkB) */
    vadd.vv v26, v18, v22        /* c1 = r1+r3 | r5+r7 */
    vsub.vv v28, v16, v20        /* c2 */
    vsub.vv v30, v18, v22        /* c3 */

    /* abs-fold at vertical stage 2 */
    vneg.v  v8, v24
    vmax.vv v8, v24, v8          /* |c0| */
    vneg.v  v9, v26
    vmax.vv v9, v26, v9          /* |c1| */
    vmax.vv v8, v8, v9           /* M1 */
    vneg.v  v9, v28
    vmax.vv v9, v28, v9          /* |c2| */
    vneg.v  v16, v30
    vmax.vv v16, v30, v16        /* |c3| */
    vmax.vv v9, v9, v16          /* M2 */
    vadd.vv v8, v8, v9           /* acc */

    vmv.v.i v14, 0
    vredsum.vs v14, v8, v14
    vmv.x.s a0, v14
    ret
    .size WelsSampleSatd4x8_rvv_hb, .-WelsSampleSatd4x8_rvv_hb

/* 16x8 = 8x8 + 8x8 (+8 cols) */
    .globl WelsSampleSatd16x8_rvv_hb
    .type  WelsSampleSatd16x8_rvv_hb, @function
WelsSampleSatd16x8_rvv_hb:
    SATD_COMPOUND_PROLOGUE
    SATD_CALL4 WelsSampleSatd8x8_rvv_hb, 0
    SATD_CALL4 WelsSampleSatd8x8_rvv_hb, 8
    SATD_COMPOUND_EPILOGUE
    .size WelsSampleSatd16x8_rvv_hb, .-WelsSampleSatd16x8_rvv_hb

/* 8x16 = 8x8 + 8x8 (+8 rows) */
    .globl WelsSampleSatd8x16_rvv_hb
    .type  WelsSampleSatd8x16_rvv_hb, @function
WelsSampleSatd8x16_rvv_hb:
    SATD_COMPOUND_PROLOGUE
    call    WelsSampleSatd8x8_rvv_hb
    add     s0, s0, a0
    slli    t1, a1, 3
    add     a0, s1, t1
    slli    t1, a3, 3
    add     a2, s2, t1
    call    WelsSampleSatd8x8_rvv_hb
    add     s0, s0, a0
    SATD_COMPOUND_EPILOGUE
    .size WelsSampleSatd8x16_rvv_hb, .-WelsSampleSatd8x16_rvv_hb

/* 16x16 = 4x 8x8 */
    .globl WelsSampleSatd16x16_rvv_hb
    .type  WelsSampleSatd16x16_rvv_hb, @function
WelsSampleSatd16x16_rvv_hb:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    sd      s0, 32(sp)
    sd      s1, 24(sp)
    sd      s2, 16(sp)
    sd      s3, 8(sp)
    mv      s0, zero
    mv      s1, a0
    mv      s2, a2
    SATD_CALL4 WelsSampleSatd8x8_rvv_hb, 0
    SATD_CALL4 WelsSampleSatd8x8_rvv_hb, 8
    slli    t1, a1, 3
    add     s3, s1, t1           /* lower-half src base (survives calls) */
    slli    t1, a3, 3
    add     t2, s2, t1           /* lower-half ref base, used immediately */
    mv      a0, s3
    mv      a2, t2
    call    WelsSampleSatd8x8_rvv_hb
    add     s0, s0, a0
    addi    a0, s3, 8
    slli    t1, a3, 3
    add     t2, s2, t1
    addi    a2, t2, 8
    call    WelsSampleSatd8x8_rvv_hb
    add     s0, s0, a0
    mv      t0, s0
    ld      ra, 40(sp)
    ld      s0, 32(sp)
    ld      s1, 24(sp)
    ld      s2, 16(sp)
    ld      s3, 8(sp)
    addi    sp, sp, 48
    mv      a0, t0
    ret
    .size WelsSampleSatd16x16_rvv_hb, .-WelsSampleSatd16x16_rvv_hb
