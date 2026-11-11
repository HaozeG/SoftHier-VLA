/* appended after the verbatim float_to_fp16 / fp16_to_float of light_redmule.cpp.
 * stdout: 65536 x float32 (fp16_to_float of every pattern), then records {u32 float bits, u16 result, u16 pad} */
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1664525u + 1013904223u; return lcg; }
static void rec(uint32_t b) {
    float f; memcpy(&f, &b, 4);
    uint16_t r = float_to_fp16(f), pad = 0;
    fwrite(&b, 4, 1, stdout); fwrite(&r, 2, 1, stdout); fwrite(&pad, 2, 1, stdout);
}
int main(void) {
    for (uint32_t h = 0; h < 65536; h++) { float f = fp16_to_float((fp16)h); fwrite(&f, 4, 1, stdout); }
    /* dense sweep: float32 exponents 2^-30 .. 2^17; random mantissas plus grid / tie / near-tie mantissas */
    for (int e = 97; e <= 144; e++)
        for (int sgn = 0; sgn < 2; sgn++)
            for (int i = 0; i < 6000; i++) {
                uint32_t m = rnd() & 0x7FFFFF;
                switch (i % 6) {
                case 1: m &= ~0x1FFFu; break;                     /* exact fp16 grid (normal) */
                case 2: m = (m & ~0x1FFFu) | 0x1000; break;       /* normal-range tie */
                case 3: m = (m & ~0x3FFFu) | 0x2000; break;       /* subnormal-range tie */
                case 4: m = (m & ~0x1FFFu) | 0x0FFF; break;       /* just below tie */
                case 5: m = (m & ~0x1FFFu) | 0x1001; break;       /* just above tie */
                }
                rec(((uint32_t)sgn << 31) | ((uint32_t)e << 23) | m);
            }
    uint32_t edge[] = {0x00000000, 0x80000000, 0x00000001, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000,
        0x33800000 /*2^-24*/, 0x33000000 /*2^-25*/, 0x33000001, 0x337FFFFF, 0x387FC000 /*max sub*/,
        0x38800000 /*min normal*/, 0x477FE000 /*65504*/, 0x477FF000 /*65520*/, 0x47800000 /*65536*/,
        0x7F7FFFFF, 0x00800000, 0x80000001, 0x34000000 /*2^-23*/};
    for (unsigned i = 0; i < sizeof edge / 4; i++) rec(edge[i]);
    return 0;
}
