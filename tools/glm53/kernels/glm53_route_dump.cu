// glm53_route_dump.cu - one-CTA diagnostic dump of the MoE v2 route outputs.
// Present only in GLM53_ROUTE_DUMP=1 manifests: one line per op call from
// ep rank 0 (device printf -> kern-serve stdout):
//   RDUMP k=<0 decode|1 verify> np=<n_post> rows=<rows> ids=<rows*9 ints>
// ids are raw top-9 slots: 0..287 routed experts, 288 always-on sink, -1
// for invalid rows. n_post = 32 * (total 32-pair align tiles), global.
__device__ static int rd_dec(char* out, int v) {
    int n = 0;
    if (v < 0) { out[n++] = 45; v = -v; }
    char tmp[10]; int m = 0;
    if (v == 0) tmp[m++] = 48;
    while (v) { tmp[m++] = (char)(48 + v % 10); v /= 10; }
    while (m) out[n++] = tmp[--m];
    return n;
}

extern "C" __global__ __launch_bounds__(32)
void glm53_route_dump(const int* __restrict__ ids, const int* __restrict__ n_post,
                      int rows, int kind, int rank_) {
    if (rank_ != 0 || threadIdx.x != 0) return;
    char line[1536];
    int p = 0;
    const char* hdr = "RDUMP k=";
    for (const char* s = hdr; *s; ++s) line[p++] = *s;
    p += rd_dec(line + p, kind);
    const char* m1 = " np=";
    for (const char* s = m1; *s; ++s) line[p++] = *s;
    p += rd_dec(line + p, n_post[0]);
    const char* m2 = " rows=";
    for (const char* s = m2; *s; ++s) line[p++] = *s;
    p += rd_dec(line + p, rows);
    if (rows > 32) rows = 32;  // line buffer bound; verify path is <=32
    if (rows < 0) rows = 0;
    const int n = rows * 9;
    for (int i = 0; i < n; ++i) {
        line[p++] = 32;
        p += rd_dec(line + p, ids[i]);
    }
    line[p] = 0;
    printf("%s\n", line);
}
