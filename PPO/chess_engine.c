/*
 * Batched chess engine used by PPO/chess_env.py (compiled on first use, see there).
 *
 * Every function works on an array of independent boards and on a range [start, end) of it, so that the Python side
 * can split a batch over threads (ctypes releases the GIL): one process, one interpreter, no copies of the state.
 *
 * Squares are numbered rank * 8 + file (a1 = 0, h8 = 63). Observations and actions are always expressed from the
 * point of view of the player to move (the board is mirrored for black), so a single network can play both colours.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef uint64_t U64;

#define MAXHIST 512
#define NPLANES 21
#define NACT 4672
#define MAXMOVES 256

enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum { FL_EP = 1, FL_CASTLE = 2 };
enum { ONGOING = 0, CHECKMATE, STALEMATE, INSUFFICIENT, FIFTY, REPETITION, MAXPLIES };

typedef struct {
    U64 pc[2][6];
    U64 occ[2];
    uint8_t side;      /* 0 white, 1 black */
    uint8_t castle;    /* 1 white short, 2 white long, 4 black short, 8 black long */
    int8_t ep;         /* en passant target square (only if a pawn can capture there) or -1 */
    uint8_t pad;
    uint16_t halfmove;
} Pos;

typedef struct {
    Pos p;
    U64 rng;
    uint32_t ply;
    uint32_t nleg;
    U64 hist[MAXHIST];       /* position hashes, hist[ply] is the current one */
    uint32_t leg[MAXMOVES];  /* legal moves of the current position (cache) */
} Board;

/* move: from | to << 6 | promotion << 12 (0 none, 1 N, 2 B, 3 R, 4 Q) | flags << 15 */
#define MV(from, to, promo, fl) ((uint32_t)((from) | ((to) << 6) | ((promo) << 12) | ((fl) << 15)))

static U64 knight_att[64], king_att[64], pawn_att[2][64], ray[8][64];
static uint8_t cmask[64];
static const int DF[8] = {0, 1, 1, 1, 0, -1, -1, -1};
static const int DR[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static const int POSDIR[8] = {1, 1, 1, 0, 0, 0, 0, 1};
static const int DIRIDX[3][3] = {{5, 6, 7}, {4, -1, 0}, {3, 2, 1}}; /* [sign(df) + 1][sign(dr) + 1] */
static const int KN_DF[8] = {1, 2, 2, 1, -1, -2, -2, -1};
static const int KN_DR[8] = {2, 1, -1, -2, -2, -1, 1, 2};
static const int MATVAL[6] = {1, 3, 3, 5, 9, 0};
static const int CPVAL[6] = {100, 320, 330, 500, 900, 0};
static int tables_ready = 0;

static inline int ctz(U64 x) { return __builtin_ctzll(x); }
static inline int popcnt(U64 x) { return __builtin_popcountll(x); }

void ce_init(void) {
    if (tables_ready) return;
    for (int sq = 0; sq < 64; sq++) {
        int f = sq & 7, r = sq >> 3;
        for (int k = 0; k < 8; k++) {
            int nf = f + KN_DF[k], nr = r + KN_DR[k];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) knight_att[sq] |= 1ULL << (nr * 8 + nf);
            nf = f + DF[k]; nr = r + DR[k];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) king_att[sq] |= 1ULL << (nr * 8 + nf);
            for (nf = f + DF[k], nr = r + DR[k]; nf >= 0 && nf < 8 && nr >= 0 && nr < 8; nf += DF[k], nr += DR[k])
                ray[k][sq] |= 1ULL << (nr * 8 + nf);
        }
        for (int c = 0; c < 2; c++) {
            int nr = r + (c == 0 ? 1 : -1);
            if (nr < 0 || nr > 7) continue;
            if (f > 0) pawn_att[c][sq] |= 1ULL << (nr * 8 + f - 1);
            if (f < 7) pawn_att[c][sq] |= 1ULL << (nr * 8 + f + 1);
        }
        cmask[sq] = 15;
    }
    cmask[0] = 15 & ~2; cmask[4] = 15 & ~3; cmask[7] = 15 & ~1;
    cmask[56] = 15 & ~8; cmask[60] = 15 & ~12; cmask[63] = 15 & ~4;
    tables_ready = 1;
}

int ce_board_size(void) { return (int)sizeof(Board); }
int ce_num_planes(void) { return NPLANES; }
int ce_num_actions(void) { return NACT; }
int ce_max_hist(void) { return MAXHIST; }

static inline U64 rng_next(U64 *s) {
    U64 x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline U64 ray_att(int d, int sq, U64 occ) {
    U64 a = ray[d][sq], b = a & occ;
    if (b) a ^= ray[d][POSDIR[d] ? ctz(b) : 63 - __builtin_clzll(b)];
    return a;
}
static inline U64 bishop_att(int sq, U64 occ) {
    return ray_att(1, sq, occ) | ray_att(3, sq, occ) | ray_att(5, sq, occ) | ray_att(7, sq, occ);
}
static inline U64 rook_att(int sq, U64 occ) {
    return ray_att(0, sq, occ) | ray_att(2, sq, occ) | ray_att(4, sq, occ) | ray_att(6, sq, occ);
}

static int attacked(const Pos *p, int sq, int by) {
    U64 occ = p->occ[0] | p->occ[1];
    if (pawn_att[by ^ 1][sq] & p->pc[by][PAWN]) return 1;
    if (knight_att[sq] & p->pc[by][KNIGHT]) return 1;
    if (king_att[sq] & p->pc[by][KING]) return 1;
    if (bishop_att(sq, occ) & (p->pc[by][BISHOP] | p->pc[by][QUEEN])) return 1;
    if (rook_att(sq, occ) & (p->pc[by][ROOK] | p->pc[by][QUEEN])) return 1;
    return 0;
}

static inline int in_check(const Pos *p) {
    return attacked(p, ctz(p->pc[p->side][KING]), p->side ^ 1);
}

static inline int piece_at(const Pos *p, int c, int sq) {
    U64 b = 1ULL << sq;
    for (int pt = 0; pt < 6; pt++)
        if (p->pc[c][pt] & b) return pt;
    return -1;
}

static void make_move(Pos *p, uint32_t m) {
    int from = m & 63, to = (m >> 6) & 63, promo = (m >> 12) & 7, fl = (m >> 15) & 7;
    int us = p->side, them = us ^ 1;
    U64 fb = 1ULL << from, tb = 1ULL << to;
    int pt = piece_at(p, us, from), cap = -1;

    if (fl & FL_EP) {
        U64 cb = 1ULL << (to + (us == 0 ? -8 : 8));
        p->pc[them][PAWN] ^= cb; p->occ[them] ^= cb; cap = PAWN;
    } else if (p->occ[them] & tb) {
        cap = piece_at(p, them, to);
        p->pc[them][cap] ^= tb; p->occ[them] ^= tb;
    }
    p->pc[us][pt] ^= fb;
    p->pc[us][promo ? promo : pt] ^= tb;
    p->occ[us] ^= fb | tb;

    if (fl & FL_CASTLE) {
        int rf, rt;
        if (to == 6) { rf = 7; rt = 5; } else if (to == 2) { rf = 0; rt = 3; }
        else if (to == 62) { rf = 63; rt = 61; } else { rf = 56; rt = 59; }
        U64 rb = (1ULL << rf) | (1ULL << rt);
        p->pc[us][ROOK] ^= rb; p->occ[us] ^= rb;
    }
    p->castle &= cmask[from] & cmask[to];
    p->ep = -1;
    if (pt == PAWN && (to - from == 16 || from - to == 16)) {
        int eps = (from + to) / 2;
        if (pawn_att[us][eps] & p->pc[them][PAWN]) p->ep = (int8_t)eps;
    }
    p->halfmove = (pt == PAWN || cap >= 0) ? 0 : p->halfmove + 1;
    p->side = (uint8_t)them;
}

static inline int add_move(uint32_t *mv, int n, int from, int to, int promo, int fl) {
    mv[n] = MV(from, to, promo, fl);
    return n + 1;
}

static int gen_pseudo(const Pos *p, uint32_t *mv, int caps_only) {
    int us = p->side, them = us ^ 1, n = 0;
    U64 own = p->occ[us], opp = p->occ[them], occ = own | opp;
    int up = us == 0 ? 8 : -8, promo_rank = us == 0 ? 6 : 1, start_rank = us == 0 ? 1 : 6;

    U64 b = p->pc[us][PAWN];
    while (b) {
        int from = ctz(b); b &= b - 1;
        int rank = from >> 3, to = from + up;
        if (!((occ >> to) & 1)) {
            if (rank == promo_rank) {
                for (int pr = 4; pr >= 1; pr--) n = add_move(mv, n, from, to, pr, 0);
            } else if (!caps_only) {
                n = add_move(mv, n, from, to, 0, 0);
                if (rank == start_rank && !((occ >> (to + up)) & 1)) n = add_move(mv, n, from, to + up, 0, 0);
            }
        }
        U64 att = pawn_att[us][from] & opp;
        while (att) {
            int t = ctz(att); att &= att - 1;
            if (rank == promo_rank) {
                for (int pr = 4; pr >= 1; pr--) n = add_move(mv, n, from, t, pr, 0);
            } else {
                n = add_move(mv, n, from, t, 0, 0);
            }
        }
        if (p->ep >= 0 && ((pawn_att[us][from] >> p->ep) & 1)) n = add_move(mv, n, from, p->ep, 0, FL_EP);
    }

    U64 targets = caps_only ? opp : ~own;
    for (int pt = KNIGHT; pt <= KING; pt++) {
        b = p->pc[us][pt];
        while (b) {
            int from = ctz(b); b &= b - 1;
            U64 att;
            switch (pt) {
                case KNIGHT: att = knight_att[from]; break;
                case BISHOP: att = bishop_att(from, occ); break;
                case ROOK: att = rook_att(from, occ); break;
                case QUEEN: att = bishop_att(from, occ) | rook_att(from, occ); break;
                default: att = king_att[from]; break;
            }
            att &= targets;
            while (att) {
                int t = ctz(att); att &= att - 1;
                n = add_move(mv, n, from, t, 0, 0);
            }
        }
    }

    if (!caps_only && p->castle) {
        int base = us == 0 ? 0 : 56, sh = us == 0 ? 1 : 4, lg = us == 0 ? 2 : 8;
        if ((p->castle & sh) && !(occ & ((1ULL << (base + 5)) | (1ULL << (base + 6)))) &&
            !attacked(p, base + 4, them) && !attacked(p, base + 5, them))
            n = add_move(mv, n, base + 4, base + 6, 0, FL_CASTLE); /* the arrival square is checked by legality */
        if ((p->castle & lg) && !(occ & ((1ULL << (base + 1)) | (1ULL << (base + 2)) | (1ULL << (base + 3)))) &&
            !attacked(p, base + 4, them) && !attacked(p, base + 3, them))
            n = add_move(mv, n, base + 4, base + 2, 0, FL_CASTLE);
    }
    return n;
}

static inline int is_legal(const Pos *p, uint32_t m) {
    Pos q = *p;
    make_move(&q, m);
    return !attacked(&q, ctz(q.pc[p->side][KING]), p->side ^ 1);
}

static int gen_legal(const Pos *p, uint32_t *mv) {
    uint32_t tmp[MAXMOVES];
    int n = gen_pseudo(p, tmp, 0), k = 0;
    for (int i = 0; i < n; i++)
        if (is_legal(p, tmp[i])) mv[k++] = tmp[i];
    return k;
}

static U64 perft(const Pos *p, int depth) {
    if (depth == 0) return 1;
    uint32_t mv[MAXMOVES];
    int n = gen_legal(p, mv);
    if (depth == 1) return (U64)n;
    U64 total = 0;
    for (int i = 0; i < n; i++) {
        Pos q = *p;
        make_move(&q, mv[i]);
        total += perft(&q, depth - 1);
    }
    return total;
}

static int material(const Pos *p, int c) {
    int s = 0;
    for (int pt = 0; pt < 5; pt++) s += MATVAL[pt] * popcnt(p->pc[c][pt]);
    return s;
}

static int insufficient(const Pos *p) {
    for (int c = 0; c < 2; c++)
        if (p->pc[c][PAWN] || p->pc[c][ROOK] || p->pc[c][QUEEN]) return 0;
    int minors = 0;
    U64 bishops = 0, knights = 0;
    for (int c = 0; c < 2; c++) {
        knights |= p->pc[c][KNIGHT];
        bishops |= p->pc[c][BISHOP];
    }
    minors = popcnt(knights) + popcnt(bishops);
    if (minors <= 1) return 1;
    const U64 light = 0x55AA55AA55AA55AAULL;
    return knights == 0 && ((bishops & light) == 0 || (bishops & ~light) == 0);
}

static U64 mix(U64 h, U64 v) {
    h = (h ^ v) * 0x9E3779B97F4A7C15ULL;
    return h ^ (h >> 31);
}

static U64 pos_hash(const Pos *p) {
    U64 h = 0x243F6A8885A308D3ULL;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) h = mix(h, p->pc[c][pt] + (U64)(c * 6 + pt) * 0x1000193ULL);
    return mix(h, (U64)p->side | ((U64)p->castle << 1) | ((U64)(p->ep + 1) << 8));
}

static int rep_count(const Board *b) {
    int c = 0;
    for (uint32_t i = 2; i <= b->p.halfmove && i <= b->ply; i += 2)
        if (b->hist[b->ply - i] == b->hist[b->ply]) c++;
    return c;
}

static void refresh_legal(Board *b) { b->nleg = (uint32_t)gen_legal(&b->p, b->leg); }

static void board_apply(Board *b, uint32_t m) {
    make_move(&b->p, m);
    if (b->ply < MAXHIST - 1) b->ply++;
    b->hist[b->ply] = pos_hash(&b->p);
}

static void set_start(Board *b) {
    memset(&b->p, 0, sizeof(Pos));
    const int back[8] = {ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK};
    for (int f = 0; f < 8; f++) {
        b->p.pc[0][back[f]] |= 1ULL << f;
        b->p.pc[0][PAWN] |= 1ULL << (8 + f);
        b->p.pc[1][PAWN] |= 1ULL << (48 + f);
        b->p.pc[1][back[f]] |= 1ULL << (56 + f);
    }
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) b->p.occ[c] |= b->p.pc[c][pt];
    b->p.castle = 15; b->p.ep = -1;
    b->ply = 0;
    b->hist[0] = pos_hash(&b->p);
    refresh_legal(b);
}

/* ------------------------------------------------------------------ actions (AlphaZero style 8 x 8 x 73) */

static int move_action(int side, uint32_t m) {
    int from = m & 63, to = (m >> 6) & 63, promo = (m >> 12) & 7;
    if (side) { from ^= 56; to ^= 56; }
    int df = (to & 7) - (from & 7), dr = (to >> 3) - (from >> 3), type;
    if (promo && promo != 4) {
        type = 64 + (promo - 1) * 3 + (df + 1);
    } else if (abs(df) * abs(dr) == 2) {
        type = 56;
        for (int k = 0; k < 8; k++)
            if (KN_DF[k] == df && KN_DR[k] == dr) type = 56 + k;
    } else {
        int dist = abs(df) > abs(dr) ? abs(df) : abs(dr);
        int sf = (df > 0) - (df < 0), sr = (dr > 0) - (dr < 0);
        type = DIRIDX[sf + 1][sr + 1] * 7 + dist - 1;
    }
    return from * 73 + type;
}

int ce_encode_move(int side, int from, int to, int promo) { return move_action(side, MV(from, to, promo, 0)); }

/* ------------------------------------------------------------------ observation */

static void fill_bits(uint8_t *plane, U64 bb) {
    while (bb) { plane[ctz(bb)] = 255; bb &= bb - 1; }
}
static void fill_const(uint8_t *plane, int v) { memset(plane, v, 64); }

static void write_obs(const Board *b, int max_plies, uint8_t *out) {
    const Pos *p = &b->p;
    int us = p->side;
    memset(out, 0, NPLANES * 64);
    for (int pt = 0; pt < 6; pt++) {
        U64 mine = p->pc[us][pt], theirs = p->pc[us ^ 1][pt];
        if (us) { mine = __builtin_bswap64(mine); theirs = __builtin_bswap64(theirs); }
        fill_bits(out + pt * 64, mine);
        fill_bits(out + (6 + pt) * 64, theirs);
    }
    if (p->ep >= 0) out[12 * 64 + (us ? p->ep ^ 56 : p->ep)] = 255;
    int mine_c = us ? (p->castle >> 2) & 3 : p->castle & 3, their_c = us ? p->castle & 3 : (p->castle >> 2) & 3;
    if (mine_c & 1) fill_const(out + 13 * 64, 255);
    if (mine_c & 2) fill_const(out + 14 * 64, 255);
    if (their_c & 1) fill_const(out + 15 * 64, 255);
    if (their_c & 2) fill_const(out + 16 * 64, 255);
    fill_const(out + 17 * 64, (p->halfmove > 100 ? 100 : p->halfmove) * 255 / 100);
    int rc = rep_count(b);
    if (rc >= 1) fill_const(out + 18 * 64, 255);
    if (rc >= 2) fill_const(out + 19 * 64, 255);
    fill_const(out + 20 * 64, (int)(b->ply > (uint32_t)max_plies ? max_plies : b->ply) * 255 / max_plies);
}

static void write_mask(const Board *b, uint8_t *mask) {
    memset(mask, 0, NACT);
    for (uint32_t i = 0; i < b->nleg; i++) mask[move_action(b->p.side, b->leg[i])] = 1;
}

/* ------------------------------------------------------------------ search (baseline opponents) */

typedef struct { long nodes, limit; } Ctx;

static int eval_pos(const Pos *p) {
    int score = 0;
    for (int c = 0; c < 2; c++) {
        int s = 0;
        for (int pt = 0; pt < 5; pt++) {
            U64 b = p->pc[c][pt];
            while (b) {
                int sq = ctz(b); b &= b - 1;
                int f = sq & 7, r = c ? 7 - (sq >> 3) : sq >> 3;
                int cen = 6 - (abs(2 * f - 7) + abs(2 * (sq >> 3) - 7)) / 2;
                s += CPVAL[pt];
                switch (pt) {
                    case PAWN: s += 2 * cen + 6 * (r - 1); break;
                    case KNIGHT: s += 5 * cen; break;
                    case BISHOP: s += 3 * cen; break;
                    case QUEEN: s += cen; break;
                    default: break;
                }
            }
        }
        score += c == p->side ? s : -s;
    }
    return score;
}

static int order_moves(const Pos *p, uint32_t *mv, int n) {
    int score[MAXMOVES];
    for (int i = 0; i < n; i++) {
        int to = (mv[i] >> 6) & 63, from = mv[i] & 63, promo = (mv[i] >> 12) & 7;
        int victim = (mv[i] >> 15) & FL_EP ? PAWN : piece_at(p, p->side ^ 1, to);
        int s = 0;
        if (victim >= 0) s = 10 * CPVAL[victim] - CPVAL[piece_at(p, p->side, from)] / 10 + 1000;
        if (promo) s += CPVAL[promo];
        score[i] = s;
    }
    for (int i = 1; i < n; i++) {
        uint32_t m = mv[i]; int s = score[i], j = i - 1;
        while (j >= 0 && score[j] < s) { mv[j + 1] = mv[j]; score[j + 1] = score[j]; j--; }
        mv[j + 1] = m; score[j + 1] = s;
    }
    return n;
}

static int qsearch(const Pos *p, int alpha, int beta, Ctx *ctx) {
    ctx->nodes++;
    int stand = eval_pos(p);
    if (stand >= beta || ctx->nodes > ctx->limit) return stand;
    if (stand > alpha) alpha = stand;
    uint32_t mv[MAXMOVES];
    int n = order_moves(p, mv, gen_pseudo(p, mv, 1));
    for (int i = 0; i < n; i++) {
        Pos q = *p;
        make_move(&q, mv[i]);
        if (attacked(&q, ctz(q.pc[p->side][KING]), p->side ^ 1)) continue;
        int s = -qsearch(&q, -beta, -alpha, ctx);
        if (s >= beta) return s;
        if (s > alpha) alpha = s;
    }
    return alpha;
}

#define MATE 30000
static int negamax(const Pos *p, int depth, int alpha, int beta, int ply, Ctx *ctx) {
    int check = in_check(p);
    if (depth <= 0) {
        if (!check) return qsearch(p, alpha, beta, ctx);
        depth = 1;
    }
    ctx->nodes++;
    uint32_t mv[MAXMOVES];
    int n = order_moves(p, mv, gen_pseudo(p, mv, 0)), legal = 0;
    for (int i = 0; i < n; i++) {
        Pos q = *p;
        make_move(&q, mv[i]);
        if (attacked(&q, ctz(q.pc[p->side][KING]), p->side ^ 1)) continue;
        legal++;
        int s = (ctx->nodes > ctx->limit) ? -eval_pos(&q) : -negamax(&q, depth - 1, -beta, -alpha, ply + 1, ctx);
        if (s >= beta) return s;
        if (s > alpha) alpha = s;
    }
    if (!legal) return check ? -MATE + ply : 0;
    return alpha;
}

/* depth 0 is a uniformly random legal move, otherwise a material + light positional alpha-beta search */
static uint32_t engine_pick(Board *b, int depth, int noise) {
    if (b->nleg == 1 || depth <= 0) return b->leg[rng_next(&b->rng) % b->nleg];
    int best = -1000000, best_i = 0;
    for (uint32_t i = 0; i < b->nleg; i++) {
        Pos q = b->p;
        make_move(&q, b->leg[i]);
        Ctx ctx = {0, 200000};
        int s = -negamax(&q, depth - 1, -MATE - 1, MATE + 1, 1, &ctx);
        /* the noise randomises near-equal moves, but mates are never blurred */
        if (noise > 0 && s > -MATE / 2 && s < MATE / 2) s += (int)(rng_next(&b->rng) % (U64)(noise + 1));
        if (s > best) { best = s; best_i = (int)i; }
    }
    return b->leg[best_i];
}

/* ------------------------------------------------------------------ batched API */

#define FOR_RANGE(e, idx, start, end) for (int _i = (start); _i < (end) && ((e = (idx) ? (idx)[_i] : _i), 1); _i++)

void ce_seed(Board *boards, int e, U64 seed) {
    U64 z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    boards[e].rng = z ? z : 1;
}

static void board_reset(Board *b, int open_min, int open_max) {
    for (int attempt = 0; attempt < 32; attempt++) {
        set_start(b);
        int k = open_min + (open_max > open_min ? (int)(rng_next(&b->rng) % (U64)(open_max - open_min + 1)) : 0);
        int ok = 1;
        for (int i = 0; i < k && ok; i++) {
            if (b->nleg == 0) { ok = 0; break; }
            board_apply(b, b->leg[rng_next(&b->rng) % b->nleg]);
            refresh_legal(b);
        }
        if (ok && b->nleg > 0 && !insufficient(&b->p)) {
            return;
        }
    }
    set_start(b);
}

void ce_reset(Board *boards, const int32_t *idx, int start, int end, int open_min, int open_max) {
    int e;
    FOR_RANGE(e, idx, start, end) board_reset(&boards[e], open_min, open_max);
}

static int8_t game_status(Board *b, int max_plies) {
    if (b->nleg == 0) return in_check(&b->p) ? CHECKMATE : STALEMATE;
    if (insufficient(&b->p)) return INSUFFICIENT;
    if (b->p.halfmove >= 100) return FIFTY;
    if (rep_count(b) >= 2) return REPETITION;
    if ((int)b->ply >= max_plies || b->ply >= MAXHIST - 1) return MAXPLIES;
    return ONGOING;
}

/*
 * Potential that rewards the mating technique: when one side is ahead in material, it grows when the king of the weaker
 * side is pushed to the edge of the board and the two kings come closer. It is antisymmetric (the potential of a
 * position for one player is minus the one for the other), as needed by the zero-sum reward shaping. Range [-1, 1].
 */
static float mate_potential(const Pos *p, int us) {
    int bal = material(p, us) - material(p, us ^ 1);
    float g = bal / 5.0f;
    g = g > 1.0f ? 1.0f : (g < -1.0f ? -1.0f : g);
    int weak = bal >= 0 ? us ^ 1 : us;
    int wk = ctz(p->pc[weak][KING]), sk = ctz(p->pc[weak ^ 1][KING]);
    int edge = abs(2 * (wk & 7) - 7);
    int e2 = abs(2 * (wk >> 3) - 7);
    if (e2 > edge) edge = e2;
    int dist = abs((wk & 7) - (sk & 7)) + abs((wk >> 3) - (sk >> 3));
    return g * 0.5f * ((edge - 1) / 6.0f + (14 - dist) / 14.0f);
}

/*
 * Plays actions[e] (canonical encoding) on every board of the range.
 * outcome: -1 illegal action (the board is untouched), otherwise the game status after the move.
 * mat_before / mat_after: material balance (pawn units) from the point of view of the player who moved.
 * pot_before / pot_after: mate_potential from the same point of view.
 */
void ce_step(Board *boards, const int32_t *idx, int start, int end, const int32_t *actions, int max_plies,
             int8_t *outcome, int16_t *mat_before, int16_t *mat_after, float *pot_before, float *pot_after) {
    int e;
    FOR_RANGE(e, idx, start, end) {
        Board *b = &boards[e];
        int us = b->p.side, found = -1;
        for (uint32_t i = 0; i < b->nleg; i++)
            if (move_action(us, b->leg[i]) == actions[e]) { found = (int)i; break; }
        if (found < 0) { outcome[e] = -1; continue; }
        mat_before[e] = (int16_t)(material(&b->p, us) - material(&b->p, us ^ 1));
        pot_before[e] = mate_potential(&b->p, us);
        board_apply(b, b->leg[found]);
        pot_after[e] = mate_potential(&b->p, us);
        mat_after[e] = (int16_t)(material(&b->p, us) - material(&b->p, us ^ 1));
        refresh_legal(b);
        outcome[e] = game_status(b, max_plies);
    }
}

void ce_observe(Board *boards, const int32_t *idx, int start, int end, int max_plies, uint8_t *obs, uint8_t *mask,
                int8_t *side) {
    int e;
    FOR_RANGE(e, idx, start, end) {
        Board *b = &boards[e];
        write_obs(b, max_plies, obs + (size_t)e * NPLANES * 64);
        if (mask) write_mask(b, mask + (size_t)e * NACT);
        side[e] = (int8_t)b->p.side;
    }
}

void ce_engine_action(Board *boards, const int32_t *idx, int start, int end, int depth, int noise, int32_t *actions) {
    int e;
    FOR_RANGE(e, idx, start, end) {
        Board *b = &boards[e];
        actions[e] = b->nleg ? move_action(b->p.side, engine_pick(b, depth, noise)) : 0;
    }
}

/* ------------------------------------------------------------------ utilities for tests and the user interface */

static const char PIECE_CHARS[] = "PNBRQK";

int ce_set_fen(Board *boards, int e, const char *fen) {
    Board *b = &boards[e];
    memset(&b->p, 0, sizeof(Pos));
    int rank = 7, file = 0;
    const char *s = fen;
    for (; *s && *s != ' '; s++) {
        if (*s == '/') { rank--; file = 0; }
        else if (*s >= '1' && *s <= '8') file += *s - '0';
        else {
            int c = (*s >= 'a') ? 1 : 0;
            char u = (char)(c ? *s - 32 : *s);
            const char *q = strchr(PIECE_CHARS, u);
            if (!q || rank < 0 || file > 7) return -1;
            b->p.pc[c][q - PIECE_CHARS] |= 1ULL << (rank * 8 + file);
            file++;
        }
    }
    if (*s++ != ' ') return -1;
    b->p.side = (*s++ == 'b');
    s++;
    for (; *s && *s != ' '; s++) {
        if (*s == 'K') b->p.castle |= 1; else if (*s == 'Q') b->p.castle |= 2;
        else if (*s == 'k') b->p.castle |= 4; else if (*s == 'q') b->p.castle |= 8;
    }
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) b->p.occ[c] |= b->p.pc[c][pt];
    if (!b->p.pc[0][KING] || !b->p.pc[1][KING]) return -1;
    if (!(b->p.pc[0][KING] & (1ULL << 4)) || !(b->p.pc[0][ROOK] & (1ULL << 7))) b->p.castle &= ~1;
    if (!(b->p.pc[0][KING] & (1ULL << 4)) || !(b->p.pc[0][ROOK] & (1ULL << 0))) b->p.castle &= ~2;
    if (!(b->p.pc[1][KING] & (1ULL << 60)) || !(b->p.pc[1][ROOK] & (1ULL << 63))) b->p.castle &= ~4;
    if (!(b->p.pc[1][KING] & (1ULL << 60)) || !(b->p.pc[1][ROOK] & (1ULL << 56))) b->p.castle &= ~8;
    b->p.ep = -1;
    if (*s == ' ') s++;
    if (*s >= 'a' && *s <= 'h' && s[1] >= '1' && s[1] <= '8') {
        int eps = (s[1] - '1') * 8 + (s[0] - 'a');
        if (pawn_att[b->p.side ^ 1][eps] & b->p.pc[b->p.side][PAWN]) b->p.ep = (int8_t)eps;
        s += 2;
    } else if (*s == '-') s++;
    if (*s == ' ') s++;
    b->p.halfmove = (uint16_t)atoi(s);
    b->ply = 0;
    b->hist[0] = pos_hash(&b->p);
    refresh_legal(b);
    return 0;
}

void ce_get_fen(const Board *boards, int e, char *out) {
    const Board *b = &boards[e];
    char *o = out;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file, found = 0;
            for (int c = 0; c < 2 && !found; c++) {
                int pt = piece_at(&b->p, c, sq);
                if (pt >= 0) {
                    if (empty) { *o++ = (char)('0' + empty); empty = 0; }
                    *o++ = (char)(PIECE_CHARS[pt] + (c ? 32 : 0));
                    found = 1;
                }
            }
            if (!found) empty++;
        }
        if (empty) *o++ = (char)('0' + empty);
        if (rank) *o++ = '/';
    }
    *o++ = ' '; *o++ = b->p.side ? 'b' : 'w'; *o++ = ' ';
    if (!b->p.castle) *o++ = '-';
    if (b->p.castle & 1) *o++ = 'K';
    if (b->p.castle & 2) *o++ = 'Q';
    if (b->p.castle & 4) *o++ = 'k';
    if (b->p.castle & 8) *o++ = 'q';
    *o++ = ' ';
    if (b->p.ep < 0) *o++ = '-';
    else { *o++ = (char)('a' + (b->p.ep & 7)); *o++ = (char)('1' + (b->p.ep >> 3)); }
    sprintf(o, " %d %d", b->p.halfmove, (int)b->ply / 2 + 1);
}

/* legal moves of the current position, packed as from | to << 6 | promotion << 12 */
int ce_legal_moves(const Board *boards, int e, uint32_t *out) {
    const Board *b = &boards[e];
    for (uint32_t i = 0; i < b->nleg; i++) out[i] = b->leg[i] & 0x7FFF;
    return (int)b->nleg;
}

/* applies a move given as from | to << 6 | promotion << 12 (it must be legal); returns the game status */
int ce_apply_move(Board *boards, int e, uint32_t move, int max_plies) {
    Board *b = &boards[e];
    for (uint32_t i = 0; i < b->nleg; i++)
        if ((b->leg[i] & 0x7FFF) == move) {
            board_apply(b, b->leg[i]);
            refresh_legal(b);
            return game_status(b, max_plies);
        }
    return -1;
}

U64 ce_perft(const Board *boards, int e, int depth) { return perft(&boards[e].p, depth); }
int ce_in_check(const Board *boards, int e) { return in_check(&boards[e].p); }
int ce_status(Board *boards, int e, int max_plies) { return game_status(&boards[e], max_plies); }
