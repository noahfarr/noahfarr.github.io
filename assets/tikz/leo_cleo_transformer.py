import sane_tikz.core as stz
import sane_tikz.formatting as fmt

width = 2.5
line_width = 3.0 * fmt.standard_line_width
rectangle_roundness = 0.1
connector_roundness = 0.2
to_add_norm_spacing = 0.15
per_line_height = 0.40
arrow_length = 0.6
residual_spacing = 2.0
trident_alpha = 0.75
trident_spacing = 0.4
spacing_between_towers = 3.0
leo_gap = 4.5
bbox_spacing = 0.45
bbox_roundness = 0.3

name2color = {
    "emb_color": (252, 224, 225),
    "multi_head_attention_color": (252, 226, 187),
    "add_norm_color": (242, 243, 193),
    "ff_color": (194, 232, 247),
    "sigmoid_color": (203, 231, 207),
    "linear_color": (220, 223, 240),
    "enc_color": (183, 226, 212),
    "gray_bbox_color": (243, 243, 244),
    "blocktext": (34, 30, 24),
}

s_arr = fmt.combine_tikz_strs([fmt.line_width(line_width), fmt.arrow_heads("end")])
s_lw = fmt.combine_tikz_strs([fmt.line_width(line_width)])
s_con = fmt.combine_tikz_strs([
    fmt.arrow_heads("end"),
    fmt.line_width(line_width),
    fmt.rounded_corners(connector_roundness),
])
s_bbox = fmt.combine_tikz_strs([
    fmt.fill_color("gray_bbox_color"),
    fmt.line_width(line_width),
    fmt.rounded_corners(bbox_roundness),
])
s_div = fmt.combine_tikz_strs([fmt.line_width(line_width), "draw=black!18"])


def trident_coords(e):
    top_left_cs, bottom_right_cs = stz.bbox(e)
    center_cs = stz.bottom_center_coords(top_left_cs, bottom_right_cs)
    left_cs = stz.translate_coords_horizontally(center_cs, -trident_alpha * width / 2.0)
    right_cs = stz.translate_coords_horizontally(center_cs, trident_alpha * width / 2.0)
    cs_lst = [left_cs, center_cs, right_cs]
    translated_cs_lst = [
        stz.translate_coords_vertically(cs, -trident_spacing) for cs in cs_lst
    ]
    return cs_lst, translated_cs_lst


def lst2str(s_lst):
    return " \\vspace{-0.05cm} \\linebreak ".join(s_lst)


def rectangle(height, color_name):
    s_fmt = fmt.combine_tikz_strs([
        fmt.line_width(line_width),
        fmt.fill_color(color_name),
        fmt.rounded_corners(rectangle_roundness),
    ])
    return stz.rectangle_from_width_and_height([0, 0], height, width, s_fmt)


def rectangle_with_text(color_name, s_lst):
    height = 0.1 + per_line_height * len(s_lst)
    s_width = fmt.combine_tikz_strs(
        [fmt.text_width(width), "align=center", "text=blocktext"])
    r = rectangle(height, color_name)
    cs = stz.center_coords(r)
    s = " \\vspace{-0.05cm} \\linebreak ".join(s_lst)
    t = stz.latex(cs, s, s_width)
    return [r, t]


def rectangle_with_add_norm(color_name, s_lst):
    r1 = rectangle_with_text("add_norm_color", ["Add \\& Norm"])
    r2 = rectangle_with_text(color_name, s_lst)
    top_left_cs, bottom_right_cs = stz.bbox(r2)
    cs = stz.translate_coords_vertically(
        stz.top_center_coords(top_left_cs, bottom_right_cs), to_add_norm_spacing)
    stz.translate_bbox_bottom_center_to_coords(r1, cs)
    c = connect_straight_vertical(r2, r1)
    return [r1, r2, c]


def connect_straight_vertical(e_from, e_to):
    from_cs = stz.coords_from_bbox_with_fn(e_from, stz.top_center_coords)
    to_cs = stz.coords_from_bbox_with_fn(e_to, stz.bottom_center_coords)
    return stz.line_segment(from_cs, to_cs, s_lw)


def connect_straight_with_arrow(e_from, e_to):
    from_cs = stz.coords_from_bbox_with_fn(e_from, stz.top_center_coords)
    to_cs = stz.coords_from_bbox_with_fn(e_to, stz.bottom_center_coords)
    return stz.line_segment(from_cs, to_cs, s_arr)


def connect_residual(e_with_addnorm, bottom_spacing, side_spacing):
    from_cs = stz.coords_from_bbox_with_fn(e_with_addnorm, stz.bottom_center_coords)
    from_cs = stz.translate_coords_vertically(from_cs, -bottom_spacing)
    if side_spacing < 0:
        to_cs = stz.coords_from_bbox_with_fn(e_with_addnorm[0], stz.left_center_coords)
    else:
        to_cs = stz.coords_from_bbox_with_fn(e_with_addnorm[0], stz.right_center_coords)
    cs = stz.translate_coords_horizontally(from_cs, side_spacing)
    return stz.open_path([from_cs, cs, [cs[0], to_cs[1]], to_cs], s_con)


def arrow_in(e):
    to_cs = stz.coords_from_bbox_with_fn(e, stz.bottom_center_coords)
    from_cs = stz.translate_coords_vertically(to_cs, -arrow_length)
    return stz.line_segment(from_cs, to_cs, s_arr)


def arrow_out(e):
    from_cs = stz.coords_from_bbox_with_fn(e, stz.top_center_coords)
    to_cs = stz.translate_coords_vertically(from_cs, arrow_length)
    return stz.line_segment(from_cs, to_cs, s_arr)


def txt(cs, s, opts):
    return stz.latex(cs, s, fmt.combine_tikz_strs(opts))


# ==================== LEO : single tower ====================
Lobs = rectangle_with_text("emb_color", ["Observation"])
Ltrunk = rectangle_with_text("ff_color", ["MLP trunk"])
Lvec = rectangle_with_text("enc_color", ["Shared vector"])
Llin = rectangle_with_text("linear_color", ["Linear"])
Lsig = rectangle_with_text("sigmoid_color", ["Sigmoid"])
stz.place_above_and_align_to_the_center(Ltrunk, Lobs, 0.7)
stz.place_above_and_align_to_the_center(Lvec, Ltrunk, 0.7)
stz.place_above_and_align_to_the_center(Llin, Lvec, 0.7)
stz.place_above_and_align_to_the_center(Lsig, Llin, 0.7)

# ==================== CLEO : state encoder ====================
obs = rectangle_with_text("emb_color", ["Observation"])
trunk = rectangle_with_text("ff_color", ["MLP trunk"])
tokens = rectangle_with_text("enc_color", ["State tokens"])
stz.place_to_the_right_and_align_to_the_bottom(obs, Lobs, leo_gap)
stz.place_above_and_align_to_the_center(trunk, obs, 0.7)
stz.place_above_and_align_to_the_center(tokens, trunk, 0.7)

# ==================== CLEO : goal decoder ====================
gemb = rectangle_with_text("emb_color", ["Goal", "Embedding"])
xattn_an = rectangle_with_add_norm("multi_head_attention_color", ["Cross-Attention"])
ff_an = rectangle_with_add_norm("ff_color", ["Feed", "Forward"])
linear = rectangle_with_text("linear_color", ["Linear"])
sigmoid = rectangle_with_text("sigmoid_color", ["Sigmoid"])
stz.place_to_the_right_and_align_to_the_bottom(gemb, obs, spacing_between_towers)
stz.place_above_and_align_to_the_center(xattn_an, gemb, 1.33)
stz.place_above_and_align_to_the_center(ff_an, xattn_an, 1.0)
stz.place_above_and_align_to_the_center(linear, ff_an, 0.6)
stz.place_above_and_align_to_the_center(sigmoid, linear, 0.6)

# ==================== connections ====================
connections = [
    connect_straight_with_arrow(Lobs, Ltrunk),
    connect_straight_with_arrow(Ltrunk, Lvec),
    connect_straight_with_arrow(Lvec, Llin),
    connect_straight_with_arrow(Llin, Lsig),
    connect_straight_with_arrow(obs, trunk),
    connect_straight_with_arrow(trunk, tokens),
    connect_straight_with_arrow(xattn_an, ff_an),
    connect_straight_with_arrow(ff_an, linear),
    connect_straight_with_arrow(linear, sigmoid),
]
connections.extend([
    connect_residual(xattn_an, 0.6, residual_spacing),
    connect_residual(ff_an, 0.6, residual_spacing),
])

# decoder bounding box (N x)
b = stz.bbox([ff_an, xattn_an, connections[-1], connections[-2]])
bb = stz.rectangle_from_additive_resizing(b[0], b[1], bbox_spacing, bbox_spacing, s_bbox)

# ==================== trident : Q from goals, K,V from tokens ====================
tri_cs, translated_tri_cs = trident_coords(xattn_an)
from_cs = stz.coords_from_bbox_with_fn(tokens, stz.top_center_coords)
mid_cs = stz.midway_coords(stz.bbox(tokens)[1], stz.bbox(bb)[0])
cs = stz.translate_coords_vertically(from_cs, 1.0)
e1 = stz.open_path([
    from_cs, cs, [mid_cs[0], cs[1]], [mid_cs[0], translated_tri_cs[0][1]],
    translated_tri_cs[0], tri_cs[0]
], s_con)
e2 = stz.open_path([
    from_cs, cs, [mid_cs[0], cs[1]], [mid_cs[0], translated_tri_cs[1][1]],
    translated_tri_cs[1], tri_cs[1]
], s_con)
gq_cs = stz.coords_from_bbox_with_fn(gemb, stz.top_center_coords)
e3 = stz.open_path(
    [gq_cs, translated_tri_cs[1], translated_tri_cs[2], tri_cs[2]], s_con)
connections.extend([e1, e2, e3])

arrows = [arrow_in(Lobs), arrow_out(Lsig), arrow_in(obs), arrow_in(gemb),
          arrow_out(sigmoid)]

# ==================== annotations ====================
s_in = ["text width=%f cm" % width, "anchor=north", "align=center"]
s_out = ["text width=%f cm" % width, "anchor=south", "align=center"]

annotations = [
    txt(stz.coords_from_bbox_with_fn(arrows[0], stz.bottom_center_coords), "state", s_in),
    txt(stz.coords_from_bbox_with_fn(arrows[1], stz.top_center_coords),
        lst2str(["per-goal", "Q-values"]), s_out),
    txt(stz.coords_from_bbox_with_fn(arrows[2], stz.bottom_center_coords), "state", s_in),
    txt(stz.coords_from_bbox_with_fn(arrows[3], stz.bottom_center_coords), "goals", s_in),
    txt(stz.coords_from_bbox_with_fn(arrows[4], stz.top_center_coords),
        lst2str(["per-goal", "Q-values"]), s_out),
]

# N x
cs = stz.coords_from_bbox_with_fn(bb, stz.right_center_coords)
annotations.append(txt([cs[0] + 0.2, cs[1]], "$N\\times$", ["anchor=west"]))

# K,V label centred on the horizontal run of the encoder feed ; Q on the goal feed
# (the run spans from the tokens' centre to mid_cs, at height from_cs[1] + 1.0)
kv_cs = [(from_cs[0] + mid_cs[0]) / 2.0, from_cs[1] + 1.12]
annotations.append(txt(kv_cs, "K, V", ["anchor=south", "font=\\footnotesize"]))
annotations.append(txt([gq_cs[0] + 0.15, gq_cs[1] + 0.2], "Q",
                       ["anchor=west", "font=\\footnotesize"]))

# ==================== titles + divider ====================
leo_bb = stz.bbox([Lobs, Ltrunk, Lvec, Llin, Lsig])
cleo_bb = stz.bbox([obs, trunk, tokens, gemb, xattn_an, ff_an, linear, sigmoid, bb])
all_bb = stz.bbox([Lobs, Lsig, obs, sigmoid, bb, annotations, arrows])
top_y = all_bb[0][1]
bot_y = all_bb[1][1]

s_title = ["anchor=south", "align=center", "font=\\large\\bfseries"]
leo_cx = (leo_bb[0][0] + leo_bb[1][0]) / 2.0
cleo_cx = (cleo_bb[0][0] + cleo_bb[1][0]) / 2.0
annotations.append(txt([leo_cx, top_y + 0.4], "LEO", s_title))
annotations.append(txt([cleo_cx, top_y + 0.4], "CLEO", s_title))

x_div = (leo_bb[1][0] + stz.bbox(obs)[0][0]) / 2.0
divider = stz.line_segment([x_div, top_y + 0.2], [x_div, bot_y - 0.2], s_div)

# ==================== draw ====================
e = [
    divider, bb, Lobs, Ltrunk, Lvec, Llin, Lsig, obs, trunk, tokens,
    gemb, xattn_an, ff_an, linear, sigmoid, connections, arrows, annotations
]
stz.draw_to_tikz_standalone(e, "both_stz.tex", name2color)
