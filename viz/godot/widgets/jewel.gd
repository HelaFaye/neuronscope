extends "res://widgets/widget.gd"
## The emotion jewel: one glance at the model's emotion vectors on the current token.
##
## Reads set_data({"names": [...], "z": [[z per emotion] per token], "colors": [...], "frame": i}),
## the same track the emotion graph draws, so the jewel's colours are the lanes' colours.
##
##   colour     the lanes' colours mixed by |z| squared (negative z mixed in darker)
##   size, glow how far the strongest emotion is from its story mean (|z| of 3 = full)
##   motion     each emotion's ribbon, face bulge or 4D turn follows its own z
##
## Styles (settings "style"): core (default; a reactor core with an orbiting
## ribbon per emotion), icosphere (a sphere pushed out where each emotion is
## strong), tesseract (a 4D cube turning faster as emotions rise), off.
## Values glide toward each token's numbers rather than jump, so stepping
## through a reply reads as one continuous motion.
## Click: hide or show the emotion graph (signal toggled). Right-click: next style.

signal toggled

const STYLES := ["core", "icosphere", "tesseract", "off"]

var names: Array = []
var series: Array = []
var colors: Array = []
var frame := 0
var t := 0.0
var cur: PackedFloat32Array = PackedFloat32Array()     # smoothed z per emotion
var mag := 0.0                                          # smoothed max |z| / 3
var mix := Color(0.4, 0.55, 1.0)

var ico_v: Array = []          # unit vectors
var ico_f: Array = []          # triangles (index triples)
var dirs: Array = []           # one direction per emotion slot, spread over the sphere


func _init() -> void:
	widget_id = "jewel"
	custom_minimum_size = Vector2(96, 96)
	mouse_filter = Control.MOUSE_FILTER_STOP
	tooltip_text = "Emotion jewel: click to hide or show the emotion graph, right-click for the next style"
	_icosphere()


func set_data(d: Dictionary) -> void:
	names = d.get("names", [])
	series = d.get("z", [])
	colors = d.get("colors", [])
	frame = int(d.get("frame", 0))
	if cur.size() != names.size():
		cur = PackedFloat32Array()
		cur.resize(names.size())
	if dirs.size() != names.size():
		dirs = _spread(names.size())


func target() -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(names.size())
	if frame < series.size():
		var row: Array = series[frame]
		for k in mini(row.size(), out.size()):
			out[k] = float(row[k])
	return out


func color_of(k: int) -> Color:
	if k < colors.size():
		return Color(str(colors[k]))
	return Color.from_hsv(fmod(0.13 * k, 1.0), 0.6, 1.0)


func _process(delta: float) -> void:
	if not is_visible_in_tree():
		return
	var sp: float = float(setting("speed", 1.0))
	t += delta * sp
	var goal := target()
	var a := 1.0 - exp(-delta * 5.0)
	var mx := 0.0
	var acc := Vector3.ZERO
	var wsum := 0.0
	for k in cur.size():
		cur[k] = lerpf(cur[k], goal[k], a)
		mx = maxf(mx, absf(cur[k]))
		var w := cur[k] * cur[k]
		var c := color_of(k)
		if cur[k] < 0:
			c = c.darkened(0.5)
		acc += Vector3(c.r, c.g, c.b) * w
		wsum += w
	mag = lerpf(mag, clampf(mx / 3.0, 0.0, 1.0), a)
	if wsum > 1e-4:
		var m := acc / wsum
		mix = mix.lerp(Color(m.x, m.y, m.z), a)
	queue_redraw()


func _gui_input(e: InputEvent) -> void:
	if e is InputEventMouseButton and e.pressed:
		if e.button_index == MOUSE_BUTTON_LEFT:
			toggled.emit()
		elif e.button_index == MOUSE_BUTTON_RIGHT:
			var i := STYLES.find(str(setting("style", "core")))
			save_setting("style", STYLES[(i + 1) % STYLES.size()])
		accept_event()


func _draw() -> void:
	var c := size / 2.0
	var r := minf(size.x, size.y) * 0.5 * float(setting("scale", 0.9))
	var glow: float = float(setting("glow", 1.0))
	var style := str(setting("style", "core"))
	if style == "off" or names.is_empty():
		draw_arc(c, r * 0.35, 0, TAU, 32, Color(0.4, 0.4, 0.45, 0.5), 1.5, true)
		return
	# a soft halo, wider and brighter as emotions rise
	for i in 6:
		var f := 1.0 - i / 6.0
		var hc := mix
		hc.a = 0.05 * glow * (0.4 + mag) * f
		draw_circle(c, r * (0.45 + 0.55 * (1.0 - f)) * (0.7 + 0.3 * mag), hc)
	match style:
		"icosphere":
			_draw_ico(c, r)
		"tesseract":
			_draw_tesseract(c, r)
		_:
			_draw_core(c, r, glow)


# ------------------------------------------------------------- core

## A reactor core: a bright centre and, per emotion, a ribbon on its own tilted
## orbit. A ribbon's length and brightness follow its emotion's |z|; positive z
## turns one way, negative the other; the core pulses with the strongest.
func _draw_core(c: Vector2, r: float, glow: float) -> void:
	var n := cur.size()
	var back := []
	var front := []
	for k in n:
		var z := cur[k]
		var s := clampf(absf(z) / 3.0, 0.0, 1.0)
		var tilt := 0.35 + 1.1 * float(k) / maxf(1.0, n)
		var spin := float(k) * TAU / maxf(1.0, n)
		var head := t * (0.8 + 1.6 * s) * (1.0 if z >= 0 else -1.0) + spin * 1.7
		var length := 0.6 + 2.4 * s
		var steps := 24
		var col := color_of(k)
		var rr := r * (0.68 + 0.12 * k / maxf(1.0, n))
		for j in steps:
			var u0 := head - length * float(j) / steps
			var u1 := head - length * float(j + 1) / steps
			var p0 := _orbit(c, rr, tilt, spin, u0)
			var p1 := _orbit(c, rr, tilt, spin, u1)
			var fade := (1.0 - float(j) / steps) * (0.5 + 0.5 * s)
			var seg := [p0[0], p1[0], col, fade, (1.6 + 3.4 * s) * (1.0 - 0.6 * float(j) / steps)]
			if p0[1] < 0:
				back.append(seg)
			else:
				front.append(seg)
	for sgm in back:
		var cc: Color = sgm[2]
		cc.a = sgm[3] * 0.35
		draw_line(sgm[0], sgm[1], cc, sgm[4], true)
	# the core: layered discs, the mix colour at the rim, nearly white at the heart
	var pulse := 1.0 + 0.08 * sin(t * (2.0 + 5.0 * mag)) * (0.3 + mag)
	var cr := r * (0.26 + 0.18 * mag) * pulse
	for i in 8:
		var f := float(i) / 8.0
		var col := mix.lerp(Color(1, 1, 1), f * f * 0.85)
		col.a = (0.35 + 0.65 * f) * minf(1.0, 0.6 + 0.4 * glow)
		draw_circle(c, cr * (1.0 - 0.11 * i), col)
	for sgm in front:
		var cc: Color = sgm[2]
		cc.a = sgm[3]
		draw_line(sgm[0], sgm[1], cc, sgm[4], true)


## A point on a tilted, turned circle -> [screen point, depth].
func _orbit(c: Vector2, rr: float, tilt: float, spin: float, u: float) -> Array:
	var p := Vector3(cos(u) * rr, sin(u) * rr * cos(tilt), sin(u) * rr * sin(tilt))
	var q := Vector3(p.x * cos(spin) - p.y * sin(spin), p.x * sin(spin) + p.y * cos(spin), p.z)
	return [c + Vector2(q.x, q.y), q.z]


# ------------------------------------------------------------- icosphere

func _icosphere() -> void:
	var g := (1.0 + sqrt(5.0)) / 2.0
	var v := [Vector3(-1, g, 0), Vector3(1, g, 0), Vector3(-1, -g, 0), Vector3(1, -g, 0),
		Vector3(0, -1, g), Vector3(0, 1, g), Vector3(0, -1, -g), Vector3(0, 1, -g),
		Vector3(g, 0, -1), Vector3(g, 0, 1), Vector3(-g, 0, -1), Vector3(-g, 0, 1)]
	var f := [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
		[10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
		[6, 2, 10], [8, 6, 7], [9, 8, 1]]
	for i in v.size():
		v[i] = v[i].normalized()
	# one subdivision: 80 faces, enough to show where it bulges
	var mid := {}
	var nf := []
	for tri in f:
		var m := []
		for e in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]:
			var key := "%d_%d" % [mini(e[0], e[1]), maxi(e[0], e[1])]
			if not mid.has(key):
				v.append(((v[e[0]] + v[e[1]]) / 2.0).normalized())
				mid[key] = v.size() - 1
			m.append(mid[key])
		nf += [[tri[0], m[0], m[2]], [tri[1], m[1], m[0]], [tri[2], m[2], m[1]], [m[0], m[1], m[2]]]
	ico_v = v
	ico_f = nf


## n directions spread evenly over the sphere (a Fibonacci lattice).
func _spread(n: int) -> Array:
	var out := []
	for i in n:
		var y := 1.0 - 2.0 * (i + 0.5) / maxf(1.0, n)
		var rad := sqrt(maxf(0.0, 1.0 - y * y))
		var th := PI * (3.0 - sqrt(5.0)) * i
		out.append(Vector3(cos(th) * rad, y, sin(th) * rad))
	return out


## The sphere is pushed out (positive z) or in (negative) around each emotion's
## direction, and each face takes the colours of the emotions bulging it.
func _draw_ico(c: Vector2, r: float) -> void:
	var rot := Basis(Vector3.UP, t * 0.5) * Basis(Vector3.RIGHT, 0.4 + 0.2 * sin(t * 0.3))
	var pts := []
	var wts := []
	for v in ico_v:
		var d := 0.0
		var w := []
		for k in cur.size():
			var f := pow(maxf(0.0, v.dot(dirs[k])), 3.0)
			d += cur[k] / 3.0 * f
			w.append(f * absf(cur[k]))
		var p: Vector3 = rot * (v * (0.55 + 0.35 * clampf(d, -0.6, 1.2))) * r
		pts.append(p)
		wts.append(w)
	var light := Vector3(-0.4, -0.6, 0.7).normalized()
	var faces := []
	for tri in ico_f:
		var a: Vector3 = pts[tri[0]]
		var b: Vector3 = pts[tri[1]]
		var e: Vector3 = pts[tri[2]]
		var nrm := (b - a).cross(e - a).normalized()
		if nrm.z < 0:
			continue
		faces.append([(a.z + b.z + e.z) / 3.0, tri, nrm])
	faces.sort_custom(func(x, y): return x[0] < y[0])
	for fc in faces:
		var tri: Array = fc[1]
		var col := Color(0, 0, 0)
		var ws := 0.0
		for i in tri:
			for k in cur.size():
				var w: float = wts[i][k]
				var ck := color_of(k)
				if cur[k] < 0:
					ck = ck.darkened(0.5)
				col += ck * w
				ws += w
		col = mix.darkened(0.35) if ws < 1e-4 else col / ws
		var shade := 0.35 + 0.65 * maxf(0.0, (fc[2] as Vector3).dot(light))
		col = Color(col.r * shade, col.g * shade, col.b * shade, 0.95)
		var poly := PackedVector2Array([c + Vector2(pts[tri[0]].x, pts[tri[0]].y),
			c + Vector2(pts[tri[1]].x, pts[tri[1]].y), c + Vector2(pts[tri[2]].x, pts[tri[2]].y)])
		draw_colored_polygon(poly, col)
		draw_polyline(poly + PackedVector2Array([poly[0]]), Color(col.lightened(0.3), 0.35), 1.0, true)


# ------------------------------------------------------------- tesseract

## A 4D cube. It turns in the XW plane with the positive emotions and in YW
## with the negative ones, faster the stronger they are; the inner cube takes
## the mix colour and each outer edge one emotion's colour.
func _draw_tesseract(c: Vector2, r: float) -> void:
	var pos := 0.0
	var neg := 0.0
	for z in cur:
		pos += maxf(0.0, z)
		neg += maxf(0.0, -z)
	var axw := t * (0.3 + 0.5 * clampf(pos / 3.0, 0.0, 2.0))
	var ayw := t * (0.2 + 0.5 * clampf(neg / 3.0, 0.0, 2.0))
	var axy := t * 0.25
	var proj := []
	for i in 16:
		var p := [1.0 if i & 1 else -1.0, 1.0 if i & 2 else -1.0, 1.0 if i & 4 else -1.0, 1.0 if i & 8 else -1.0]
		var x: float = p[0] * cos(axw) - p[3] * sin(axw)
		var w: float = p[0] * sin(axw) + p[3] * cos(axw)
		var y: float = p[1] * cos(ayw) - w * sin(ayw)
		w = p[1] * sin(ayw) + w * cos(ayw)
		var x2: float = x * cos(axy) - p[2] * sin(axy)
		var z2: float = x * sin(axy) + p[2] * cos(axy)
		var k4 := 1.0 / (2.6 - w)
		var v3 := Vector3(x2, y, z2) * k4
		var k3 := 1.0 / (3.0 - v3.z)
		proj.append([c + Vector2(v3.x, v3.y) * k3 * r * 2.4, w])
	var n := cur.size()
	var e := 0
	for i in 16:
		for b in 4:
			var j := i ^ (1 << b)
			if j < i:
				continue
			var inner: bool = proj[i][1] < 0 and proj[j][1] < 0
			var col := mix if inner or n == 0 else color_of(e % n)
			var s := mag if inner or n == 0 else clampf(absf(cur[e % n]) / 3.0, 0.0, 1.0)
			col.a = 0.35 + 0.65 * s
			draw_line(proj[i][0], proj[j][0], col, 1.0 + 2.5 * s, true)
			e += 1
	for p in proj:
		draw_circle(p[0], 1.5 + 2.0 * mag, mix.lightened(0.4))
