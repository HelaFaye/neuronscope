extends Node3D
## NeuronScope Godot client.
##
## Fetches the same api/meta, api/theme and api/trace the three.js client
## uses. All thresholding and flagging happened in Python; this only draws, so
## the frontends cannot drift on what counts as a flagged token or cell.
##
##   NS_API=http://127.0.0.1:7880 godot --path viz/godot
##   NS_FRAME=146 NS_PAUSED=1 ...    # open on one token, paused (for screenshots)
##   NS_API=https://studio:7870/viz/<id> NS_TOKEN=... godot --path viz/godot   # a reply checked in Studio
##
## MultiMesh keeps the whole field in one draw call, and glow runs at half
## resolution, because the GPU is usually also running the model.
##
## Layout: x = neuron column (ordered by classifier weight when the trace has a
## classifier, so the H-neurons form a band at the left), y = layer. Flagged
## H-neurons are drawn as rings in a second MultiMesh on top of the field.

const DEFAULT_API := "http://127.0.0.1:7880"
# Shared with the three.js client (viz/bloom.py); tests/test_bloom_demo.py keeps them equal.
const SIZE_IDLE := 1.1
const SIZE_ACTIVE := 2.4
const SIZE_FLAG := 7.0
const FPS := 8.0

var api: String
var token: String
var meta: Dictionary
var theme: Dictionary
var n_frames := 0
var n_cells := 0
var layer_of := PackedInt32Array()
var neuron_of := PackedInt32Array()
var intensity := PackedFloat32Array()
var state := PackedByteArray()
var prob := PackedFloat32Array()
var flagged_set := {}
var threshold := 0.5

var frame := 0
var playing := true
var accum := 0.0
var orbit := 0.0
var xs := 0.06               # x units per neuron column, fitted to the trace in _build
var center := Vector3.ZERO
var radius := 520.0

var flags: MultiMeshInstance3D
var info: RichTextLabel
var strip: RiskStrip
var context: RichTextLabel

@onready var cells: MultiMeshInstance3D = $Cells
@onready var hud: Label = $UI/HUD
@onready var cam: Camera3D = $Camera3D


## Risk over the whole reply: per-token line, threshold, flagged stretches and
## a playhead. Click to jump to a token.
class RiskStrip extends Control:
	var owner_node
	func _gui_input(e: InputEvent) -> void:
		if e is InputEventMouseButton and e.pressed and owner_node.n_frames > 1:
			owner_node.seek(int(round(e.position.x / size.x * (owner_node.n_frames - 1))))
	func _draw() -> void:
		var n: int = owner_node.n_frames
		var w := size.x
		var h := size.y
		draw_rect(Rect2(Vector2.ZERO, size), Color(0.086, 0.086, 0.118, 0.92))
		if n < 2:
			return
		var hal := Color(owner_node.theme.halluc)
		var x := func(i): return float(i) / (n - 1) * w
		var y := func(p): return h - 2.0 - (h - 4.0) * p
		if owner_node.prob.size() == n:
			var bw: float = max(2.0, w / n)
			for i in owner_node.flagged_set:
				draw_rect(Rect2(x.call(i) - bw / 2.0, 0, bw, h), Color(hal.r, hal.g, hal.b, 0.2))
			var ty: float = y.call(owner_node.threshold)
			var dx := 0.0
			while dx < w:
				draw_line(Vector2(dx, ty), Vector2(min(dx + 4.0, w), ty), Color(0.54, 0.54, 0.57), 1.0)
				dx += 8.0
			var pts := PackedVector2Array()
			for i in n:
				pts.append(Vector2(x.call(i), y.call(owner_node.prob[i])))
			draw_polyline(pts, hal, 1.5, true)
		else:
			draw_string(ThemeDB.fallback_font, Vector2(8, h / 2 + 4),
				"no classifier scores in this trace: nothing can be flagged", HORIZONTAL_ALIGNMENT_LEFT, -1, 12,
				Color(0.54, 0.54, 0.57))
		draw_rect(Rect2(x.call(owner_node.frame) - 1, 0, 2, h), Color.WHITE)


func _ready() -> void:
	api = OS.get_environment("NS_API")
	if api == "":
		api = DEFAULT_API
	api = api.trim_suffix("/")
	token = OS.get_environment("NS_TOKEN")
	hud.text = "connecting to %s…" % api
	_build_ui()
	_fetch("/api/meta", _on_meta)


func _build_ui() -> void:
	info = RichTextLabel.new()
	info.bbcode_enabled = true
	info.fit_content = true
	info.scroll_active = false
	info.position = Vector2(14, 10)
	info.size = Vector2(640, 120)
	info.add_theme_font_size_override("normal_font_size", 14)
	info.mouse_filter = Control.MOUSE_FILTER_IGNORE
	$UI.add_child(info)
	strip = RiskStrip.new()
	strip.owner_node = self
	strip.anchor_left = 0.0
	strip.anchor_right = 1.0
	strip.anchor_top = 1.0
	strip.anchor_bottom = 1.0
	strip.offset_left = 14
	strip.offset_right = -14
	strip.offset_top = -100
	strip.offset_bottom = -54
	$UI.add_child(strip)
	context = RichTextLabel.new()
	context.bbcode_enabled = true
	context.scroll_active = false
	context.anchor_left = 0.0
	context.anchor_right = 1.0
	context.anchor_top = 1.0
	context.anchor_bottom = 1.0
	context.offset_left = 14
	context.offset_right = -14
	context.offset_top = -48
	context.offset_bottom = -8
	context.add_theme_font_size_override("normal_font_size", 15)
	context.mouse_filter = Control.MOUSE_FILTER_IGNORE
	$UI.add_child(context)
	strip.visible = false
	context.visible = false


func _fetch(path: String, cb: Callable, binary := false) -> void:
	var r := HTTPRequest.new()
	add_child(r)
	r.request_completed.connect(
		func(_res, code, _h, body):
			r.queue_free()
			if code != 200:
				hud.text = "backend returned %d for %s\nis viz/bloom.py (or Studio) running, and NS_TOKEN set if needed?" % [code, path]
				return
			cb.call(body if binary else JSON.parse_string(body.get_string_from_utf8()))
	)
	var headers := PackedStringArray()
	if token != "":
		headers.append("Authorization: Bearer " + token)
	var err := r.request(api + path, headers)
	if err != OK:
		hud.text = "cannot reach %s" % api


func _on_meta(d) -> void:
	meta = d
	n_frames = int(meta.frames)
	threshold = float(meta.get("threshold", 0.5))
	# JSON numbers arrive as floats; key the set by int.
	for i in meta.get("flagged", []):
		flagged_set[int(i)] = true
	var p = meta.get("prob")
	if p != null:
		prob = PackedFloat32Array(p)
	_fetch("/api/theme", _on_theme)


func _on_theme(d) -> void:
	theme = d
	# The Environment owns the background now (BG_COLOR), so set that rather
	# than the clear colour, which BG_COLOR paints over.
	var env: Environment = $WorldEnvironment.environment
	if env:
		env.background_color = Color(theme.bg)
		env.fog_light_color = Color(theme.bg)
	_fetch("/api/trace", _on_trace, true)


func _on_trace(body: PackedByteArray) -> void:
	# Layout matches the docstring in viz/bloom.py exactly.
	var T := body.decode_s32(0)
	var N := body.decode_s32(4)
	var L := body.decode_s32(8)
	n_frames = T
	n_cells = N
	var o := 12
	layer_of = body.slice(o, o + N * 4).to_int32_array()
	o += N * 4
	neuron_of = body.slice(o, o + N * 4).to_int32_array()
	o += N * 4
	intensity = body.slice(o, o + T * N * 4).to_float32_array()
	o += T * N * 4
	state = body.slice(o, o + T * N)
	var start := OS.get_environment("NS_FRAME")
	if start.is_valid_int():
		frame = clampi(start.to_int(), 0, max(0, T - 1))
	playing = OS.get_environment("NS_PAUSED") != "1"
	hud.text = ""
	strip.visible = true
	context.visible = true
	_build(L)


func _ring_texture() -> ImageTexture:
	# A solid core inside a ring: flagged H-neurons read as a different kind of
	# mark, not just a brighter dot, whatever the colour vision.
	var n := 64
	var img := Image.create(n, n, false, Image.FORMAT_RGBA8)
	for yy in n:
		for xx in n:
			var r := Vector2(xx + 0.5 - n / 2.0, yy + 0.5 - n / 2.0).length() / (n / 2.0)
			var a := 0.0
			if r < 0.44:
				a = 1.0
			elif r < 0.72:
				a = 0.15
			elif r < 1.0:
				a = clampf((1.0 - r) / 0.12, 0.0, 1.0)
			img.set_pixel(xx, yy, Color(1, 1, 1, a))
	return ImageTexture.create_from_image(img)


func _multimesh(count: int) -> MultiMesh:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var quad := QuadMesh.new()
	quad.size = Vector2(1.6, 1.6)
	mm.mesh = quad
	mm.instance_count = count
	return mm


func _build(layers: int) -> void:
	# Fit the field to the view: neuron columns span 640 units whatever the
	# trace's width (512 bins or 14336 neurons); layers are 8 units apart. The
	# same numbers as the three.js client in viz/bloom.py.
	var width := 640.0
	xs = width / max(1, int(meta.get("neurons", 512)))
	var height := layers * 8.0
	center = Vector3(width / 2.0, height / 2.0 - 40.0, 0.0)
	radius = 0.65 * max(width, height)
	var mm := _multimesh(n_cells)
	for i in n_cells:
		mm.set_instance_transform(i, Transform3D(Basis.IDENTITY,
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))
	cells.multimesh = mm
	# Flagged H-neurons: their own MultiMesh, unshaded, drawn over the field.
	flags = MultiMeshInstance3D.new()
	var fm := StandardMaterial3D.new()
	fm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	fm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	fm.billboard_mode = BaseMaterial3D.BILLBOARD_ENABLED
	fm.billboard_keep_scale = true    # without it billboards ignore instance scale
	fm.vertex_color_use_as_albedo = true
	fm.no_depth_test = true
	fm.render_priority = 10
	fm.albedo_texture = _ring_texture()
	flags.material_override = fm
	flags.multimesh = _multimesh(n_cells)
	add_child(flags)
	_paint()


func seek(i: int) -> void:
	frame = clampi(i, 0, n_frames - 1)
	_paint()


func _paint() -> void:
	if cells.multimesh == null:
		return
	var mm := cells.multimesh
	var fmm := flags.multimesh
	var base := frame * n_cells
	var c_idle := Color(0.16, 0.16, 0.22)
	var c_act := Color(theme.active)
	var c_hal := Color(theme.halluc)
	for i in n_cells:
		var v: float = intensity[base + i]
		var st: int = state[base + i]
		var col: Color = c_idle
		var gain := 0.35
		if st >= 1:
			col = c_act
			gain = 0.6 + 0.8 * v
		mm.set_instance_color(i, Color(col.r * gain, col.g * gain, col.b * gain, 1.0 if st > 0 else 0.35))
		var s: float = (SIZE_ACTIVE if st >= 1 else SIZE_IDLE) * (0.6 + 0.9 * v)
		mm.set_instance_transform(i, Transform3D(Basis.IDENTITY.scaled(Vector3(s, s, s)),
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))
		# Full colour, never pushed past 1, so the hue survives the glow instead of washing to white.
		var fs: float = SIZE_FLAG * (0.8 + 0.5 * v) if st == 2 else 0.0
		fmm.set_instance_color(i, c_hal)
		fmm.set_instance_transform(i, Transform3D(Basis.IDENTITY.scaled(Vector3(fs, fs, fs)),
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.5)))
	_update_text()
	strip.queue_redraw()


func _hex(c: Color) -> String:
	return "#" + c.to_html(false)


func _update_text() -> void:
	var flagged: bool = flagged_set.has(frame)
	var hal := _hex(Color(theme.halluc))
	var act := _hex(Color(theme.active))
	var mode := str(meta.get("mode", "unscored"))
	var mode_text := "tokens flagged at risk ≥ %d%% (the classifier's own threshold)" % int(round(threshold * 100))
	if mode == "relative":
		mode_text = "[color=#ff9b8a]relative mode: some tokens are always flagged[/color]"
	elif mode == "unscored":
		mode_text = "[color=#ff9b8a]no classifier scores: nothing can be flagged[/color]"
	var cells_note := str(meta.get("cells_note", ""))
	var order := "x: neurons by classifier weight, H-neurons at left · y: layer" if meta.get("order") == "weight" else "x: neuron index · y: layer"
	var risk := ""
	if prob.size() == n_frames:
		risk = "   risk [b]%d%%[/b]" % int(round(prob[frame] * 100))
	info.text = ("[b]%s[/b]   [color=#8a8a92]%d/%d[/color]%s%s\n" % [str(meta.get("model", "trace")), frame + 1,
		n_frames, risk, ("   [color=%s][b]FLAGGED[/b][/color]" % hal) if flagged else ""]) + \
		("[color=%s]●[/color] active   [color=%s]◉[/color] H-neuron on a flagged token\n" % [act, hal]) + \
		("[color=#8a8a92]%s%s\n%s[/color]" % [mode_text, ("\n" + cells_note) if cells_note.begins_with("none") else "", order])
	# The reply around the current token, shaded by risk; the current token boxed.
	var labels: Array = meta.get("labels", [])
	var lo := maxi(0, frame - 14)
	var hi := mini(labels.size(), frame + 15)
	var out := ""
	for i in range(lo, hi):
		var t := str(labels[i]).replace("[", "(").replace("]", ")").replace("\n", " ")
		if prob.size() == n_frames and prob[i] > 0.15:
			var a := clampf((prob[i] - 0.15) / 0.85, 0.0, 1.0) * 0.6
			var bg := Color(theme.halluc)
			bg.a = a
			t = "[bgcolor=%s]%s[/bgcolor]" % ["#" + bg.to_html(true), t]
		if flagged_set.has(i):
			t = "[u]%s[/u]" % t
		if i == frame:
			t = "[b][color=#ffffff]⟦%s⟧[/color][/b]" % t
		out += t
	context.text = ("…" if lo > 0 else "") + out + ("…" if hi < labels.size() else "")


func _process(delta: float) -> void:
	if n_cells == 0:
		return
	if playing:
		accum += delta
		if accum >= 1.0 / FPS:
			accum = 0.0
			frame = (frame + 1) % n_frames
			_paint()
	# A slow sway in front of the field rather than a full orbit: from behind, the
	# billboards are the same picture mirrored, and edge-on the field vanishes.
	orbit += delta * 0.15
	var a := PI / 2.0 + 0.35 * sin(orbit)
	cam.position = center + Vector3(cos(a) * radius, 0.3 * radius, sin(a) * radius)
	cam.look_at(center, Vector3.UP)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_SPACE:
				playing = not playing
			KEY_LEFT:
				seek((frame - 1 + n_frames) % n_frames)
			KEY_RIGHT:
				seek((frame + 1) % n_frames)
			KEY_N:
				# next flagged token
				for k in range(1, n_frames + 1):
					var j := (frame + k) % n_frames
					if flagged_set.has(j):
						seek(j)
						break
			KEY_ESCAPE:
				get_tree().quit()
