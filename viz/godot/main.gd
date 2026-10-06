extends Node3D
## NeuronScope Godot client.
##
## Fetches the same /api/meta, /api/theme and /api/trace the three.js client
## uses. All thresholding and flagging happened in Python; this only draws, so
## the three frontends cannot drift on what counts as "hallucinating".
##
##   NS_API=http://127.0.0.1:7880 godot --path viz/godot
##   NS_FRAME=146 NS_PAUSED=1 ...    # open on one token, paused (for screenshots)
##
## MultiMesh keeps the whole field in one draw call, and glow runs at half
## resolution, because the GPU is usually also running the model.

const DEFAULT_API := "http://127.0.0.1:7880"

var api: String
var meta: Dictionary
var theme: Dictionary
var n_frames := 0
var n_cells := 0
var layer_of := PackedInt32Array()
var neuron_of := PackedInt32Array()
var intensity := PackedFloat32Array()
var state := PackedByteArray()

var frame := 0
var playing := true
var accum := 0.0
var orbit := PI / 2.0        # start in front of the field (+z)
var xs := 0.06               # x units per neuron column, fitted to the trace in _build
var center := Vector3.ZERO
var radius := 520.0

@onready var cells: MultiMeshInstance3D = $Cells
@onready var hud: Label = $UI/HUD
@onready var cam: Camera3D = $Camera3D


func _ready() -> void:
	api = OS.get_environment("NS_API")
	if api == "":
		api = DEFAULT_API
	hud.text = "connecting to %s…" % api
	_fetch("/api/meta", _on_meta)


func _fetch(path: String, cb: Callable, binary := false) -> void:
	var r := HTTPRequest.new()
	add_child(r)
	r.request_completed.connect(
		func(_res, code, _h, body):
			r.queue_free()
			if code != 200:
				hud.text = "backend returned %d for %s\nis viz/bloom.py running?" % [code, path]
				return
			cb.call(body if binary else JSON.parse_string(body.get_string_from_utf8()))
	)
	var err := r.request(api + path)
	if err != OK:
		hud.text = "cannot reach %s" % api


func _on_meta(d) -> void:
	meta = d
	n_frames = int(meta.frames)
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
	_build(L)


func _build(layers: int) -> void:
	# Fit the field to the view: neuron columns span 640 units whatever the
	# trace's width (512 bins or 14336 neurons); layers are 8 units apart. The
	# same numbers as the three.js client in viz/bloom.py.
	var width := 640.0
	xs = width / max(1, int(meta.get("neurons", 512)))
	var height := layers * 8.0
	center = Vector3(width / 2.0, height / 2.0, 0.0)
	radius = 0.65 * max(width, height)
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var quad := QuadMesh.new()
	quad.size = Vector2(1.6, 1.6)
	mm.mesh = quad
	mm.instance_count = n_cells
	for i in n_cells:
		# Positions are static; only colour and scale change per frame, so the
		# transform buffer is written once.
		mm.set_instance_transform(i, Transform3D(
			Basis.IDENTITY,
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))
	cells.multimesh = mm
	_paint()


func _paint() -> void:
	if cells.multimesh == null:
		return
	var mm := cells.multimesh
	var base := frame * n_cells
	var c_idle := Color(0.16, 0.16, 0.22)
	var c_act := Color(theme.active)
	var c_hal := Color(theme.halluc)
	for i in n_cells:
		var v: float = intensity[base + i]
		var st: int = state[base + i]
		var col: Color = c_idle
		var gain := 0.35
		if st == 2:
			col = c_hal
			gain = 1.4 + 1.2 * v      # above the HDR threshold, so it blooms
		elif st == 1:
			col = c_act
			gain = 0.6 + 0.8 * v
		mm.set_instance_color(i, Color(col.r * gain, col.g * gain, col.b * gain,
			1.0 if st > 0 else 0.35))
		var s: float = (4.2 if st == 2 else 2.4 if st == 1 else 1.1) * (0.6 + 0.9 * v)
		mm.set_instance_transform(i, Transform3D(
			Basis.IDENTITY.scaled(Vector3(s, s, s)),
			Vector3(neuron_of[i] * xs, layer_of[i] * 8.0, 0.0)))

	# JSON numbers arrive as floats, and Array.has() does not equate 146 with 146.0.
	var flagged: bool = meta.flagged.has(float(frame)) or meta.flagged.has(frame)
	var label: String = str(meta.labels[frame]) if frame < meta.labels.size() else ""
	hud.text = "%s\n%d/%d  z %.2f%s\n%s" % [
		str(meta.get("model", "trace")), frame + 1, n_frames,
		float(meta.z[frame]), "  FLAGGED" if flagged else "", label]


func _process(delta: float) -> void:
	if n_cells == 0:
		return
	if playing:
		accum += delta
		if accum >= 0.125:            # ~8 fps, matching viz/timeline.py
			accum = 0.0
			frame = (frame + 1) % n_frames
			_paint()
	# A slow sway in front of the field rather than a full orbit: from behind, the
	# billboards are the same picture mirrored, and edge-on the field vanishes.
	orbit += delta * 0.15
	var a := PI / 2.0 + 0.45 * sin(orbit)
	cam.position = center + Vector3(cos(a) * radius, 0.3 * radius, sin(a) * radius)
	cam.look_at(center, Vector3.UP)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventKey and e.pressed:
		match e.keycode:
			KEY_SPACE:
				playing = not playing
			KEY_LEFT:
				frame = (frame - 1 + n_frames) % n_frames
				_paint()
			KEY_RIGHT:
				frame = (frame + 1) % n_frames
				_paint()
			KEY_ESCAPE:
				get_tree().quit()
