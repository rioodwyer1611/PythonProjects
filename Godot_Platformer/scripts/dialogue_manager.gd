extends Node

signal dialogue_finished
signal dialogue_started

var lines: Array[String] = []
var index: int = 0
var is_active: bool = false

var _ui: Node = null
var _ignore_next_advance: bool = false


func register_ui(ui: Node) -> void:
	_ui = ui
	if _ui.has_method("hide_dialogue"):
		_ui.hide_dialogue()


func start(new_lines: Array[String]) -> void:
	if is_active or new_lines.is_empty():
		return
	lines = new_lines.duplicate()
	index = 0
	is_active = true
	_ignore_next_advance = true
	dialogue_started.emit()
	_show_current_line()


func advance() -> void:
	if not is_active:
		return
	index += 1
	if index >= lines.size():
		end_dialogue()
	else:
		_show_current_line()


func end_dialogue() -> void:
	if not is_active:
		return
	is_active = false
	lines.clear()
	index = 0
	if _ui != null and _ui.has_method("hide_dialogue"):
		_ui.hide_dialogue()
	dialogue_finished.emit()


func _show_current_line() -> void:
	if _ui == null or not _ui.has_method("show_line"):
		return
	_ui.show_line(lines[index])


func _unhandled_input(event: InputEvent) -> void:
	if not is_active:
		return
	if _ignore_next_advance:
		return
	if event.is_action_pressed("interact") or event.is_action_pressed("ui_accept"):
		advance()
		get_viewport().set_input_as_handled()


func _process(_delta: float) -> void:
	if _ignore_next_advance:
		_ignore_next_advance = false
