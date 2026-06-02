extends CanvasLayer

@onready var panel: Panel = $Panel
@onready var dialogue_label: Label = $Panel/MarginContainer/VBoxContainer/DialogueLabel
@onready var hint_label: Label = $Panel/MarginContainer/VBoxContainer/HintLabel


func _ready() -> void:
	hide_dialogue()
	DialogueManager.register_ui(self)


func show_line(text: String) -> void:
	panel.visible = true
	dialogue_label.text = text
	hint_label.text = "E or Enter — next"


func hide_dialogue() -> void:
	panel.visible = false
	dialogue_label.text = ""
