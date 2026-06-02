extends StaticBody2D

@export var dialogue_lines: Array[String] = [
	"Welcome, traveler.",
	"The path ahead is dangerous.",
	"Collect every coin if you want my blessing.",
]

@onready var interaction_zone: Area2D = $InteractionZone
@onready var prompt_label: Label = $PromptLabel

var player_in_range: bool = false


func _ready() -> void:
	interaction_zone.body_entered.connect(_on_body_entered)
	interaction_zone.body_exited.connect(_on_body_exited)
	DialogueManager.dialogue_started.connect(_on_dialogue_started)
	DialogueManager.dialogue_finished.connect(_on_dialogue_finished)
	_update_prompt()


func _unhandled_input(event: InputEvent) -> void:
	if DialogueManager.is_active:
		return
	if not player_in_range:
		return
	if event.is_action_pressed("interact"):
		_start_dialogue()
		get_viewport().set_input_as_handled()


func _start_dialogue() -> void:
	var lines := _build_dialogue_lines()
	DialogueManager.start(lines)


func _build_dialogue_lines() -> Array[String]:
	var lines := dialogue_lines.duplicate()
	if GameState.checkpoint_complete:
		if not GameState.buff_applied:
			lines.append("You found every coin. Take this speed blessing!")
		else:
			lines.append("Your blessing is already with you. Go forth!")
	else:
		lines.append("Come back when you have found all the coins.")
	return lines


func _on_body_entered(body: Node2D) -> void:
	if body.is_in_group("player"):
		player_in_range = true
		_update_prompt()


func _on_body_exited(body: Node2D) -> void:
	if body.is_in_group("player"):
		player_in_range = false
		_update_prompt()


func _on_dialogue_started() -> void:
	var player := get_tree().get_first_node_in_group("player")
	if player != null and player.has_method("set_can_move"):
		player.set_can_move(false)
	_update_prompt()


func _on_dialogue_finished() -> void:
	var player := get_tree().get_first_node_in_group("player")
	if player != null and player.has_method("set_can_move"):
		player.set_can_move(true)
	if GameState.checkpoint_complete and not GameState.buff_applied:
		GameState.grant_movement_buff(player)
	_update_prompt()


func _update_prompt() -> void:
	if DialogueManager.is_active:
		prompt_label.visible = false
	elif player_in_range:
		prompt_label.text = "Press E to talk"
		prompt_label.visible = true
	else:
		prompt_label.visible = false
