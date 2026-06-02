extends CharacterBody2D

const GRAVITY := 980.0

@export var base_speed: float = 200.0
@export var jump_velocity: float = -1000.0

var can_move: bool = true
var _speed_multiplier: float = 1.0
var _jump_multiplier: float = 1.0


func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y += GRAVITY * delta

	if can_move and not DialogueManager.is_active:
		var direction := Input.get_axis("move_left", "move_right")
		velocity.x = direction * base_speed * _speed_multiplier
		if Input.is_action_just_pressed("jump") and is_on_floor():
			velocity.y = jump_velocity * _jump_multiplier
	else:
		velocity.x = move_toward(velocity.x, 0.0, base_speed * _speed_multiplier * delta * 8.0)

	move_and_slide()


func apply_movement_buff(speed_mult: float, jump_mult: float) -> void:
	_speed_multiplier = speed_mult
	_jump_multiplier = jump_mult


func set_can_move(enabled: bool) -> void:
	can_move = enabled
	if not enabled:
		velocity.x = 0.0
