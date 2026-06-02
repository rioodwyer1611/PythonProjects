extends Area2D


func _ready() -> void:
	add_to_group("coin")
	body_entered.connect(_on_body_entered)


func _on_body_entered(body: Node2D) -> void:
	if not body.is_in_group("player"):
		return
	GameState.register_coin()
	queue_free()
