extends Node2D

@onready var coin_hud: Label = $UI/CoinHUD


func _ready() -> void:
	var coin_count := get_tree().get_nodes_in_group("coin").size()
	GameState.reset_coin_tracking(coin_count)
	GameState.coins_changed.connect(_on_coins_changed)
	_on_coins_changed(GameState.coins_collected, GameState.total_coins)


func _on_coins_changed(collected: int, total: int) -> void:
	coin_hud.text = "Coins: %d/%d" % [collected, total]
	if GameState.checkpoint_complete:
		coin_hud.text += " — Checkpoint complete!"
