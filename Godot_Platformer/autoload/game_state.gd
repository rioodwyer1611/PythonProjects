extends Node

signal coins_changed(collected: int, total: int)
signal checkpoint_reached

var total_coins: int = 0
var coins_collected: int = 0
var checkpoint_complete: bool = false
var buff_applied: bool = false

const BUFF_SPEED_MULTIPLIER := 1.4
const BUFF_JUMP_MULTIPLIER := 1.4


func reset_coin_tracking(total: int) -> void:
	total_coins = total
	coins_collected = 0
	checkpoint_complete = total == 0
	coins_changed.emit(coins_collected, total_coins)


func register_coin() -> void:
	if checkpoint_complete:
		return
	coins_collected += 1
	coins_changed.emit(coins_collected, total_coins)
	if coins_collected >= total_coins:
		checkpoint_complete = true
		checkpoint_reached.emit()


func grant_movement_buff(player: Node) -> void:
	if buff_applied or player == null:
		return
	if not player.has_method("apply_movement_buff"):
		return
	player.apply_movement_buff(BUFF_SPEED_MULTIPLIER, BUFF_JUMP_MULTIPLIER)
	buff_applied = true
