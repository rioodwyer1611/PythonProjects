# Godot Platformer

A minimal Godot 4.2+ 2D platformer with coin collection, NPC dialogue, and a movement buff gated behind collecting all coins before talking to the NPC.

## Requirements

- [Godot 4.2](https://godotengine.org/download) or newer

## Run

1. Open Godot and **Import** the `Godot_Platformer` folder (select `project.godot`).
2. Press **F5** to run the main scene (`scenes/main.tscn`).

## Controls

| Action | Keys |
|--------|------|
| Move left / right | A / D or arrow keys |
| Jump | Space, W, or Up |
| Talk / advance dialogue | E (also Enter to advance lines) |

## Gameplay

1. Move and jump across platforms; collect all **6 coins** (HUD shows progress).
2. Walk up to the **purple NPC** on the right platform. When close, **Press E to talk** appears.
3. Press **E** to open dialogue; press **E** or **Enter** to advance lines.
4. If you collected **all coins before finishing dialogue**, you receive a **movement buff** (faster run and higher jump) when the conversation ends.
5. If you talk before collecting all coins, dialogue still works but you get no buff until you return after the checkpoint.

## Customize

- **Dialogue lines**: edit `dialogue_lines` on the NPC in `scenes/npc.tscn` or in `scripts/npc.gd`.
- **Buff strength**: change `BUFF_SPEED_MULTIPLIER` and `BUFF_JUMP_MULTIPLIER` in `autoload/game_state.gd`.
- **Player speed / jump**: edit exports on the Player scene (`base_speed`, `jump_velocity`).
- **Coins**: add or remove `Coin` instances in `scenes/main.tscn` (they use group `"coin"`; count is automatic).

## Project layout

```
Godot_Platformer/
├── autoload/game_state.gd    # Coins, checkpoint, buff
├── scripts/
│   ├── dialogue_manager.gd   # Dialogue flow (autoload)
│   ├── dialogue_ui.gd
│   ├── player.gd
│   ├── npc.gd
│   ├── coin.gd
│   └── main.gd
└── scenes/
	├── main.tscn
	├── player.tscn
	├── npc.tscn
	├── coin.tscn
	└── dialogue_ui.tscn
```

## Physics layers

| Layer | Name | Usage |
|-------|------|--------|
| 1 | world | Ground and platforms |
| 2 | player | Player body |
| 3 | interaction | NPC interaction zone |
