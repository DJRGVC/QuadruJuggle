# Sim-to-Real Setup Notes

## Hardware

- **Robot:** Unitree Go1
- **Onboard compute:** `unitree@192.168.123.14` (main board, ARM64)
- **Raspberry Pi:** `pi@192.168.123.161` (peripherals only, no Docker)
- **Connection:** Ethernet cable directly from Ubuntu workstation to Go1

### IP map
| IP | Board | Notes |
|---|---|---|
| 192.168.123.10 | Motion controller | Alive, SSH works |
| 192.168.123.14 | Main compute (unitree-desktop) | Docker lives here |
| 192.168.123.161 | Raspberry Pi | No Docker, peripheral tasks only |

### Connecting from Ubuntu workstation
```bash
# Set static IP on your Ethernet interface first (if not already on 192.168.123.x)
sudo ip addr add 192.168.123.100/24 dev eth0

# SSH in
ssh unitree@192.168.123.14   # password: 123
```

### Connecting from Windows
```powershell
# Set static IP via Admin PowerShell
New-NetIPAddress -InterfaceAlias "Ethernet 5" -IPAddress 192.168.123.100 -PrefixLength 24

# SSH
ssh unitree@192.168.123.14
```

**Note:** VS Code Remote-SSH does not work on the Go1 onboard computer (ARM64, insufficient resources for VS Code Server). Use terminal only.

---

## Docker on the Robot

Docker is installed on `unitree@192.168.123.14`. Three relevant images are pre-pulled:

| Image | Size | Purpose |
|---|---|---|
| `qiayuanl/unitree:latest` | 1.93 GB | Main control stack (ROS 2, legged_controllers, OnnxController) |
| `gaoyuman/unitree-rl:latest` | 1.88 GB | Alternative RL deployment stack |
| `ros2-jazzy-ctrl:latest` | 1.09 GB | ROS 2 Jazzy control utilities |

### Backup the image (do this before any changes)
```bash
# On robot — save to disk
docker save qiayuanl/unitree:latest -o ~/unitree_backup.tar

# On workstation — copy to local machine
scp unitree@192.168.123.14:~/unitree_backup.tar ~/unitree_backup.tar
```

---

## Inside `qiayuanl/unitree:latest`

### Key compiled libraries
```
/opt/ros/humble/lib/liblegged_rl_controllers.so   — OnnxController plugin
/opt/ros/humble/lib/liblegged_controllers.so       — StateEstimator, StandbyController
/opt/ros/humble/lib/liblegged_estimation.so        — state estimation
/opt/ros/humble/lib/liblegged_model.so             — robot model
/opt/ros/humble/lib/libonnxruntime.so              — ONNX inference runtime
```

Source code is not in the image — packages are pre-compiled. The image runs ROS 2 Humble on Ubuntu 22.04 (ARM64).

### OnnxController internals (from symbol inspection)

Key methods in `legged::OnnxController`:
```
on_configure()      — loads .onnx file at startup
on_activate()       — activates when controller is switched to (e.g. R1+X on joystick)
getObservations()   — assembles obs vector from ros2_control hardware interfaces
playModel()         — runs onnxruntime Session::Run() with the obs vector
update()            — called at 50 Hz, orchestrates getObservations → playModel → write joint targets
```

### Command interface
`OnnxController` subscribes to `/cmd_vel` (`geometry_msgs/TwistStamped`) for the command slice of the observation vector. This is a **3D command** (linear x/y, angular z) matching the joystick walking policy.

**Implication for HIL juggling:** The stock `OnnxController` cannot directly accept a 6D torso command from π1. Options:
1. Ask Qiayuan (qiayuanl) if the command topic/dimension is configurable via `controllers.yaml`
2. Run π1+π2 as a standalone Python ROS 2 node and bypass `OnnxController` entirely
3. Export a combined π1→π2 ONNX (46D ball+robot state → 12D joints) that wraps both networks

---

## Software Stack Overview

```
real.launch.py
  ├── robot_state_publisher      (500 Hz, broadcasts TF tree from URDF)
  ├── ros2_control_node          (500 Hz hardware loop, loads controllers)
  │     ├── state_estimator      [active on boot]
  │     ├── standby_controller   [active on boot — safe PD pose]
  │     ├── walking_controller   [inactive — activate with R1+X]
  │     ├── getup_controller     [inactive — activate with L1+X]
  │     └── handstand_controller [inactive]
  └── teleop.launch.py
        ├── joy_linux_node       (reads gamepad)
        └── joy_teleop           (maps buttons → /cmd_vel + controller switching)
```

### Controller loop rates
| Loop | Rate | What it does |
|---|---|---|
| Hardware loop | 500 Hz | Reads sensors, writes motor commands via ros2_control |
| Policy inference | 50 Hz | `OnnxController::update()` → onnxruntime → joint targets |
| Robot state publisher | 500 Hz | Broadcasts TF frames |

### Joystick button map (Go1)
| Buttons | Action |
|---|---|
| L1 + left stick | Publish velocity on `/cmd_vel` |
| R1 + X | Activate `walking_controller` (your policy) |
| L1 + X | Activate `getup_controller` |
| Circle | E-stop — deactivate all controllers |

---

## π2 ONNX Export

Policy checkpoint → ONNX pipeline implemented in [scripts/tests/export_pi2_onnx.py](../scripts/tests/export_pi2_onnx.py).

```bash
python scripts/tests/export_pi2_onnx.py
# Output: sim_to_real/unitree_bringup/config/go1/pi2.onnx
```

π2 architecture: `39 → 256 → 128 → 12` (ELU activations, no final activation)
- Input `obs` [1, 39]: torso_cmd_norm(6), base_lin_vel(3), base_ang_vel(3), proj_gravity(3), joint_pos_rel(12), joint_vel(12)
- Output `actions` [1, 12]: joint position residuals (scaled by 0.25, added to default pose)

The `go1_joystick.onnx` that ships with `unitree_bringup` takes 48D input (3D `/cmd_vel` instead of 6D torso command) and is **not** the juggling policy.

---

## Contacts

- **Qiayuan Li** — author of `unitree_bringup`, `legged_rl_controllers`, Docker image. Contact for questions about `OnnxController` obs/command interface. Introduced via Hanqin Shi.
