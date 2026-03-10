"""Generate a research overview PDF for the QuadruJuggle project."""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT = os.path.join(
    os.path.dirname(__file__), "..", ".claude", "QuadruJuggle_Research_Overview.pdf"
)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARGIN = 20
LINE_H = 7
SECTION_H = 10
INDENT = 8


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "QuadruJuggle - Research Overview", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def section(pdf: Doc, title: str):
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(230, 230, 250)
    pdf.cell(0, SECTION_H, f"  {title}", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)


def subsection(pdf: Doc, title: str):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, LINE_H, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)


def body(pdf: Doc, text: str):
    pdf.set_x(MARGIN)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(pdf.epw, LINE_H, text)


def bullet(pdf: Doc, text: str, indent: int = INDENT):
    pdf.set_x(MARGIN + indent)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(pdf.epw - indent, LINE_H, f"-  {text}")
    pdf.set_x(MARGIN)


def table_row(pdf: Doc, col1: str, col2: str,
              col1_w: float = 55, fill: bool = False):
    if fill:
        pdf.set_fill_color(240, 240, 248)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 9)
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.multi_cell(col1_w, LINE_H, col1, border=1, fill=True,
                   new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_xy(x + col1_w, y)
    pdf.multi_cell(0, LINE_H, col2, border=1, fill=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def table_header(pdf: Doc, h1: str, h2: str, col1_w: float = 55):
    pdf.set_fill_color(60, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(col1_w, LINE_H, f"  {h1}", border=1, fill=True)
    pdf.cell(0, LINE_H, f"  {h2}", border=1, fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

pdf = Doc(orientation="P", unit="mm", format="A4")
pdf.set_margins(MARGIN, MARGIN, MARGIN)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()


# ── Title page ───────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 30, 100)
pdf.ln(8)
pdf.cell(0, 12, "QuadruJuggle", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "Teaching a Quadruped to Juggle", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 7, "Research Overview - February 2026", align="C",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_text_color(0, 0, 0)
pdf.ln(6)

pdf.set_draw_color(100, 100, 180)
pdf.set_line_width(0.5)
pdf.line(MARGIN, pdf.get_y(), 210 - MARGIN, pdf.get_y())
pdf.ln(8)


# ── 1. Overview ──────────────────────────────────────────────────────────────
section(pdf, "1.  Project Overview")
body(pdf,
     "QuadruJuggle trains a Unitree Go1 quadruped to balance and juggle a ping-pong "
     "ball on a 170 mm disc paddle fixed to its back.  The robot cannot touch the ball "
     "directly; it controls ball position by modulating trunk posture via 12-DOF joint "
     "targets.  No arm, no gripper - the body is the actuator.")

pdf.ln(2)
body(pdf,
     "Current focus: Stage 1 - static ball balancing from full ground-truth state.  "
     "The full five-stage roadmap is in Section 9.")


# ── 2. Prior Work ────────────────────────────────────────────────────────────
section(pdf, "2.  Prior Work - Velocity Tracking")
body(pdf,
     "Before this project the primary task was standard quadrupedal locomotion: training "
     "the Go1 to track commanded linear and angular velocities on flat and procedurally "
     "generated rough terrain.")

subsection(pdf, "Task")
body(pdf,
     "Policy input: target velocity (forward, lateral, yaw) plus proprioception "
     "(joint positions, velocities, foot contact forces, IMU).  "
     "Actions: 12-DOF joint position targets.  "
     "Rewards: velocity tracking error, joint torques, foot slip, body orientation.")

subsection(pdf, "Terrain curriculum")
body(pdf,
     "Agents started on flat ground and were promoted to rougher terrain "
     "(stepped, sloped, random heightfield) as performance improved.  "
     "This is the canonical Isaac Lab starting point and produced the PPO "
     "hyperparameters used here.")

subsection(pdf, "Relevance")
body(pdf,
     "The locomotion policy will be re-introduced in Stage 3, where the robot "
     "must balance the ball while simultaneously tracking velocity commands.")


# ── 3. Current Task - Static Ball Balance ────────────────────────────────────
section(pdf, "3.  Current Task - Static Ball Balance")
body(pdf,
     "A 170 mm diameter disc paddle is tracked to the robot trunk via a kinematic "
     "rigid body updated at 200 Hz (every physics step).  A 40 mm ping-pong ball "
     "(2.7 g) is spawned above the paddle at each episode reset and must stay on.")

subsection(pdf, "Key dimensions")
table_header(pdf, "Parameter", "Value", col1_w=70)
for i, (k, v) in enumerate([
    ("Paddle diameter",        "170 mm  (radius 85 mm)"),
    ("Paddle offset from trunk","70 mm above trunk origin (body frame)"),
    ("Ball diameter",          "40 mm  (radius 20 mm)"),
    ("Ball mass",              "2.7 g  (ITTF standard)"),
    ("Ball restitution",       "0.85 with combine_mode=max (see Section 5.3)"),
    ("Episode length",         "30 s  (1 500 steps at 50 Hz policy rate)"),
    ("Physics timestep",       "5 ms  (200 Hz); policy decimation = 4"),
    ("Spawn XY std (Stage A)", "30 mm Gaussian offset from paddle centre"),
    ("Spawn drop height (A)",  "80 mm above paddle surface (mean), std 20 mm"),
    ("Termination: ball off",  "XY distance > 300 mm from paddle centre"),
    ("Termination: ball below","ball drops > 50 mm below paddle surface"),
]):
    table_row(pdf, f"  {k}", f"  {v}", col1_w=70, fill=(i % 2 == 0))
pdf.ln(2)

subsection(pdf, "Observation space  (39 dimensions)")
for obs in [
    "ball_pos_in_paddle_frame  (3) - ball XYZ relative to paddle centre",
    "ball_vel_in_paddle_frame  (3) - ball linear velocity in trunk frame",
    "base_lin_vel              (3) - trunk linear velocity (body frame)",
    "base_ang_vel              (3) - trunk angular velocity (body frame)",
    "projected_gravity         (3) - gravity vector in body frame",
    "joint_pos_rel            (12) - joint positions minus default pose",
    "joint_vel                (12) - joint velocities",
]:
    bullet(pdf, obs)

subsection(pdf, "Actions")
bullet(pdf, "12-DOF joint position targets  (scale 0.25 rad, offset from default pose)")


# ── 4. Software Stack ────────────────────────────────────────────────────────
section(pdf, "4.  Software Stack")
table_header(pdf, "Component", "Details", col1_w=50)
for i, (k, v) in enumerate([
    ("Simulator",      "NVIDIA Isaac Lab 2.x / Isaac Sim - GPU rigid-body simulation"),
    ("RL algorithm",   "RSL-RL PPO (OnPolicyRunner)"),
    ("Robot",          "Unitree Go1 - 12 DOF, nominal trunk height 0.40 m"),
    ("Training scale", "12 288 parallel envs, env spacing 3.0 m"),
    ("Policy arch.",   "MLP 256->128->64  ELU activations; matching value network"),
    ("Package mgmt.",  "uv  (uv run --active python scripts/rsl_rl/train.py)"),
    ("OS / GPU",       "Ubuntu 24.04, NVIDIA GPU, CUDA 12.1+"),
    ("Logging",        "TensorBoard - logs/rsl_rl/go1_ball_balance/"),
]):
    table_row(pdf, f"  {k}", f"  {v}", col1_w=50, fill=(i % 2 == 0))
pdf.ln(2)


# ── 5. Reward Function ───────────────────────────────────────────────────────
section(pdf, "5.  Reward Function")
body(pdf,
     "Three problems required diagnosis during reward design: the die-fast local "
     "minimum, loss of ball centering gradient, and incorrect bounce physics.")

subsection(pdf, "5.1  Die-fast minimum")
body(pdf,
     "base_height weight -20 accumulated to ~-10 000 over 500 steps, making "
     "early termination optimal.  TensorBoard showed 14-step episodes, 99 % "
     "ball_below terminations.")
body(pdf,
     "Fix: alive +1/step makes longer episodes always better.  "
     "early_termination -200 (one-shot on ball_off/ball_below) calibrated per "
     "Zhuang et al. (CoRL 2023): penalty >> episode_length x alive_rate.  "
     "base_height weight reduced to -5.")

subsection(pdf, "5.2  Loss of centering gradient")
body(pdf,
     "The ball rolled steadily toward the robot's front along the spine.  "
     "Root cause: the Gaussian centering kernel (std=0.25 m) is nearly flat "
     "once the ball is 5-10 cm from centre - exp(-(0.1m)^2 / (2*0.25^2)) = 0.92.  "
     "The policy sees almost no gradient to chase the ball back.")
body(pdf,
     "Fix: ball_xy_dist_penalty (Ji et al., ICRA 2023) provides a constant linear "
     "gradient at all offsets.  At 10 cm from centre, weight -2.0 gives -0.20/step - "
     "sufficient to override drift without interfering with standing.")

subsection(pdf, "5.3  Incorrect bounce physics")
body(pdf,
     "Ball restitution was set to 0.85, but bounce height was ~4-5x too low.  "
     "The paddle is loaded from a USD file (disc.usda).  UsdFileCfg does not accept "
     "a physics_material argument, so the paddle used PhysX default restitution 0.0.  "
     "PhysX computes effective restitution as the average of both bodies: "
     "(0.85 + 0.0) / 2 = 0.43.")
body(pdf,
     "Fix: set restitution_combine_mode='max' on the ball's RigidBodyMaterialCfg.  "
     "PhysX then uses max(ball_r, paddle_r) = max(0.85, 0.0) = 0.85 regardless of "
     "the paddle's default material.  No USD file edit required.")
body(pdf,
     "Real-life equivalent surface at r ~ 0.85: carbon-fibre plate (1-2 mm thick), "
     "acrylic sheet (3 mm), or melamine-coated MDF (standard ping-pong table top).")

subsection(pdf, "5.4  Current reward table")
table_header(pdf, "Term", "Weight  |  Notes", col1_w=48)
for i, (k, v) in enumerate([
    ("alive",             " +1.0  |  +1 every step; makes longer episodes strictly better"),
    ("base_height",       " -5.0  |  linear penalty when trunk < 0.34 m"),
    ("early_termination", "-200.0 |  one-shot on ball_off / ball_below termination"),
    ("ball_on_paddle",    " +8.0  |  Gaussian 3-D centering, std=0.25 m, time_scale=4"),
    ("ball_lateral_vel",  " -1.0  |  ball XY speed, height-gated"),
    ("ball_xy_dist",      " -2.0  |  linear XY dist from paddle centre; constant gradient"),
    ("trunk_tilt",        " -2.0  |  projected gravity XY magnitude (roll/pitch)"),
    ("body_lin_vel",      " -0.10 |  trunk linear speed"),
    ("body_ang_vel",      " -0.05 |  trunk angular speed"),
    ("action_rate",       " -0.01 |  L2 norm of action delta"),
    ("joint_torques",     " -2e-4 |  joint effort"),
]):
    table_row(pdf, f"  {k}", f"  {v}", col1_w=48, fill=(i % 2 == 0))
pdf.ln(2)
body(pdf,
     "ball_on_paddle uses a 3-D Gaussian from the ideal resting point "
     "(paddle centre + ball_radius * paddle_normal).  A height gate (trunk < 0.34 m "
     "-> reward = 0) prevents a collapsed robot from earning ball reward.  "
     "time_scale=4 linearly grows the reward from 1x to 5x over the episode, "
     "favouring sustained balance over short lucky stretches.")


# ── 6. Spawn + Sigma Curriculum ──────────────────────────────────────────────
section(pdf, "6.  Spawn and Sigma Curriculum")
body(pdf,
     "Training uses an automatic four-stage curriculum that advances when the policy "
     "sustains time_out% >= 85 % for 20 consecutive iterations.  Spawn difficulty "
     "and Gaussian sigma tighten in lock-step.  The curriculum is one-way "
     "(no rollback) and implemented as a monkey-patch on runner.log() in "
     "scripts/rsl_rl/train.py, requiring no changes to the Isaac Lab environment.")

pdf.ln(2)
table_header(pdf, "Stage", "xy_std  |  drop height  |  sigma (ball_on_paddle std)", col1_w=22)
for i, (stage, vals) in enumerate([
    ("A  (default)", "30 mm  |  80 mm mean, 20 mm std  |  250 mm"),
    ("B",            "60 mm  |  150 mm mean, 30 mm std |  150 mm"),
    ("C",            "100 mm |  300 mm mean, 50 mm std |  100 mm"),
    ("D",            "150 mm |  600 mm mean, 80 mm std |   80 mm"),
]):
    table_row(pdf, f"  {stage}", f"  {vals}", col1_w=22, fill=(i % 2 == 0))
pdf.ln(2)

body(pdf,
     "Stage D drop height of 600 mm gives ball arrival speed "
     "sqrt(2 * 9.81 * 0.60) ~ 3.4 m/s - comparable to a gentle human throw.  "
     "As sigma tightens from 250 mm to 80 mm the Gaussian kernel provides a "
     "proportionally stronger centering gradient, matching the ROGER (RSS 2025) "
     "sigma curriculum pattern.")

subsection(pdf, "Implementation")
for b in [
    "Trigger: time_out fraction extracted from RSL-RL ep_infos each iteration.",
    "Advance: above_count reaches 20 -> stage += 1, above_count resets to 0.",
    "Param mutation: event_manager term params (xy_std, drop_height_mean, "
    "drop_height_std) and reward_manager term param (std) updated in-place.  "
    "Takes effect on the next episode reset.",
    "Status printed every iteration: stage, iter count in stage, sustain counter, "
    "current time_out% vs threshold.",
]:
    bullet(pdf, b)


# ── 7. Academic Literature ───────────────────────────────────────────────────
section(pdf, "7.  Relevant Literature")
body(pdf,
     "Citations below are restricted to work that directly shaped design decisions "
     "in this project.  Venue abbreviations: ICRA, RSS, CoRL, IROS = robotics "
     "conference proceedings; Science Robotics = journal.")

subsection(pdf, "7.1  Collapse prevention: survival bonus and terminal penalty")
for b in [
    "Rudin et al., 'Learning to Walk in Minutes Using Massively Parallel Deep RL' "
    "(CoRL 2022, ETH RSL) - RSL-RL / legged_gym; established is_alive +1/step as "
    "the primary survival signal.  All subsequent ETH RSL papers inherit this.",
    "Zhuang et al., 'Robot Parkour Learning' (CoRL 2023, UCB) - one-shot fall "
    "penalty calibrated so penalty >> episode_length x per_step_reward.  This "
    "calibration rule set the early_termination weight of -200 here.",
    "Ji et al., 'DribbleBot: Dynamic Legged Manipulation in the Wild' (ICRA 2023, "
    "MIT) - quadruped soccer-ball dribbling; the most directly analogous task.  "
    "Same survival + terminal structure, Gaussian proximity reward, body-vel penalty.",
    "Caluwaerts et al., 'CaT: Constraints as Terminations' (IROS 2024) - replaces "
    "large penalty weights with probabilistic episode termination on constraint "
    "violation.  Adopted by ETH RSL's SoloParkour.  An alternative to the "
    "base_height penalty if die-fast re-emerges.",
    "Huang et al., 'SoloParkour' (CoRL 2024 Best Paper, ETH RSL) - survival bonus "
    "combined with CaT-style terminations; most stable collapse-prevention result "
    "in current literature.",
]:
    bullet(pdf, b)

subsection(pdf, "7.2  Task reward: Gaussian centering and sigma curriculum")
for b in [
    "Ji et al., DribbleBot (ICRA 2023) - ball proximity reward exp(-d^2/sigma^2), "
    "sigma=0.30 m, height-gated to prevent kick-away exploit.  Source of "
    "ball_on_paddle_exp and ball_xy_dist_penalty designs here.",
    "Portela et al., 'ROGER' (RSS 2025, Oxford Robotics) - sigma curriculum on a "
    "Gaussian proximity kernel: std decays 0.50 -> 0.15 -> 0.08 m as "
    "mean_episode_length stabilises.  Directly adopted here with stages A-D.",
    "Hoeller et al., 'ANYmal Parkour' (Science Robotics 2024, ETH RSL) - binary "
    "height gate on task reward prevents posture-violation exploit.  Matches the "
    "min_height gate in ball_on_paddle_exp.",
    "Cheng et al., 'Extreme Parkour' (ICRA 2024, CMU) - multiplicative penalty "
    "gating: r_task * exp(-lambda * sum_penalty).  Suppresses task signal when "
    "posture is poor.  Candidate for Phase 2 if ball reward begins dominating "
    "over standing.",
]:
    bullet(pdf, b)

subsection(pdf, "7.3  Body motion regularisation")
body(pdf,
     "The following lightweight terms appear consistently across all surveyed papers "
     "at weights 0.001 - 0.01 relative to the primary task signal:")
for b in [
    "action_rate_l2: L2 norm of (action_t - action_{t-1}).  Prevents joint "
    "chattering.  Universal across all papers; weight ~0.01.",
    "joint_torques_l2: reduces hardware wear and energy use.  Weight 1e-4 to 1e-3.",
    "body_lin_vel / body_ang_vel: DribbleBot (Ji 2023) used these to prevent "
    "the robot from swinging its trunk to influence the ball through inertia.  "
    "Critical for Stages 1-2 here for the same reason.",
    "trunk_tilt via projected_gravity XY: Margolis & Agrawal, 'Walk These Ways' "
    "(CoRL 2022, MIT) - more numerically stable than quaternion deviation; "
    "continuous and bounded in [-1, 1].",
]:
    bullet(pdf, b)


# ── 8. Training Results ──────────────────────────────────────────────────────
section(pdf, "8.  Training Results")
body(pdf,
     "As of late February 2026, the Phase 2 reward structure (Section 5) is running "
     "with the Stage A curriculum.  Key metrics at ~700 training iterations:")

for item in [
    "time_out: 87-91 % - ball stays on paddle for the full 30 s in ~9 of 10 episodes",
    "ball_below: ~7 %, ball_off: ~3 % - rare terminations",
    "action_rate penalty: -1.1/step - robot making active corrective adjustments",
    "mean_action_noise_std: ~2.0 - policy still exploring; has not converged prematurely",
    "alive reward: growing monotonically, confirming stable standing",
]:
    bullet(pdf, item)

pdf.ln(2)
body(pdf,
     "Outstanding changes since last evaluation: ball_xy_dist_penalty added (fixes "
     "forward drift), restitution_combine_mode='max' applied (fixes under-bounce), "
     "four-stage curriculum installed.  These have not yet run long enough for "
     "published metrics.")

subsection(pdf, "Known failure modes addressed")
table_header(pdf, "Symptom", "Root cause  ->  Fix", col1_w=52)
for i, (s, f) in enumerate([
    ("14-step episodes, 99% ball_below",
     "base_height -20 dominated; die-fast optimal  ->  alive +1, early_term -200, base_height -5"),
    ("Ball rolls to robot front",
     "Gaussian flat tail past 5cm; no centering pull  ->  ball_xy_dist penalty -2.0"),
    ("Bounce height 4-5x too low",
     "UsdFileCfg no physics_material; paddle r=0.0, avg=0.43  ->  combine_mode=max on ball"),
    ("TypeError on launch",
     "physics_material kwarg invalid on UsdFileCfg  ->  removed; combine_mode fix instead"),
]):
    table_row(pdf, f"  {s}", f"  {f}", col1_w=52, fill=(i % 2 == 0))
pdf.ln(2)


# ── 9. Research Roadmap ───────────────────────────────────────────────────────
section(pdf, "9.  Research Roadmap")
body(pdf,
     "Five stages, each adding one new challenge.  Current work is Stage 1.")

pdf.ln(2)

stages = [
    (
        "Stage 1  (current) - Static Balance, Privileged State",
        [
            "Robot stands; ball spawned above paddle and must stay on.",
            "Observations: full ground-truth ball position and velocity.",
            "Algorithm: PPO, 12 288 parallel envs, four-stage spawn curriculum.",
            "Success criterion: time_out > 85 % for 50+ consecutive iterations.",
        ]
    ),
    (
        "Stage 2  - Active Bouncing, Privileged State",
        [
            "Reward restructured to require rhythmic ball contact rather than passive balance.",
            "New terms: ball_apex_height (consistent bounce height), contact_rate.",
            "Policy must learn a periodic joint trajectory - structurally different "
            "from the static Stage 1 posture.",
        ]
    ),
    (
        "Stage 3  - Bouncing Under Velocity Tracking",
        [
            "Re-introduce velocity command from locomotion baseline.",
            "Robot must walk, steer, and juggle simultaneously.",
            "Terrain curriculum (flat -> rough) re-applied.",
        ]
    ),
    (
        "Stage 4  - Egocentric Camera, Teacher-Student",
        [
            "Upward-facing camera on robot back replaces ground-truth ball observations.",
            "Teacher policy (Stage 2/3) distilled to student via DAgger.",
            "Student uses CNN or DINOv2 encoder; teacher provides labels on visited states.",
            "Domain randomisation: ball colour, lighting, camera noise for sim-to-real.",
        ]
    ),
    (
        "Stage 5  - Full Integration",
        [
            "Camera perception (Stage 4) + locomotion + juggling (Stage 3).",
            "No privileged state at inference.",
            "Target deployment: physical Go1, Ubuntu 24.04.",
            "Evaluation: bounce count per metre, robustness to terrain perturbations.",
        ]
    ),
]

for stage_title, bullets in stages:
    subsection(pdf, stage_title)
    for b in bullets:
        bullet(pdf, b)
    pdf.ln(1)


# ── 10. PPO to DAgger ─────────────────────────────────────────────────────────
section(pdf, "10.  Algorithm Transition: PPO to DAgger")
body(pdf,
     "Stages 1-3 use PPO throughout.  Stage 4 introduces DAgger (Ross et al., "
     "AISTATS 2011) for perception distillation.")

for b in [
    "The Stage 1-3 teacher policy is a stable oracle that can label any state the "
    "student encounters online - the core DAgger requirement.",
    "Behaviour cloning alone fails under distributional shift when the student "
    "visits states outside the teacher's demonstration set.  DAgger corrects this "
    "by querying the teacher on student-visited states iteratively.",
    "The camera encoder (e.g., DINOv2 fine-tuned on synthetic ball/paddle imagery) "
    "can be pre-trained and frozen; only a small adapter head is trained via DAgger.",
]:
    bullet(pdf, b)


# ── 11. Key References ────────────────────────────────────────────────────────
section(pdf, "11.  References")
for ref in [
    "Isaac Lab - https://isaac-sim.github.io/IsaacLab/",
    "Rudin et al., 'Learning to Walk in Minutes Using Massively Parallel Deep RL', "
    "CoRL 2022, ETH RSL",
    "Margolis & Agrawal, 'Walk These Ways', CoRL 2022, MIT",
    "Ji et al., 'DribbleBot: Dynamic Legged Manipulation in the Wild', ICRA 2023, MIT",
    "Zhuang et al., 'Robot Parkour Learning', CoRL 2023, UCB",
    "Cheng et al., 'Extreme Parkour with Legged Robots', ICRA 2024, CMU",
    "Hoeller et al., 'ANYmal Parkour: Learning Agile Navigation', "
    "Science Robotics 2024, ETH RSL",
    "Caluwaerts et al., 'CaT: Constraints as Terminations for Legged Locomotion "
    "Reinforcement Learning', IROS 2024",
    "Huang et al., 'SoloParkour: Constrained Reinforcement Learning for Visual "
    "Locomotion', CoRL 2024 Best Paper, ETH RSL",
    "Portela et al., 'ROGER: Reinforcement Learning for Object Grasping and "
    "Re-Orientation', RSS 2025, Oxford Robotics",
    "Ross et al., 'A Reduction of Imitation Learning and Structured Prediction "
    "to No-Regret Online Learning', AISTATS 2011  [DAgger]",
    "Oquab et al., 'DINOv2: Learning Robust Visual Features without Supervision', "
    "TMLR 2024",
    "Unitree Go1 - 12 DOF quadruped, ~12 kg, nominal standing height 0.40 m",
]:
    bullet(pdf, ref)

pdf.ln(4)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 6, "QuadruJuggle  -  February 2026",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# ---------------------------------------------------------------------------
pdf.output(OUTPUT)
print(f"PDF written to: {OUTPUT}")
