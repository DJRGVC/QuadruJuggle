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
LINE_H = 7       # standard line height
SECTION_H = 10   # section heading line height
INDENT = 8


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "QuadruJuggle - Research Overview", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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
    pdf.cell(0, SECTION_H, f"  {title}", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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


def table_row(pdf: Doc, col1: str, col2: str, col1_w: float = 55, fill: bool = False):
    if fill:
        pdf.set_fill_color(240, 240, 248)
    else:
        pdf.set_fill_color(255, 255, 255)
    pdf.set_font("Helvetica", "", 9)
    x = pdf.get_x()
    y = pdf.get_y()
    pdf.multi_cell(col1_w, LINE_H, col1, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_xy(x + col1_w, y)
    pdf.multi_cell(0, LINE_H, col2, border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def table_header(pdf: Doc, h1: str, h2: str, col1_w: float = 55):
    pdf.set_fill_color(60, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(col1_w, LINE_H, f"  {h1}", border=1, fill=True)
    pdf.cell(0, LINE_H, f"  {h2}", border=1, fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------

pdf = Doc(orientation="P", unit="mm", format="A4")
pdf.set_margins(MARGIN, MARGIN, MARGIN)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()


# ── Title page ──────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 30, 100)
pdf.ln(8)
pdf.cell(0, 12, "QuadruJuggle", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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


# ── 1. Overview ─────────────────────────────────────────────────────────────
section(pdf, "1.  Project Overview")
body(pdf,
     "QuadruJuggle is a reinforcement learning research project aimed at training a "
     "Unitree Go1 quadruped robot to perform dynamic object manipulation - specifically, "
     "to balance and eventually actively juggle a ping-pong ball on a small disc paddle "
     "mounted to its back.  The robot cannot contact the ball directly; instead it "
     "modulates its posture and gait to keep the ball centred on the paddle - using its "
     "body as the actuator.")

pdf.ln(2)
body(pdf,
     "The broader research arc progresses through five increasingly demanding stages, "
     "described in Section 7.  The current work focuses on Stage 1: static ball balancing "
     "from privileged state observations.")


# ── 2. Prior work ───────────────────────────────────────────────────────────
section(pdf, "2.  Prior Work - Velocity Tracking")
body(pdf,
     "Before this project, the primary experimental task was standard quadrupedal "
     "locomotion: training the Go1 to track commanded linear and angular velocities on "
     "both smooth flat terrain and procedurally generated rough terrain.")

subsection(pdf, "Task description")
body(pdf,
     "The policy received a target velocity command (forward, lateral, and yaw) and "
     "learned to track it using the 12-DOF joint position interface.  Observations "
     "included proprioception (joint positions, velocities, foot contact forces, IMU "
     "accelerations/gyroscope) together with the velocity command.  Rewards penalised "
     "deviation from the commanded velocity, large joint torques, foot-slipping, and "
     "body orientation error.")

subsection(pdf, "Terrain curriculum")
body(pdf,
     "Training used a terrain curriculum: agents began on flat ground, and as their "
     "performance improved they were promoted to progressively rougher procedural terrain "
     "(stepped, sloped, and randomised heightfield patches).  This curriculum is the "
     "canonical starting point for legged locomotion research with Isaac Lab and provides "
     "a robust, well-understood baseline.")

subsection(pdf, "Relevance to current work")
body(pdf,
     "The velocity-tracking work established familiarity with the Isaac Lab / RSL-RL "
     "training pipeline and the Go1 asset, and produced a well-tuned set of PPO "
     "hyperparameters that served as the starting point for the ball-balance task.  "
     "The locomotion policy will eventually be re-introduced in Stage 3 (Section 7), "
     "where the robot must balance the ball while simultaneously tracking velocity "
     "commands.")


# ── 3. Current task ─────────────────────────────────────────────────────────
section(pdf, "3.  Current Task - Static Ball Balance")
body(pdf,
     "A 170 mm diameter disc paddle (radius 85 mm) is rigidly tracked to the robot's "
     "trunk via a kinematic rigid body that is repositioned every policy step.  A "
     "40 mm diameter ping-pong ball (mass 2.7 g) is spawned 100 mm above the paddle "
     "centre at the start of each episode with a small Gaussian XY offset (sigma = 50 mm) "
     "to prevent the policy from learning a degenerate perfectly-centred solution.")

subsection(pdf, "Observation space  (39 dimensions)")
for obs in [
    "ball_pos_in_paddle_frame  (3) - XYZ of ball relative to paddle centre",
    "ball_vel_in_paddle_frame  (3) - linear velocity of ball in trunk frame",
    "base_lin_vel              (3) - trunk linear velocity (body frame)",
    "base_ang_vel              (3) - trunk angular velocity (body frame)",
    "projected_gravity         (3) - gravity vector in body frame (tilt proxy)",
    "joint_pos_rel            (12) - joint positions minus default pose",
    "joint_vel                (12) - joint velocities",
]:
    bullet(pdf, obs)

subsection(pdf, "Action space")
bullet(pdf, "12-DOF joint position targets  (scaled by 0.25, offset from default pose)")

subsection(pdf, "Termination conditions")
for t in [
    "ball_off_paddle  - ball XY distance from paddle centre exceeds 300 mm",
    "ball_below_paddle - ball drops more than 50 mm below paddle surface",
    "time_out  - episode length exceeds 10 seconds (500 steps at 50 Hz)",
]:
    bullet(pdf, t)


# ── 4. Software stack ────────────────────────────────────────────────────────
section(pdf, "4.  Software Stack")

table_header(pdf, "Component", "Details", col1_w=50)
rows = [
    ("Simulator",      "NVIDIA Isaac Lab 2.x / Isaac Sim - GPU-accelerated rigid-body simulation"),
    ("Algorithm",      "RSL-RL PPO (Phase 1); DAgger / Behaviour Cloning planned for Phase 2"),
    ("Robot",          "Unitree Go1 - 12 DOF (3 per leg), nominal trunk height ~ 0.40 m"),
    ("Training scale", "12 288 parallel environments, ~500-step episodes"),
    ("Policy arch.",   "MLP  256 -> 128 -> 64  (ELU activations); matching critic"),
    ("Vision (future)","CNN or DINOv2 encoder from on-board upward-facing camera"),
    ("Package mgmt.",  "uv - all runs via  uv run --active python scripts/rsl_rl/train.py"),
    ("OS / GPU",       "Ubuntu 24.04, NVIDIA Ampere+ GPU, CUDA 12.1+"),
    ("Logging",        "TensorBoard - logs/rsl_rl/go1_ball_balance/"),
]
for i, (k, v) in enumerate(rows):
    table_row(pdf, f"  {k}", f"  {v}", col1_w=50, fill=(i % 2 == 0))
pdf.ln(2)


# ── 5. Reward function ───────────────────────────────────────────────────────
section(pdf, "5.  Reward Function Design")
body(pdf,
     "Designing the reward function for this task required resolving several competing "
     "objectives and eliminating a sequence of degenerate local minima.  The key "
     "difficulty is that the robot must simultaneously learn two hard sub-tasks - "
     "standing upright, and keeping an object balanced on its back - neither of which "
     "provides useful gradient for the other in the early stages of training.")

subsection(pdf, "5.1  The die-fast local minimum")
body(pdf,
     "In early experiments, the base_height penalty (applied whenever the trunk dropped "
     "below a threshold) had a large weight (-20).  Accumulated over 500 steps, this "
     "penalty far outweighed the ball reward, making it optimal for the policy to "
     "collapse in a few steps and terminate early.  TensorBoard showed 14-step episodes "
     "with 99 % ball_below_paddle terminations.")
body(pdf,
     "Fix: introduce a height-gated survival bonus (alive_upright, +12/step at "
     "current tuning) that makes longer episodes strictly better.  The early_termination "
     "term adds a one-shot penalty (-5) when the ball leaves the paddle, making dying "
     "also costly.  The base_height weight was reduced to -10.")

subsection(pdf, "5.2  Active-ball-control instability")
body(pdf,
     "Once the policy learned to stand, ball_on_paddle (the Gaussian centering reward) "
     "was weighted equally to alive (both 8.0 at the time).  After ~40 iterations the "
     "policy began making active corrective movements in response to ball position, and "
     "these movements destabilised standing.  Both termination rates rose simultaneously, "
     "indicating the robot was over-correcting in multiple directions.")
body(pdf,
     "Fix: reduce ball_on_paddle weight (8.0 -> 1.5) so alive (12.0) is 8× dominant.  "
     "An action_rate_l2 penalty (-0.01) discourages large rapid joint changes without "
     "freezing the policy entirely.")

subsection(pdf, "5.3  Policy escape from good basin")
body(pdf,
     "With the above fixes the policy reached 80 % episode-timeout (ball on paddle for "
     "the full 10 s) but then slowly degraded: time_out declined from 0.80 to 0.40 "
     "over 60 iterations before collapsing.  The adaptive LR schedule (desired_kl=0.01) "
     "grew the effective learning rate during the stable phase until a single rollout "
     "pushed the policy out of the good basin.")
body(pdf,
     "Fix: tighten PPO constraints - desired_kl 0.01 -> 0.004, clip_param 0.2 -> 0.1, "
     "num_learning_epochs 5 -> 3, max_grad_norm 1.0 -> 0.5.  These changes limit how "
     "far any single update can move the policy, preventing the adaptive LR from "
     "destabilising a well-converged solution.")

subsection(pdf, "5.4  Uncontrollable ball dynamics from initial drop")
body(pdf,
     "The ball was originally spawned 200 mm above the paddle (restitution = 0.85), "
     "producing chaotic multi-bounce trajectories that the robot could not influence "
     "through joint control.  The ball_height and ball_lateral_vel penalties then "
     "provided large gradient pointing in essentially random directions.")
body(pdf,
     "Fix: reduce drop height to 100 mm and restitution to 0.5.  This cuts the initial "
     "kinetic energy by ~75 % and significantly reduces chaotic bounce drift, making the "
     "ball penalty gradients more informative.")

subsection(pdf, "5.5  Current reward table")
table_header(pdf, "Term", "Weight  |  Notes", col1_w=45)
reward_rows = [
    ("alive_upright",      "+12.0  |  Height-gated survival - primary signal"),
    ("early_termination",  " -5.0  |  One-shot penalty on ball_off / ball_below"),
    ("ball_on_paddle_exp", " +1.5  |  Gaussian XY centering, std=0.30 m, height-gated"),
    ("ball_lateral_vel",   " -0.5  |  Penalise ball sliding (height-gated)"),
    ("ball_height",        " -0.5  |  Penalise ball airborne above paddle"),
    ("base_height",        "-10.0  |  Penalise trunk below 0.28 m"),
    ("trunk_tilt",         " -1.0  |  Gravity XY magnitude (roll/pitch proxy)"),
    ("body_lin_vel",       " -0.01 |  Light trunk motion penalty"),
    ("body_ang_vel",       "-0.005 |  Light trunk rotation penalty"),
    ("action_rate_l2",     " -0.01 |  Smooth joint commands"),
    ("joint_torques_l2",   "-2e-4  |  Joint effort regularisation"),
]
for i, (k, v) in enumerate(reward_rows):
    table_row(pdf, f"  {k}", f"  {v}", col1_w=45, fill=(i % 2 == 0))
pdf.ln(2)


# ── 6. Training results so far ───────────────────────────────────────────────
section(pdf, "6.  Training Results So Far")
body(pdf,
     "After several iterations of reward shaping, the policy reliably reaches 80 % "
     "episode-timeout (ball remaining on paddle for the full 10 s in 8 out of 10 "
     "episodes) within 20-30 training iterations (~30 min wall-clock).  The key "
     "metrics observed in TensorBoard:")

for item in [
    "mean_episode_length peaks at ~450 steps (out of 500 maximum)",
    "Episode_Termination/time_out reaches 0.80, indicating robust ball retention",
    "Episode_Termination/ball_off_paddle and ball_below_paddle both near 0.05",
    "Episode_Reward/alive monotonically increasing, confirming stable standing",
    "Episode_Reward/trunk_tilt small (< 0.02), confirming upright posture",
]:
    bullet(pdf, item)

pdf.ln(2)
body(pdf,
     "The remaining engineering challenge is preventing the policy from slowly drifting "
     "out of the good basin after the initial peak.  The current PPO constraint tightening "
     "(desired_kl = 0.004, clip = 0.1) is the most recent intervention and is still "
     "being evaluated.")


# ── 7. Research roadmap ──────────────────────────────────────────────────────
section(pdf, "7.  Research Roadmap")
body(pdf,
     "The project is organised as a staged curriculum in both task complexity and "
     "perception modality.  Each stage adds exactly one new challenge on top of the "
     "previous, allowing the reward function and training dynamics to be understood "
     "cleanly before compounding difficulty.")

pdf.ln(2)

stages = [
    (
        "Stage 1  (current) - Static Ball Balancing, Privileged State",
        [
            "Robot stands still; ball spawned above paddle and must stay on.",
            "Policy receives full ground-truth ball position and velocity.",
            "Algorithm: PPO with RSL-RL, 12 288 parallel envs.",
            "Success criterion: > 80 % episode timeout, stable for 200+ iterations.",
        ]
    ),
    (
        "Stage 2 - Active Ball Bouncing, Privileged State",
        [
            "Reward restructured to incentivise controlled juggling: the robot must "
            "actively strike the ball upward and catch it repeatedly.",
            "Requires learning a rhythmic, periodic joint policy - fundamentally "
            "different from the static balance posture.",
            "Ball restitution will be increased back toward 0.85 once bouncing is rewarded.",
            "New reward terms: ball_apex_height (reward for achieving consistent bounce "
            "height), ball_contact_rate (reward for rhythmic contact).",
        ]
    ),
    (
        "Stage 3 - Ball Bouncing Under Velocity Tracking",
        [
            "Re-introduce the velocity command from prior locomotion work.",
            "The policy must simultaneously maintain a commanded forward/lateral/yaw "
            "velocity while keeping the ball bouncing on the paddle.",
            "Terrain curriculum (flat -> rough) re-applied as in prior velocity-tracking work.",
            "This is the most mechanically demanding stage: the robot must walk, steer, "
            "and juggle concurrently.",
        ]
    ),
    (
        "Stage 4 - Egocentric Perception Module",
        [
            "A small upward-facing fisheye camera is mounted on the robot's back, "
            "directed at the paddle and ball.",
            "A teacher policy (Stage 1 or 2, privileged state) is distilled into a "
            "student policy via DAgger or Behaviour Cloning.",
            "The student replaces ball_pos / ball_vel observations with a latent "
            "embedding from a CNN or DINOv2 encoder.",
            "Domain randomisation applied to ball colour, lighting, background, and "
            "camera noise to support sim-to-real transfer.",
        ]
    ),
    (
        "Stage 5 - Full Integration: Perception + Bouncing + Velocity Tracking",
        [
            "Combine the egocentric perception module (Stage 4) with the full "
            "locomotion + juggling task (Stage 3).",
            "All observations are either proprioceptive or camera-derived - no "
            "privileged state at inference time.",
            "Target deployment: physical Unitree Go1 on Ubuntu 24.04.",
            "Evaluation: ball bounce count per metre of travel, robustness to "
            "terrain perturbations, and sim-to-real gap measurement.",
        ]
    ),
]

for stage_title, bullets in stages:
    subsection(pdf, stage_title)
    for b in bullets:
        bullet(pdf, b)
    pdf.ln(1)


# ── 8. PPO -> DAgger transition ───────────────────────────────────────────────
section(pdf, "8.  Algorithm Transition: PPO to DAgger")
body(pdf,
     "Phase 1 (Stages 1-3) uses PPO throughout because it is well-suited to continuous "
     "control with dense rewards and does not require demonstrations.  The transition to "
     "DAgger (Dataset Aggregation) is motivated by Stage 4, where the perception module "
     "must be trained efficiently without re-running expensive PPO rollouts from scratch.")

subsection(pdf, "Why DAgger for the perception stage")
for b in [
    "The Stage 1-3 teacher policy provides a stable, high-quality oracle that can "
    "label any state the student encounters - the core requirement for DAgger.",
    "BC (Behaviour Cloning) alone suffers from distributional shift when the student "
    "visits states not seen in the teacher's demonstrations.  DAgger iteratively "
    "queries the teacher on student-visited states, correcting this drift.",
    "The camera encoder can be pre-trained (e.g., DINOv2 fine-tuned on synthetic "
    "ball/paddle imagery) and frozen, with only a small adapter head trained via DAgger.",
    "DAgger's data efficiency is critical because GPU memory is shared between the "
    "Isaac Sim physics engine and the vision encoder.",
]:
    bullet(pdf, b)

pdf.ln(2)
body(pdf,
     "An alternative under consideration is GAIL or adversarial imitation, which could "
     "allow the student to learn from teacher trajectories without requiring online "
     "teacher queries.  This would be advantageous if the teacher policy is expensive "
     "to run in parallel with the camera encoder during training.")


# ── 9. Key references ────────────────────────────────────────────────────────
section(pdf, "9.  Key References & Tools")
for ref in [
    "Isaac Lab - https://isaac-sim.github.io/IsaacLab/  (simulation + RL framework)",
    "RSL-RL - Rudin et al., Learning to Walk in Minutes (CoRL 2022)",
    "DAgger - Ross et al., A Reduction of Imitation Learning (AISTATS 2011)",
    "DINOv2 - Oquab et al., DINOv2: Learning Robust Visual Features (2023)",
    "GaussGym - Gaussian Splatting + IsaacGym, candidate for visual sim-to-real",
    "Unitree Go1 - 12 DOF quadruped, 12 kg, nominal standing height 0.40 m",
]:
    bullet(pdf, ref)

pdf.ln(4)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 6, "Generated by Claude Code  ·  QuadruJuggle project  ·  February 2026",
         align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

# ---------------------------------------------------------------------------
pdf.output(OUTPUT)
print(f"PDF written to: {OUTPUT}")
