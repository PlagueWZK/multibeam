# Multibeam Algorithm Score Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the current planner to a composite-objective algorithm that minimizes leakage first, then controls overlap and route length using unique-coverage metrics.

**Architecture:** Keep the current main-line plus perpendicular-expansion backbone, but add an evaluation layer that scores candidate lines by incremental unique coverage, overlap penalty, length cost, and shape cost. After the main pass, run a targeted hole-filling pass over under-covered regions and report all metrics using a unique-coverage accounting model.

**Tech Stack:** Python, NumPy, Pandas, Shapely, pytest

---

## Internal Grade Decision

- Grade: `XL`
- Reason: work spans algorithm evaluation, metric model, integration wiring, and verification; it benefits from staged execution with explicit checkpoints.

## Wave Structure

### Wave 1: Metric foundation
- Define unique-coverage accounting model
- Add coverage/score helper structures
- Add failing tests for metric bounds

### Wave 2: Candidate scoring
- Implement candidate-line evaluator
- Inject score-based acceptance into expansion flow
- Add tests for score ordering and rejection behavior

### Wave 3: Hole-filling pass
- Detect low-coverage frontier cells
- Seed targeted extra lines
- Verify leakage decreases or remains stable

### Wave 4: Integration and reporting
- Update Excel/reporting output
- Update README and algorithm docs
- Run verification and emit cleanup receipts

## Ownership Boundaries

- `multibeam/models.py`: scoring/metric dataclasses only
- `multibeam/planner_utils.py`: pure helper functions for coverage and scoring
- `multibeam/Planner.py`: orchestration and decision flow only
- `tests/`: deterministic checks for metrics and scoring

## Verification Commands

- `python -m pytest tests -q`
- `python main.py`

## Rollback Rules

- If candidate scoring breaks planning completion, disable score-gated acceptance but keep unique-coverage metrics
- If hole-filling causes instability, keep the main score-based planner and gate the hole-filling pass behind a default-on flag that can be disabled quickly

## Phase Cleanup Expectations

- Remove temporary debug prints before completion
- Leave runtime receipts under `outputs/runtime/vibe-sessions/20260408-algo-score-b/`
- Record verification status and unresolved limits explicitly

## Task Breakdown

### Task 1: Add failing tests for metric integrity

**Files:**
- Create: `tests/test_metrics.py`
- Modify: `multibeam/planner_utils.py`

**Steps:**
1. Write tests asserting unique coverage never exceeds total area and leakage never goes negative.
2. Run `python -m pytest tests/test_metrics.py -q` and confirm failure.
3. Implement minimal helper functions to support the assertions.
4. Re-run the same tests and confirm pass.

### Task 2: Add scoring data structures and helpers

**Files:**
- Modify: `multibeam/models.py`
- Modify: `multibeam/planner_utils.py`
- Create: `tests/test_scoring.py`

**Steps:**
1. Add dataclasses for score breakdown and coverage summary.
2. Add candidate scoring helpers for gain, overlap, length, and bend cost.
3. Add tests that compare a high-gain candidate against a high-overlap candidate.
4. Run `python -m pytest tests/test_scoring.py -q`.

### Task 3: Inject score-based acceptance into planner expansion

**Files:**
- Modify: `multibeam/Planner.py`
- Modify: `main.py`

**Steps:**
1. Thread score configuration through planner init.
2. Evaluate each generated candidate line before recording it.
3. Accept, reject, or prioritize candidates based on composite score.
4. Keep old expansion skeleton intact where possible.
5. Run targeted tests plus `python -m pytest tests -q`.

### Task 4: Add hole-filling pass

**Files:**
- Modify: `multibeam/Planner.py`
- Modify: `multibeam/planner_utils.py`
- Create: `tests/test_hole_filling.py`

**Steps:**
1. Detect uncovered or under-covered frontier cells from the accumulated coverage map.
2. Seed additional candidate lines from the worst cells.
3. Stop when improvement falls below threshold or no feasible candidate remains.
4. Add tests asserting the pass does not worsen leakage.
5. Run `python -m pytest tests/test_hole_filling.py -q`.

### Task 5: Update reporting and docs

**Files:**
- Modify: `multibeam/Planner.py`
- Modify: `README.md`

**Steps:**
1. Update Excel global metrics to include unique coverage, repeated coverage, and bounded leakage.
2. Document the new optimization strategy in `README.md`.
3. Run `python main.py` and inspect output paths and metrics.
