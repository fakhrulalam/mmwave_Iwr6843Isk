
## Slide 1 - Title

Title:
Social Behavior Inference and Interaction Mapping using mmWave Radar

Content:
- Your name, department, advisor, university
- Date and committee members (if needed)

Speaker notes:
- Thank the committee.
- State this is a proposal defense, so focus is: problem significance, technical plan, feasibility, and expected contributions.

---

## Slide 2 - Outline

Content:
- Motivation and problem statement
- Research gap and objectives
- Literature review
- Radar fundamentals and system design
- Data collection and feature engineering
- Model design and preliminary results
- Future plan and expected outcomes

Speaker notes:
- Tell the audience exactly how the talk is organized.
- Mention that preliminary results validate feasibility.

---

## Slide 3 - Introduction: Why this problem matters

Content:
- Social behavior understanding is important for:
  - Elderly care and assisted living
  - Smart spaces and occupancy analytics
  - Safety monitoring and anomaly detection
- Existing methods often rely on cameras or wearables.

Speaker notes:
- Emphasize practical impact and real-world need.

---

## Slide 4 - Why mmWave Radar

Content:
- Privacy-preserving sensing (no identity-revealing images)
- Works in low light and visually cluttered scenes
- Captures motion dynamics (range, Doppler, angle)
- Potentially robust for continuous ambient sensing

Speaker notes:
- Position mmWave as a strong trade-off between privacy and sensing quality.

---

## Slide 5 - Human Activity Recognition (HAR) context

Content:
- HAR identifies activities such as walking, sitting, standing, and gestures.
- Most radar HAR studies are single-person focused.
- Social behavior inference requires understanding interactions among multiple people.

Speaker notes:
- Bridge from standard HAR to your proposed social behavior focus.

---

## Slide 6 - Concept of the proposed system

Content:
Pipeline figure labels:
- Radar sensing -> signal processing -> features -> temporal model -> social behavior prediction

Target social classes (example):
- Group standing
- Group walking together
- Group sitting/discussion
- Approaching and splitting

Speaker notes:
- Explain this as the end-to-end conceptual architecture.

---

## Slide 7 - Problem Statement and Research Gap

Content:
Current limitations:
- Vision systems raise privacy concerns.
- Radar studies mostly classify single-person activities.
- Limited robust work on multi-person social behavior using mmWave radar.

Research gap:
- Lack of a reliable framework for real-time social interaction mapping using only mmWave radar.

Research question:
- Can mmWave radar features and temporal models reliably infer social behavior classes in multi-person settings?

---

## New Literature Review Sequence (Use as 4 consecutive slides)

### Literature Review Slide 1 - mmDoppler-Based Single-Person HAR (Primary Reference)

Recommended paper focus:
- mm Doppler paper in this folder (use as your opening reference)

Textbox (3 bullets):
- What they did: Demonstrated single-person activity classification using radar Doppler or range-Doppler representations and a deep learning classifier.
- Limitation: The task is activity-centric for one person, so interaction-level context (between people) is not modeled.
- Scope for our work: This directly supports our first milestone, where we already achieved single-person activity classification with range-Doppler data.

Show 2 figures:
- Figure A: One figure from the mm Doppler paper showing the signal representation or model pipeline.
- Figure B: One result figure from the mm Doppler paper (accuracy, confusion matrix, or per-class performance).

---

### Literature Review Slide 2 - Efficient Single-Person Radar HAR (Z. Li et al., 2023)

Recommended paper:
- Z. Li et al., Scientific Reports, 2023 (adaptive thresholding for radar HAR)

Textbox (3 bullets):
- What they did: Used micro-Doppler with adaptive thresholding and efficient deep learning for robust single-person HAR.
- Limitation: Focused on individual motion classes and efficiency, not group interaction semantics.
- Scope for our work: Justifies our range-Doppler preprocessing and motivates an efficient pipeline for future real-time deployment.

Show 2 figures:
- Figure A: ROI extraction or adaptive thresholding workflow from the paper.
- Figure B: Accuracy and compute-efficiency comparison figure (or confusion matrix).

---

### Literature Review Slide 3 - Toward Multi-Person Radar Understanding (A. K. Alhazmi et al., 2024)

Recommended paper:
- A. K. Alhazmi et al., 2024 (mmWave human monitoring with positional and Doppler features)

Textbox (3 bullets):
- What they did: Leveraged spatial features (x, y, z), Doppler, and learning-based classification for radar human monitoring.
- Limitation: Still does not present a generalized social behavior taxonomy across realistic multi-person interaction classes.
- Scope for our work: Bridges from single-person classification to richer spatial-temporal descriptors needed for multi-person social activity recognition.

Show 2 figures:
- Figure A: System setup or feature pipeline (point cloud/position plus Doppler).
- Figure B: Reported classification performance figure from the paper.

---

### Literature Review Slide 4 - Multi-Person Feasibility and Our Research Position (F. Jin et al., 2019 -> Our Goal)

Recommended paper:
- F. Jin et al., IEEE Radar Conference, 2019 (multiple-person behavior detection)

Textbox (3 bullets):
- What they did: Showed feasibility of multi-person radar behavior detection using point-cloud tracking and Doppler-informed deep models.
- Limitation: Prior multi-person studies are narrow in behavior scope and scenario design; our thesis is not proposing a healthcare-only system.
- Scope for our work: We now show successful results on multi-person activity using our own research data and target generalized social behaviors (approaching, splitting, standing together, walking together, sitting discussion).

Show 2 figures:
- Figure A: Multi-target tracking or interaction scene figure from the paper.
- Figure B: Your own preliminary multi-person result figure (confusion matrix/class-wise F1) to smoothly transition into your research goal slide.

---

## Slide 8 - Literature Review (Part 1)

Content structure (table):
- Study
- Sensor and method
- Main findings
- Limitations relative to your work

Talk points:
- Abdullah et al. (2024): high accuracy, mostly single-person context.
- Li et al. (2023): efficient micro-Doppler HAR, not social interaction-focused.

Speaker notes:
- Be fair: acknowledge strengths, then clearly identify unmet needs.

---

## Slide 9 - Literature Review (Part 2)

Talk points:
- Jin et al. (2019): multi-patient behavior detection in healthcare context.
- Limitation: domain-specific setting, not generalized social behavior taxonomy.

Synthesis statement:
- Existing studies prove radar feasibility, but generalized, multi-class social interaction inference remains underexplored.

---

## Slide 10 - Research Goal and Objectives

Main goal:
- Develop and validate a mmWave-based framework for social behavior inference and interaction mapping.

Objectives:
- Build a repeatable data acquisition pipeline.
- Engineer informative frame-level and temporal features.
- Train and compare temporal deep learning models (LSTM, CNN-based baseline).
- Evaluate generalization across social behavior classes.

Expected contribution:
- A practical privacy-preserving social behavior recognition pipeline.

---

## Slide 11 - mmWave Radar Fundamentals

Content:
- FMCW chirp transmission and reflected signal processing
- Range estimation via FFT
- Velocity (Doppler) from phase/frequency shift across chirps
- Angle of arrival (AoA) from antenna phase differences

Useful equation block for explanation:
- Range resolution: Delta R = c / (2B)
- Doppler frequency relation: f_d = 2v / lambda

Speaker notes:
- Keep equations intuitive; focus on physical meaning.

---

## Slide 12 - Sensor Setup Workflow

Content:
- Install mmWave SDK and required firmware
- Flash board with TI people-tracking firmware
- Configure radar profile using custom cfg files
- Parse UART data stream into usable TLV payloads

Speaker notes:
- Stress reproducibility and custom parsing capability.

---

## Slide 13 - Configuration Parameters

Content (table):
- Frequency, bandwidth, chirp duration, frame rate
- Field-of-view, antenna configuration
- Range and Doppler resolutions

Speaker notes:
- Explain parameter choices from the lens of your target use-case (indoor social interactions).

---

## Slide 14 - Experimental Setup

Content:
- Sensor mounting height and tilt
- Power and UART communication settings
- Physical environment constraints and participant zones

Speaker notes:
- Mention controls for consistency (distance zones, movement boundaries, repeated trials).

---

## Slide 15 - Data Structure and Parsing

Content:
- Frame header and sequential frame indexing
- TLV payloads parsed:
  - Point cloud
  - Target list
- Extracted attributes: x/y/z, velocity components, SNR, confidence, track ID

Speaker notes:
- Show this slide as the basis of your data engineering validity.

---

## Slide 16 - System Design Overview

Content:
- Sensor setup
- Data acquisition
- Feature vectorization
- Model preparation
- Preliminary evaluation

Speaker notes:
- This is your full proposal pipeline in one slide.

---

## Slide 17 - Single Activity Baseline (Range-Doppler)

Content:
- Example classes: sitting, standing, walking, waving
- Representative range-Doppler snapshots
- Purpose: baseline verification before social behavior modeling

Speaker notes:
- Explain why single-activity competence is a prerequisite.

---

## Slide 18 - Preliminary Results (Single activity)

Content:
- Report baseline metrics (accuracy/F1/confusion matrix)
- Highlight strongest and weakest classes

Speaker notes:
- If exact values are not finalized, mark as preliminary and include confidence interval plan.

---

## Slide 19 - Point Cloud Based Recognition

Content:
- Point-cloud temporal pattern examples
- Advantages over pure range-Doppler in social contexts

Speaker notes:
- Emphasize spatial relationship cues among multiple subjects.

---

## Slide 20 - Preliminary Results (Point cloud model)

Content:
- Validation curves and class-wise performance
- Error examples and observed failure modes

Speaker notes:
- This demonstrates honest scientific analysis and readiness for proposal stage.

---

## Slide 21 - Social Activity Results

Content:
- Initial social classes predicted (standing together, sitting discussion, approaching, splitting, walking together)
- Early qualitative and quantitative outcomes

Speaker notes:
- Position as proof-of-feasibility, not final thesis claim.

---

## Slide 22 - Model Preparation and Training Setup

Content (comparison table):
- LSTM vs 1D-CNN baseline
- Data split, optimizer, learning rate, epochs, early stopping, batch size
- Loss function and output layer choice

Speaker notes:
- Explain why temporal models are natural for radar frame sequences.

---

## Slide 23 - Feature Vectorization

Content:
Frame-level features:
- Centroid features: x_centroid, y_centroid, z_centroid
- Spread features: x_range, y_range, y_variance
- Group descriptors: point_count, low_height_ratio
- Motion descriptors: moving_point_ratio, mean_velocity

Interpretation examples:
- Approaching vs splitting from y_centroid/y_range trends
- Sitting discussion via low_height_ratio
- Static vs moving groups via mean_velocity and moving_point_ratio

Speaker notes:
- This is a key contribution slide. Explain feature-to-behavior mapping clearly.

---

## Slide 24 - Future Work

Content:
- Real-time inference pipeline deployment
- Expand social behavior taxonomy and micro-activities
- Multi-radar fusion for improved robustness
- Cross-environment generalization tests

Speaker notes:
- Tie future work to thesis milestones.

---

## Slide 25 - References

Content:
- Use consistent citation format (IEEE style)
- Verify complete venue, volume, and page details
- Include only sources actually cited in slides

Speaker notes:
- Keep this clean and committee-friendly.

---

## Slide 26 - Closing / Q&A

Content:
- Thank you
- I welcome questions and suggestions

Speaker notes:
- Optional final line:
  "My proposal is ready for committee feedback on scope, evaluation protocol, and expected thesis contributions."

---

## Extra: Committee-Focused Backup Slide Ideas (Optional)

- Risk and mitigation table
  - Class imbalance -> weighted loss / augmentation
  - Overfitting -> regularization and cross-validation
  - Domain shift -> leave-one-setting-out testing
- Proposed thesis timeline (month-by-month)
- Evaluation protocol details
  - Metrics: macro-F1, confusion matrix, per-class recall
  - Validation strategy: subject-independent and setting-aware splits

---

## Quick Delivery Tips for 40-minute Defense

- Keep 1 key message per slide.
- Spend more time on slides 7, 10, 15, 22, 23 (core proposal value).
- Be explicit about what is done vs. what is proposed next.
- Pre-answer likely questions:
  - Why these classes?
  - Why these features?
  - How will you validate generalization?
  - What is the novelty over existing radar HAR studies?
